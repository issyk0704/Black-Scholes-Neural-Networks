"""Market data from Yahoo Finance, cached as CSV files.

Price history is stored the way Yahoo returns it with ``auto_adjust=False``:
split-adjusted but not dividend-adjusted. :func:`unadjusted_close` recovers the
price that actually traded on a day that comes before a later split.

Each option-chain snapshot is saved with the market inputs that applied when it
was taken (spot, risk-free rate, dividend yield, historical volatility), so a
dataset built from old snapshots never needs to go back to the network.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from bsnn import instruments, paths

log = logging.getLogger(__name__)

DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL", "NVDA")
DEFAULT_PERIOD = "2y"
DEFAULT_RATE = 0.04  # fallback when the T-bill yield can't be fetched
TRADING_DAYS = 252
VOL_WINDOW = 30
MARKET_TZ = "America/New_York"

HISTORY_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
SNAPSHOT_COLUMNS = ["ticker", "snapshotTime", "underlyingPrice", "riskFreeRate", "dividendYield", "histVol"]


# --- Price history -----------------------------------------------------------

def market_dates(values) -> pd.DatetimeIndex:
    """Parse timestamps (with or without UTC offsets) into New York calendar dates."""
    if isinstance(values, pd.DatetimeIndex) and values.tz is not None:
        idx = values.tz_convert(MARKET_TZ).tz_localize(None)
    else:
        text = pd.Index(values).astype(str)
        if text.str.contains(r"[+-]\d{2}:\d{2}$").any():
            idx = pd.to_datetime(text, utc=True, format="mixed").tz_convert(MARKET_TZ).tz_localize(None)
        else:
            idx = pd.to_datetime(text, format="mixed")
    return pd.DatetimeIndex(idx).normalize()


def market_date(timestamp) -> pd.Timestamp:
    """The New York calendar date of a single timestamp."""
    ts = pd.Timestamp(timestamp)
    if ts.tz is not None:
        ts = ts.tz_convert(MARKET_TZ).tz_localize(None)
    return ts.normalize()


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = market_dates(df.index)
    df.index.name = "Date"
    df = df[[c for c in HISTORY_COLUMNS if c in df.columns]].dropna(subset=["Close"])
    for col in ("Dividends", "Stock Splits"):
        if col not in df.columns:
            df[col] = 0.0
    return df[~df.index.duplicated(keep="last")].sort_index()


def history_path(ticker: str, directory: Path | None = None) -> Path:
    safe = ticker.upper().replace(":", "_")  # "FRED:DGS2" isn't a valid Windows file name
    return Path(directory or paths.STOCK_DIR) / f"{safe}_data.csv"


FRED_PREFIX = "FRED:"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
PERIOD_OFFSETS = {"mo": "months", "y": "years"}


def _period_start(period: str) -> pd.Timestamp:
    """'6mo' -> six months ago, '2y' -> two years ago."""
    for suffix, unit in PERIOD_OFFSETS.items():
        if period.endswith(suffix):
            return pd.Timestamp.today().normalize() - pd.DateOffset(**{unit: int(period[:-len(suffix)])})
    raise ValueError(f"Unsupported period {period!r}")


def _fetch_fred(series: str, period: str, start) -> pd.DataFrame:
    """A FRED series (e.g. DGS2, the 2-year Treasury yield) shaped like a Yahoo price history.

    FRED publishes one value a day, a business day late, so Open/High/Low equal Close.
    """
    raw = pd.read_csv(FRED_URL.format(series=series), na_values=".").dropna()
    values = raw.set_index(pd.to_datetime(raw.iloc[:, 0])).iloc[:, 1].astype(float)
    values = values.loc[pd.Timestamp(start) if start else _period_start(period):]
    return pd.DataFrame({"Open": values, "High": values, "Low": values, "Close": values, "Volume": 0.0})


def fetch_history(ticker: str, period: str = DEFAULT_PERIOD, start=None, directory: Path | None = None,
                  save: bool = True) -> pd.DataFrame:
    """Download daily history from Yahoo, or from FRED for a ``FRED:<series>`` ticker.

    ``start`` overrides ``period``.
    """
    if ticker.upper().startswith(FRED_PREFIX):
        raw = _fetch_fred(ticker[len(FRED_PREFIX):].upper(), period, start)
    else:
        tk = yf.Ticker(ticker)
        raw = tk.history(start=start, auto_adjust=False) if start else tk.history(period=period, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No price history returned for {ticker!r}")
    df = clean_history(raw)
    if save:
        path = history_path(ticker, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, date_format="%Y-%m-%d")
    return df


def load_history(ticker: str, directory: Path | None = None) -> pd.DataFrame:
    return clean_history(pd.read_csv(history_path(ticker, directory), index_col=0))


def get_history(ticker: str, refresh: bool = False, period: str = DEFAULT_PERIOD,
                directory: Path | None = None) -> pd.DataFrame:
    """Price history from the cache, downloading it when asked to or when nothing is cached.

    Only :data:`DEFAULT_PERIOD` downloads are written to the cache, so a short
    download never replaces a longer cached history. If a download fails and a
    cached copy exists, the cached copy is returned.
    """
    if refresh or not history_path(ticker, directory).exists():
        try:
            return fetch_history(ticker, period=period, directory=directory, save=period == DEFAULT_PERIOD)
        except Exception:
            if not history_path(ticker, directory).exists():
                raise
            log.warning("Fetching %s failed; using cached history", ticker, exc_info=True)
    return load_history(ticker, directory)


def unadjusted_close(history: pd.DataFrame, asof=None) -> float:
    """Close on the last trading day at or before ``asof``, undoing any later splits."""
    upto = history if asof is None else history.loc[:market_date(asof)]
    if upto.empty:
        raise ValueError(f"No price history on or before {asof}")
    later_splits = history.loc[history.index > upto.index[-1], "Stock Splits"]
    factor = later_splits[later_splits > 0].prod()
    return float(upto["Close"].iloc[-1] * factor)


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Annualised rolling standard deviation of daily log returns."""
    return np.log(close).diff().rolling(window).std() * np.sqrt(TRADING_DAYS)


def dividend_yield(history: pd.DataFrame, asof=None) -> float:
    """Trailing 12-month dividends divided by the close on ``asof`` (default: latest day)."""
    asof = history.index[-1] if asof is None else market_date(asof)
    window = history.loc[asof - pd.Timedelta(days=365):asof]
    if window.empty:
        return 0.0
    close = window["Close"].iloc[-1]
    return float(window["Dividends"].sum() / close) if close > 0 else 0.0


@lru_cache(maxsize=64)
def _risk_free_rate(date_key: str | None) -> float:
    irx = yf.Ticker("^IRX")
    if date_key is None:
        hist = irx.history(period="1mo")
    else:
        day = pd.Timestamp(date_key)
        hist = irx.history(start=day - pd.Timedelta(days=14), end=day + pd.Timedelta(days=1))
    closes = hist["Close"].dropna() if not hist.empty else hist
    if closes.empty:
        raise ValueError("No ^IRX data")
    return float(closes.iloc[-1]) / 100


def risk_free_rate(asof=None) -> float:
    """13-week US T-bill yield (Yahoo ^IRX) as a decimal, on or before ``asof``.

    Falls back to :data:`DEFAULT_RATE` when the yield can't be fetched.
    """
    key = None if asof is None else market_date(asof).strftime("%Y-%m-%d")
    try:
        return _risk_free_rate(key)
    except Exception:
        log.warning("Couldn't fetch the T-bill yield; using %.2f%%", DEFAULT_RATE * 100, exc_info=True)
        return DEFAULT_RATE


def foreign_rate_from_futures(spot: float, futures_price: float, years_to_expiry: float,
                              domestic_rate: float) -> float:
    """Foreign interest rate implied by covered interest parity: F = S * exp((r_dom - r_for) * T)."""
    if min(spot, futures_price, years_to_expiry) <= 0:
        raise ValueError("Spot, futures price and time to expiry must be positive")
    return domestic_rate - np.log(futures_price / spot) / years_to_expiry


def implied_foreign_rate(spot: float, futures_ticker: str, domestic_rate: float) -> float:
    """Foreign rate implied by Yahoo's front-month FX futures (e.g. 6E=F for EUR)."""
    tk = yf.Ticker(futures_ticker)
    expiry = pd.Timestamp(tk.info["expireDate"], unit="s", tz="UTC")
    futures_price = float(clean_history(tk.history(period="5d"))["Close"].iloc[-1])
    years = (expiry - pd.Timestamp.now(tz="UTC")).total_seconds() / (365 * 24 * 3600)
    return foreign_rate_from_futures(spot, futures_price, years, domestic_rate)


# --- Option chains -----------------------------------------------------------

SECONDS_PER_YEAR = 365 * 24 * 3600


def years_to_expiry(snapshot_time, expiry_date):
    """Years from the snapshot to 4pm New York time on the expiry date."""
    snap = pd.to_datetime(pd.Series(np.asarray(snapshot_time)), utc=True, format="mixed")
    expiry = pd.to_datetime(pd.Series(np.asarray(expiry_date).astype(str)))
    expiry = (expiry + pd.Timedelta(hours=16)).dt.tz_localize(MARKET_TZ)
    years = (expiry.dt.tz_convert("UTC") - snap).dt.total_seconds() / SECONDS_PER_YEAR
    return years.to_numpy()


def parity_dividend_yield(chain: pd.DataFrame, spot: float, rate: float, snapshot_time,
                          min_days: float = 14, max_days: float = 120) -> float:
    """Dividend yield implied by put-call parity on near-the-money pairs.

    For European options C - P = exp(-rT) * (F - K), which gives the forward F
    and so the yield q = r - ln(F/S)/T. Used for cash indices (SPX, NDX, RUT),
    whose Yahoo price history has no dividends. Returns 0.0 if no pairs qualify.
    """
    quoted = chain[(chain["bid"] > 0) & (chain["ask"] > 0)]
    mids = quoted.assign(mid=(quoted["bid"] + quoted["ask"]) / 2)
    # The third Friday lists both the AM-settled monthly (SPX) and the PM weekly (SPXW) at the same strikes: one mid each.
    calls = mids[mids["OptionType"] == "Call"].groupby(["ExpiryDate", "strike"])["mid"].mean()
    puts = mids[mids["OptionType"] == "Put"].groupby(["ExpiryDate", "strike"])["mid"].mean()
    pairs = pd.concat({"call": calls, "put": puts}, axis=1, join="inner").reset_index()
    if pairs.empty:
        return 0.0
    T = years_to_expiry([pd.Timestamp(snapshot_time).isoformat()] * len(pairs), pairs["ExpiryDate"])
    near = ((T * 365 >= min_days) & (T * 365 <= max_days) & (abs(pairs["strike"] / spot - 1) < 0.03)).to_numpy()
    if not near.any():
        return 0.0
    forward = pairs["strike"][near] + np.exp(rate * T[near]) * (pairs["call"][near] - pairs["put"][near])
    return float(np.median(rate - np.log(forward / spot) / T[near]))


def is_cash_index(ticker: str) -> bool:
    """Yahoo's cash indices (^SPX, ^NDX, ^RUT) start with '^' and have European options."""
    return ticker.startswith("^")


def option_snapshot_path(ticker: str, snapshot_time, directory: Path | None = None) -> Path:
    """Where a snapshot is saved. New snapshots are gzipped (a full day is ~10 MB as plain CSV)."""
    day = market_date(snapshot_time)
    return Path(directory or paths.OPTIONS_DIR) / f"{ticker.upper()}_options_{day:%Y-%m-%d}.csv.gz"


def save_option_snapshot(chain: pd.DataFrame, directory: Path | None = None) -> Path:
    first = chain.iloc[0]
    path = option_snapshot_path(first["ticker"], first["snapshotTime"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_csv(path, index=False)
    return path


def snapshot_label(path: Path) -> str:
    """'SPY  2026-10-05' for SPY_options_2026-10-05.csv(.gz)."""
    ticker, _, rest = Path(path).name.partition("_options_")
    return f"{ticker}  {rest.split('.')[0]}"


def enrich_snapshot(chain: pd.DataFrame, ticker: str, snapshot_time, *, spot: float,
                    history: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Attach the market inputs that applied at ``snapshot_time`` to every contract.

    A cash index's price history has no dividends, so its yield comes from put-call parity.
    """
    upto = history.loc[:market_date(snapshot_time)]
    vol = realized_vol(upto["Close"]).iloc[-1] if len(upto) else np.nan
    if not np.isfinite(vol):
        raise ValueError(f"Not enough price history before {snapshot_time} to estimate volatility")
    if is_cash_index(ticker):
        q = parity_dividend_yield(chain, spot, rate, snapshot_time)
    else:
        q = dividend_yield(history, snapshot_time)
    return chain.assign(
        ticker=ticker.upper(),
        snapshotTime=pd.Timestamp(snapshot_time).isoformat(),
        underlyingPrice=float(spot),
        riskFreeRate=float(rate),
        dividendYield=q,
        histVol=float(vol),
    )


def latest_price(ticker: str) -> float:
    """The most recent price Yahoo has, including futures trading outside US hours."""
    tk = yf.Ticker(ticker)
    return _live_price(tk, clean_history(tk.history(period="5d", auto_adjust=False)))


RATIO_MINUTES = 15


def aligned_ratio(underlying: str, futures: str, download=None) -> float:
    """Futures points per underlying point from the minutes both traded: the median of the last ``RATIO_MINUTES``
    shared 1-minute closes, the futures' newest bar left out (Yahoo hands it over while it is still forming). NaN
    when they share no minute.

    Two latest prices mix different moments: Yahoo's futures run about ten minutes behind the ETF in the session, and
    after the 16:00 close the ETF stops while the futures go on. Fri 20 Nov 2026 16:30: QQQ's 16:00 close 608.45 and
    NQ's 16:30 price gave 41.478 against about 41.2-41.3 from shared minutes, so the Monday 09:15 card's NQ walls sat
    about 150-180 points high. The basis itself barely moves in a day."""
    download = download or yf.download
    bars = download([underlying, futures], period="5d", interval="1m", progress=False, auto_adjust=False)
    closes = bars["Close"][[underlying, futures]]
    newest = closes[futures].dropna().index.max() if closes[futures].notna().any() else None
    both = closes.dropna()
    if len(both) > 1 and both.index[-1] == newest:
        both = both.iloc[:-1]
    if both.empty:
        return float("nan")
    tail = both.tail(RATIO_MINUTES)
    return float((tail[futures] / tail[underlying]).median())


def matching_futures_price(underlying: str, futures: str, spot: float, download=None) -> float:
    """The futures price at the same moment as ``spot``: spot x :func:`aligned_ratio`; Yahoo's latest futures price
    (a different moment, logged) only when the two share no minute."""
    ratio = aligned_ratio(underlying, futures, download)
    if np.isfinite(ratio) and ratio > 0:
        return float(spot * ratio)
    log.warning("No minute shared by %s and %s; using the futures' latest price", underlying, futures)
    return latest_price(futures)


def _live_price(tk, history: pd.DataFrame) -> float:
    try:
        price = float(tk.fast_info["lastPrice"])
        if np.isfinite(price) and price > 0:
            return price
    except Exception:
        log.debug("fast_info price unavailable", exc_info=True)
    return float(history["Close"].iloc[-1])


# Below this, a chain was fetched outside US hours or on a holiday (Yahoo blanks the quotes, ~0%). Thin proxies
# such as UUP, FXE and FXB quote only 45-70% of their contracts on a normal day; build_dataset drops the rest.
MIN_LIVE_SHARE = 0.25
# New York times. The collector runs inside the session, away from the open and close.
SESSION_START, SESSION_END = dt.time(9, 45), dt.time(15, 55)
# Quotes stay at their closing values after 16:00 until SPX's overnight session starts at 20:15,
# whose quotes are wide and thin; snapshots from 09:45 to 20:00 are fit for training.
QUOTES_RELIABLE_UNTIL = dt.time(20, 0)


def _new_york(ts: pd.Timestamp | None) -> pd.Timestamp:
    ts = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(tz="UTC")
    return ts.tz_convert(MARKET_TZ) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(MARKET_TZ)


def us_session_open(now: pd.Timestamp | None = None) -> bool:
    """True on a weekday between 09:45 and 15:55 New York time."""
    now = _new_york(now)
    return now.weekday() < 5 and SESSION_START <= now.time() <= SESSION_END


def quotes_reliable(now: pd.Timestamp | None = None) -> bool:
    """True on a weekday between 09:45 and 20:00 New York time: in the session, or after
    the close while quotes hold their closing values."""
    now = _new_york(now)
    return now.weekday() < 5 and SESSION_START <= now.time() <= QUOTES_RELIABLE_UNTIL


def snapshot_problem(chain: pd.DataFrame) -> str | None:
    """Why a chain shouldn't be saved for training, or None if it's fine.

    It must be taken on a weekday between 09:45 and 20:00 New York time (SPX
    options also quote overnight, but those quotes are wide and thin) and most
    contracts must have a live quote.
    """
    if chain.empty:
        return "the chain is empty"
    taken = _new_york(pd.Timestamp(chain["snapshotTime"].iloc[0]))
    if taken.weekday() >= 5 or not SESSION_START <= taken.time() <= QUOTES_RELIABLE_UNTIL:
        return "it was taken outside US trading hours (09:45-20:00 New York)"
    share = live_quote_share(chain)
    if share < MIN_LIVE_SHARE:
        return f"only {share:.0%} of {len(chain):,} contracts had a live quote"
    return None


# Yahoo's cash-index chains are missing most of their open interest (19 Nov 2026: SPXW dailies showed 0, the SPX
# monthly about 4,800 contracts against millions), so the index's share of the levels was nearly empty. Cboe's free
# delayed-quotes JSON filled it until Cboe's firewall blocked it (20 Nov 2026, also in a normal browser). The OCC (the
# clearing house) publishes every series' open interest in one public report per underlying, the official figure
# every vendor starts from (once a day, as of the previous close): it fills Yahoo's open interest column, matched on
# the contract (root, expiry, call / put, strike). Yahoo's prices and volume are kept.
OCC_URL = "https://marketdata.theocc.com/series-search?symbolType=U&symbol={}"
OCC_INDEX = {"^SPX": ("SPX", ("SPX", "SPXW")), "^NDX": ("NDX", ("NDX", "NDXP"))}
_OCC = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")
AM_SETTLED = {"SPX", "NDX"}   # the monthly roots settle on the expiry day's opening prints (SPXW / NDXP at the close)
OPEN_NY = dt.time(9, 30)
_OCC_LINE = re.compile(r"^\s*([A-Z0-9]+)\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+)\s+(\d{3})\s+(?:C\s+P|C|P)?\s*"
                       r"(\d+)\s+(\d+)\s+(\d+)\s*$")


def _occ_text(symbol: str) -> str:
    import requests          # (the OCC refuses urllib's default client; requests is what Market Lab uses too)

    response = requests.get(OCC_URL.format(symbol), timeout=60)
    response.raise_for_status()
    return response.text


def occ_open_interest(ticker: str, get_text: Callable[[str], str] = _occ_text) -> dict:
    """The OCC's open interest for an index's options: {(root, "YYYY-MM-DD", "Call" / "Put", strike): contracts}."""
    symbol, roots = OCC_INDEX[ticker]
    out = {}
    for line in get_text(symbol).splitlines():
        m = _OCC_LINE.match(line)
        if not m or m.group(1) not in roots:
            continue
        root, y, mo, d, whole, thousandths, call_oi, put_oi, _ = m.groups()
        expiry, strike = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", int(whole) + int(thousandths) / 1000
        out[(root, expiry, "Call", strike)] = float(call_oi)
        out[(root, expiry, "Put", strike)] = float(put_oi)
    if not out:
        raise ValueError(f"the OCC report had no {ticker} options")
    return out


# Staleness guard. The OCC builds each trading day's file about 19:00 Chicago (20:00 New York) with that day's closing
# open interest, and the file still lists the series that expired that day; SPXW / NDXP expire every trading day, so
# a file's earliest expiry is the day it was built for. Each index chain records that day and when the file was read,
# and the card says so when the file is older than it should be at the read (or the OCC couldn't be read at all).
OCC_BUILT = dt.time(21, 0)    # 20:00 New York plus an hour's margin
OCC_DAY, OCC_READ = "occDay", "occRead"   # columns: the OCC file's trading day ("" = OCC unavailable), when read


def business_day(day: dt.date) -> bool:
    """An NYSE trading day (weekdays only when the ``holidays`` package isn't installed: the guard never fails)."""
    if day.weekday() >= 5:
        return False
    try:
        import holidays

        return day not in holidays.financial_holidays("NYSE", years=day.year)
    except Exception:
        return True


EARLY_CLOSE = dt.time(13, 0)   # US stocks on a half day


def early_close(day: dt.date) -> bool:
    """A US half day (stocks close 13:00 New York): the day after Thanksgiving, Christmas Eve and 3 July, when they
    are trading days."""
    thanksgiving = dt.date(day.year, 11, 1) + dt.timedelta(days=(3 - dt.date(day.year, 11, 1).weekday()) % 7 + 21)
    halves = {thanksgiving + dt.timedelta(days=1), dt.date(day.year, 12, 24), dt.date(day.year, 7, 3)}
    return day in halves and business_day(day)


def occ_due(now) -> dt.date:
    """The trading day the OCC's latest file should be for at ``now``: the last trading day whose evening build
    (:data:`OCC_BUILT` New York) has passed."""
    at = pd.Timestamp(now)
    at = (at.tz_localize("UTC") if at.tzinfo is None else at).tz_convert(MARKET_TZ)
    day = at.date() if at.time() >= OCC_BUILT else at.date() - dt.timedelta(days=1)
    while not business_day(day):
        day -= dt.timedelta(days=1)
    return day


def occ_file_day(oi: dict) -> dt.date:
    """The trading day an OCC report was built for: its earliest expiry (see :data:`OCC_BUILT`)."""
    return min(dt.date.fromisoformat(key[1]) for key in oi)


def occ_note(chain: pd.DataFrame) -> str | None:
    """A line for the card when an index chain's open interest isn't the latest there was at the read: the OCC's file
    older than :func:`occ_due`, or the OCC unavailable (then Yahoo's index open interest, most of it missing). None for
    a fresh file, an ETF's chain, or a snapshot saved before this was recorded."""
    if chain.empty or OCC_DAY not in chain or OCC_READ not in chain:
        return None
    name = str(chain["ticker"].iloc[0]).lstrip("^")
    day = chain[OCC_DAY].iloc[0]
    if not isinstance(day, str) or not day:             # "" comes back from a saved CSV as NaN
        return f"{name} open interest: the OCC was unavailable, so Yahoo's (most of it missing)"
    built = dt.date.fromisoformat(day)
    if built < occ_due(chain[OCC_READ].iloc[0]):
        return f"{name} open interest may be a day old (the OCC's latest file was for {built:%a %d %b})"
    return None


def drop_settled_at_open(chain: pd.DataFrame, now) -> pd.DataFrame:
    """``chain`` without the morning-settled monthlies (SPX / NDX root) expiring on ``now``'s day once the 09:30 New
    York open has settled them: they stopped trading the day before and carry no gamma after the opening prints."""
    at = pd.Timestamp(now).tz_convert(MARKET_TZ)
    if at.time() < OPEN_NY or chain.empty:
        return chain
    roots = chain["contractSymbol"].astype(str).str.extract(_OCC)[0]
    gone = roots.isin(AM_SETTLED) & (chain["ExpiryDate"].astype(str) == at.date().isoformat())
    return chain[~gone].reset_index(drop=True)


def fill_open_interest(chain: pd.DataFrame, oi: dict) -> tuple[pd.DataFrame, int]:
    """Yahoo's chain with its open interest taken from ``oi`` (see :func:`occ_open_interest`) wherever the contract is
    there; the count of contracts filled."""
    def key(row):
        m = _OCC.match(str(row["contractSymbol"]))
        return (m.group(1), str(row["ExpiryDate"]), row["OptionType"], float(row["strike"])) if m else None

    keys = chain.apply(key, axis=1)
    found = keys.map(lambda k: oi.get(k) if k is not None else None)
    out = chain.copy()
    out["openInterest"] = found.where(found.notna(), pd.to_numeric(out["openInterest"], errors="coerce")).astype(float)
    return out, int(found.notna().sum())


def fetch_option_chain(ticker: str, directory: Path | None = None, save: bool = True,
                       progress: Callable[[str], None] | None = None,
                       max_expiries: int | None = None) -> pd.DataFrame:
    """Download every listed expiry for ``ticker`` and save it as a dated snapshot.

    A chain that :func:`snapshot_problem` rejects is returned but not saved, so it
    can't end up in a training set.
    """
    tk = yf.Ticker(ticker)
    snapshot_time = pd.Timestamp.now(tz="UTC")
    frames = []
    expiries = list(tk.options)[:max_expiries]
    if not expiries:
        raise ValueError(f"No listed options found for {ticker!r}")
    for i, expiry in enumerate(expiries, 1):
        if progress:
            progress(f"{ticker.upper()}: expiry {expiry} ({i}/{len(expiries)})")
        chain = tk.option_chain(expiry)
        frames.append(chain.calls.assign(OptionType="Call", ExpiryDate=expiry))
        frames.append(chain.puts.assign(OptionType="Put", ExpiryDate=expiry))
    whole = pd.concat(frames, ignore_index=True)
    if ticker.upper() in OCC_INDEX:
        whole = drop_settled_at_open(whole, snapshot_time)
        try:
            oi = occ_open_interest(ticker.upper())
            whole, filled = fill_open_interest(whole, oi)
            whole[OCC_DAY] = occ_file_day(oi).isoformat()
            if progress:
                progress(f"{ticker.upper()}: open interest for {filled:,} contracts from the OCC")
        except Exception:
            whole[OCC_DAY] = ""
            log.warning("The OCC's %s open interest unavailable; keeping Yahoo's", ticker, exc_info=True)
        whole[OCC_READ] = snapshot_time.isoformat()
    history = clean_history(tk.history(period="1y", auto_adjust=False))
    spot = _live_price(tk, history)
    chain = enrich_snapshot(whole, ticker, snapshot_time, spot=spot, history=history, rate=risk_free_rate())
    futures = instruments.futures_for(ticker)
    if futures is not None:
        # The futures price that matches the spot (spot x the ratio from minutes both traded), from the contract we
        # trade (see contracts.py), recorded so a later switch can be allowed for.
        from bsnn import contracts

        contract = contracts.futures_ticker(futures.symbol, snapshot_time)
        try:
            chain["futuresPrice"] = matching_futures_price(ticker, contract, spot)
            chain["futuresTicker"] = contract
        except Exception:
            log.warning("Couldn't fetch %s alongside %s", contract, ticker, exc_info=True)
    if save and snapshot_problem(chain) is None:
        save_option_snapshot(chain, directory)
    return chain


def live_quote_share(chain: pd.DataFrame) -> float:
    """Share of contracts with a two-sided quote. Yahoo blanks bids and asks outside US hours."""
    return float(((chain["bid"] > 0) & (chain["ask"] > 0)).mean()) if len(chain) else 0.0


def list_option_snapshots(directory: Path | None = None) -> list[Path]:
    """Saved snapshots, sorted by ticker then date.

    Includes sub-folders, such as the cloud-collected repository cloned to
    ``data/options/cloud``. A snapshot saved twice (by the laptop and the cloud
    job on the same day) is listed once, preferring the copy nearest the top.
    """
    found = sorted(Path(directory or paths.OPTIONS_DIR).rglob("*_options_*.csv*"),
                   key=lambda p: (len(p.parts), str(p)))
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(snapshot_label(path), path)
    return sorted(unique.values(), key=lambda p: p.name)


def load_option_snapshots(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in SNAPSHOT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{Path(path).name} is missing columns: {', '.join(missing)}")
        frames.append(df)
    if not frames:
        raise ValueError("No option snapshots selected")
    return pd.concat(frames, ignore_index=True)
