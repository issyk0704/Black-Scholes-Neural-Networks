"""Market data from Yahoo Finance, cached as CSV files.

Price history is stored the way Yahoo returns it with ``auto_adjust=False``:
split-adjusted but not dividend-adjusted. :func:`unadjusted_close` recovers the
price that actually traded on a day that comes before a later split.

Each option-chain snapshot is saved with the market inputs that applied when it
was taken (spot, risk-free rate, dividend yield, historical volatility), so a
dataset built from old snapshots never needs to go back to the network.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from bsnn import paths

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
    calls = mids[mids["OptionType"] == "Call"].set_index(["ExpiryDate", "strike"])["mid"]
    puts = mids[mids["OptionType"] == "Put"].set_index(["ExpiryDate", "strike"])["mid"]
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


def _live_price(tk, history: pd.DataFrame) -> float:
    try:
        price = float(tk.fast_info["lastPrice"])
        if np.isfinite(price) and price > 0:
            return price
    except Exception:
        log.debug("fast_info price unavailable", exc_info=True)
    return float(history["Close"].iloc[-1])


MIN_LIVE_SHARE = 0.5  # below this, a chain was fetched outside US hours or on a holiday


def fetch_option_chain(ticker: str, directory: Path | None = None, save: bool = True,
                       progress: Callable[[str], None] | None = None,
                       max_expiries: int | None = None) -> pd.DataFrame:
    """Download every listed expiry for ``ticker`` and save it as a dated snapshot.

    A chain where fewer than :data:`MIN_LIVE_SHARE` of contracts have a live quote
    is returned but not saved, so it can't end up in a training set.
    """
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)[:max_expiries]
    if not expiries:
        raise ValueError(f"No listed options found for {ticker!r}")
    snapshot_time = pd.Timestamp.now(tz="UTC")
    frames = []
    for i, expiry in enumerate(expiries, 1):
        if progress:
            progress(f"{ticker.upper()}: expiry {expiry} ({i}/{len(expiries)})")
        chain = tk.option_chain(expiry)
        frames.append(chain.calls.assign(OptionType="Call", ExpiryDate=expiry))
        frames.append(chain.puts.assign(OptionType="Put", ExpiryDate=expiry))
    history = clean_history(tk.history(period="1y", auto_adjust=False))
    chain = enrich_snapshot(pd.concat(frames, ignore_index=True), ticker, snapshot_time,
                            spot=_live_price(tk, history), history=history, rate=risk_free_rate())
    if save and live_quote_share(chain) >= MIN_LIVE_SHARE:
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
