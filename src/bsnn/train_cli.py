"""Command-line training, for reproducible experiments without the GUI.

    bsnn-train                          # smile model, ATM vol, random-expiry split
    bsnn-train --split date --save      # train on older snapshots, test on the newest; keep the model
    bsnn-train --kind price --hist-vol --tickers SPY QQQ
"""

from __future__ import annotations

import argparse

import pandas as pd

from bsnn import market_data
from bsnn.features import build_dataset
from bsnn.model import MODEL_KINDS, SPLITS, TrainConfig, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the neural-network pricer and compare it with Black-Scholes.")
    parser.add_argument("--tickers", nargs="*", help="only use snapshots for these tickers")
    parser.add_argument("--kind", choices=list(MODEL_KINDS), default=TrainConfig.kind)
    parser.add_argument("--split", choices=list(SPLITS), default=TrainConfig.split)
    parser.add_argument("--hist-vol", action="store_true",
                        help="use 30-day historical vol instead of the expiry's ATM implied vol")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--layers", type=int, nargs="+", default=list(TrainConfig.hidden_layers))
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--save", action="store_true", help="save the trained model under models/")
    args = parser.parse_args(argv)

    files = market_data.list_option_snapshots()
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        files = [f for f in files if f.name.split("_")[0] in wanted]
    dataset = build_dataset(market_data.load_option_snapshots(files))
    print(f"{len(dataset):,} contracts from {len(files)} snapshot(s)")

    config = TrainConfig(kind=args.kind, use_atm_iv=not args.hist_vol, hidden_layers=tuple(args.layers),
                         epochs=args.epochs, split=args.split, seed=args.seed)

    def report(epoch, total, logs):
        if epoch % 10 == 0:
            print(f"  epoch {epoch}/{total}  loss {logs['loss']:.4f}  val_loss {logs['val_loss']:.4f}")

    result = train(dataset, config, on_epoch=report)
    rows = result.pricer.metadata["rows"]
    print(f"\n{result.pricer.description}. Test set: {rows['test']:,} contracts (trained on {rows['train']:,}, "
          f"validated on {rows['validation']:,}; {result.pricer.metadata['epochs_run']} epochs)\n")
    with pd.option_context("display.width", 160, "display.max_columns", 10, "display.float_format", "{:.2f}".format):
        print(result.metrics)
    if args.save:
        print(f"\nSaved to {result.pricer.save()}")


if __name__ == "__main__":
    main()
