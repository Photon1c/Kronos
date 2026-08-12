#!/usr/bin/env python3
"""
Kronos test script — chart output + predictions for daily stock data.

Reads Yahoo-format daily CSVs from ~/data/stocks/<TICKER>.csv, normalizes them
for Kronos (chronological OHLCV), loads the pretrained model from Hugging Face,
forecasts the next N trading days, prints the prediction table, and saves:
  - a chart (ground truth vs prediction) as PNG
  - the predictions as CSV

Usage examples (run from the Kronos repo root):
  # Quick test on AAPL: forecast 30 days, compare against held-out ground truth
  ~/envs/kronosenv/bin/python test_predict_stocks.py --ticker AAPL

  # Forecast into the future (beyond the last data point) for AMD
  ~/envs/kronosenv/bin/python test_predict_stocks.py --ticker AMD --future

  # Lighter model + more forecast paths
  ~/envs/kronosenv/bin/python test_predict_stocks.py --ticker QQQ --model NeoQuasar/Kronos-mini --samples 5

  # Custom window / horizon / output dir
  ~/envs/kronosenv/bin/python test_predict_stocks.py --ticker SPY --lookback 256 --pred-len 20 --outdir /tmp/kronos_test
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: save charts to file, never call plt.show()
import matplotlib.pyplot as plt

# Make `from model import ...` work regardless of CWD
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

DEFAULT_DATA_DIR = os.path.expanduser("~/data/stocks")
DEFAULT_OUT_DIR = os.path.expanduser("~/data/kronos_output")


def load_yahoo_csv(path: str) -> pd.DataFrame:
    """Load a Yahoo-format daily CSV and normalize it for Kronos."""
    raw = pd.read_csv(path)
    # Yahoo format: Date,Close/Last,Volume,Open,High,Low (newest first, $ prices)
    rename = {"Close/Last": "close", "Open": "open", "High": "high", "Low": "low"}
    df = raw.rename(columns=rename).copy()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(str).str.replace("$", "", regex=False).astype(float)
    df["volume"] = pd.to_numeric(df.get("Volume"), errors="coerce").fillna(0.0)
    df["timestamps"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["timestamps", "open", "high", "low", "close"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    df = df[["timestamps", "open", "high", "low", "close", "volume"]]
    return df


def make_future_timestamps(last_ts: pd.Timestamp, n: int) -> pd.Series:
    """Generate the next n trading days after the last observed date."""
    return pd.Series(pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=n))


def plot_prediction(
    kline_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame | None,
    ticker: str,
    model_name: str,
    out_png: str,
) -> None:
    """Ground truth vs prediction close-price chart, saved to out_png."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Historical context (last 120 bars)
    tail = kline_df.tail(120)
    ax.plot(
        tail["timestamps"],
        tail["close"],
        label="Historical close",
        color="blue",
        linewidth=1.5,
    )

    # Held-out ground truth window (only when we're testing against real data)
    if gt_df is not None:
        ax.plot(
            gt_df["timestamps"],
            gt_df["close"],
            label="Ground truth",
            color="green",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )

    ax.plot(
        pred_df.index,
        pred_df["close"],
        label="Kronos prediction",
        color="red",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    ax.set_title(f"{ticker} — Kronos forecast (model: {model_name.split('/')[-1]})")
    ax.set_ylabel("Close price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Kronos daily-stock forecast test")
    ap.add_argument("--ticker", default="AAPL", help="Stock symbol (default: AAPL)")
    ap.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Dir with <TICKER>.csv files"
    )
    ap.add_argument(
        "--lookback",
        type=int,
        default=512,
        help="Historical bars fed to the model (max 512 for small/base)",
    )
    ap.add_argument(
        "--pred-len", type=int, default=30, help="Number of days to forecast"
    )
    ap.add_argument(
        "--samples", type=int, default=1, help="Forecast paths to sample & average"
    )
    ap.add_argument(
        "--model",
        default="NeoQuasar/Kronos-small",
        help="HF model id (mini/small/base)",
    )
    ap.add_argument(
        "--tokenizer",
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="HF tokenizer id (use Kronos-Tokenizer-2k with Kronos-mini)",
    )
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument(
        "--outdir", default=DEFAULT_OUT_DIR, help="Where to save chart + CSV"
    )
    ap.add_argument(
        "--future",
        action="store_true",
        help="Forecast beyond the last data point (no ground truth to compare)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    csv_path = os.path.join(args.data_dir, f"{args.ticker.upper()}.csv")
    if not os.path.exists(csv_path):
        print(f"✗ No data file: {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading {csv_path} ...")
    df = load_yahoo_csv(csv_path)
    print(
        f"  {len(df)} daily bars, {df['timestamps'].iloc[0].date()} → {df['timestamps'].iloc[-1].date()}"
    )

    need = args.lookback if args.future else args.lookback + args.pred_len
    if len(df) < need:
        print(f"✗ Need at least {need} bars, have {len(df)}", file=sys.stderr)
        return 1

    # Split: context window, then the forecast horizon.
    # Test mode holds out the last pred_len bars as ground truth, so the
    # context must end pred_len bars before the last bar. --future forecasts
    # beyond the last data point, so the context must end at the last bar
    # (otherwise the projection starts from a stale window and appears to gap
    # down/up from the spot price on the chart).
    if args.future:
        x_df = df.loc[
            len(df) - args.lookback : len(df) - 1,
            ["timestamps", "open", "high", "low", "close", "volume"],
        ].reset_index(drop=True)
    else:
        x_df = df.loc[
            len(df) - args.lookback - args.pred_len : len(df) - args.pred_len - 1,
            ["timestamps", "open", "high", "low", "close", "volume"],
        ].reset_index(drop=True)
    x_ts = x_df["timestamps"]
    x_df = x_df[["open", "high", "low", "close", "volume"]]

    if args.future:
        y_ts = make_future_timestamps(df["timestamps"].iloc[-1], args.pred_len)
    else:
        y_ts = df["timestamps"].iloc[-args.pred_len :].reset_index(drop=True)

    print(
        f"Loading tokenizer ({args.tokenizer}) and model ({args.model}) ... "
        f"(first run downloads from Hugging Face)"
    )
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(
        model, tokenizer, device=args.device, max_context=args.lookback
    )

    print(f"Forecasting {args.pred_len} days from {args.lookback} days of context ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=args.samples,
        verbose=True,
    )

    print("\nForecasted data (close prices):")
    out = pred_df[["open", "high", "low", "close"]].copy()
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    out["close"] = out["close"].round(2)
    print(out.to_string())

    # Accuracy check when ground truth exists
    if not args.future:
        gt_close = df["close"].iloc[-args.pred_len :].values
        fc_close = pred_df["close"].values
        mape = np.mean(np.abs((gt_close - fc_close) / gt_close)) * 100
        print(f"\nMean absolute % error vs ground truth: {mape:.2f}%")

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.ticker.upper()}_{'future' if args.future else 'test'}_{stamp}"
    out_png = os.path.join(args.outdir, f"{base}.png")
    out_csv = os.path.join(args.outdir, f"{base}_predictions.csv")

    # Build full series for plotting (history + horizon)
    plot_df = (
        df[["timestamps", "close"]]
        .iloc[-(args.lookback + args.pred_len) :]
        .reset_index(drop=True)
    )
    if args.future:
        # No ground truth beyond the data end
        plot_prediction(
            plot_df, pred_df, None, args.ticker.upper(), args.model, out_png
        )
    else:
        gt_df = (
            df[["timestamps", "close"]].iloc[-args.pred_len :].reset_index(drop=True)
        )
        plot_prediction(
            plot_df, pred_df, gt_df, args.ticker.upper(), args.model, out_png
        )

    pred_df.to_csv(out_csv)
    print(f"\n✓ Chart:   {out_png}")
    print(f"✓ Predictions: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
