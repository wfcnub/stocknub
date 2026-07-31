import argparse
from pathlib import Path

from selectTickerToProcess.config import SelectionConfig
from selectTickerToProcess.main import build_selection_universe
from selectTickerToProcess.validation import append_selection_history
from utils import paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a stable model-development universe using fundamentals and OHLCV",
    )
    parser.add_argument(
        "--ohlcv_folder_path",
        type=Path,
        default=paths.get_ohlcv_dir(),
    )
    parser.add_argument(
        "--foreign_flow_folder_path",
        type=Path,
        default=paths.get_foreign_flow_non_regular_dir(),
    )
    parser.add_argument(
        "--industry_file_path",
        type=Path,
        default=paths.get_ticker_and_industry_list_path(),
    )
    parser.add_argument(
        "--fundamental_cache_path",
        type=Path,
        default=paths.get_fundamental_history_path(),
    )
    parser.add_argument(
        "--fundamental_snapshot_path",
        type=Path,
        default=None,
        help="Optional offline snapshot; when set, no Yahoo requests are made",
    )
    parser.add_argument(
        "--selected_output_path",
        type=Path,
        default=paths.get_selected_ticker_and_industry_list_path(),
    )
    parser.add_argument(
        "--tactical_output_path",
        type=Path,
        default=paths.get_tactical_ticker_list_path(),
    )
    parser.add_argument(
        "--audit_output_path",
        type=Path,
        default=paths.get_ticker_selection_audit_path(),
    )
    parser.add_argument(
        "--audit_history_path",
        type=Path,
        default=paths.get_ticker_selection_history_path(),
    )
    parser.add_argument("--top_n", type=int, default=150)
    parser.add_argument("--tactical_top_n", type=int, default=25)
    parser.add_argument("--min_history_rows", type=int, default=504)
    parser.add_argument("--min_traded_days", type=int, default=220)
    parser.add_argument("--min_adv_60", type=float, default=5_000_000_000)
    parser.add_argument("--min_fundamental_score", type=float, default=0.20)
    parser.add_argument("--min_technical_score", type=float, default=0.20)
    parser.add_argument("--max_sector_share", type=float, default=0.25)
    parser.add_argument("--fundamental_workers", type=int, default=4)
    parser.add_argument("--cache_ttl_days", type=int, default=30)
    parser.add_argument("--refresh_fundamentals", action="store_true")
    parser.add_argument("--skip_auxiliary_data_check", action="store_true")
    parser.add_argument("--disable_hysteresis", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = SelectionConfig(
        top_n=args.top_n,
        min_history_rows=args.min_history_rows,
        min_traded_days_252=args.min_traded_days,
        min_median_value_traded_60=args.min_adv_60,
        min_fundamental_score=args.min_fundamental_score,
        min_technical_score=args.min_technical_score,
        max_sector_share=args.max_sector_share,
        fundamental_fetch_workers=args.fundamental_workers,
        fundamental_cache_ttl_days=args.cache_ttl_days,
        require_auxiliary_data=not args.skip_auxiliary_data_check,
        incumbent_bonus=0 if args.disable_hysteresis else 0.02,
    )

    print("=" * 80)
    print("PIPELINE DESCRIPTION: SELECT FUNDAMENTALLY AND TECHNICALLY STRONG TICKERS")
    print("=" * 80)
    print(f"OHLCV directory: {args.ohlcv_folder_path}")
    print(f"Target model universe: {config.top_n}")
    print(f"Minimum 60-day median traded value: IDR {config.min_median_value_traded_60:,.0f}")

    selected, audit, summary = build_selection_universe(
        ohlcv_dir=args.ohlcv_folder_path,
        auxiliary_dir=(
            None if args.skip_auxiliary_data_check else args.foreign_flow_folder_path
        ),
        industry_path=args.industry_file_path,
        config=config,
        fundamental_cache_path=args.fundamental_cache_path,
        fundamental_snapshot_path=args.fundamental_snapshot_path,
        previous_selection_path=args.selected_output_path,
        refresh_fundamentals=args.refresh_fundamentals,
    )

    args.audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output_path, index=False)

    if summary["fundamental_fetch_coverage"] < config.min_fetch_coverage:
        raise RuntimeError(
            "Fundamental coverage is too low to rank the market safely: "
            f"{summary['fundamental_fetch_coverage']:.1%} < {config.min_fetch_coverage:.1%}. "
            f"Audit saved to {args.audit_output_path}."
        )

    append_selection_history(audit, args.audit_history_path)
    args.selected_output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.selected_output_path, index=False)
    tactical = selected.sort_values(
        ["technical_score", "Ticker"], ascending=[False, True]
    ).head(args.tactical_top_n)
    args.tactical_output_path.parent.mkdir(parents=True, exist_ok=True)
    tactical.to_csv(args.tactical_output_path, index=False)

    print(f"Universe: {summary['universe_count']}")
    print(f"Passed OHLCV gates: {summary['ohlcv_eligible_count']}")
    print(f"Fundamental coverage: {summary['fundamental_fetch_coverage']:.1%}")
    print(f"Passed score floors: {summary['selection_eligible_count']}")
    print(f"Selected: {summary['selected_count']}")
    if len(selected):
        print("Industry breakdown:")
        for industry, count in selected["Industry"].value_counts().items():
            print(f" - {industry}: {count}")
    print(f"Selected output: {args.selected_output_path}")
    print(f"Tactical shortlist: {args.tactical_output_path}")
    print(f"Full audit: {args.audit_output_path}")
    print(f"Prospective history: {args.audit_history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
