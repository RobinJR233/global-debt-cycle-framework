"""
Global Debt Cycle Assessment Framework
========================================
CLI entry point for running the pipeline.
"""

import argparse
import json
import logging
import sys
from datetime import datetime

from src.orchestration.pipeline import (
    run_pipeline_for_country,
    run_pipeline_batch,
    get_global_summary,
)
from src.data_ingestion.pipeline import get_ingestion_summary


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_country(args):
    """Run pipeline for a single country."""
    result = run_pipeline_for_country(
        country=args.country.upper(),
        start_year=args.start,
        end_year=args.end,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        latest = result["latest_year"]
        dsm = result["dsm"]
        ext = result["external_vulnerability"]

        print(f"\n{'=' * 60}")
        print(f"  {latest['country']} — Debt Cycle Assessment {latest['year']}")
        print(f"{'=' * 60}")
        print(f"  Composite Score : {latest['composite_score']}")
        print(f"  Phase           : {latest['phase']}")
        print(f"  Velocity        : {latest['velocity']:+}")
        print(f"  EWS Score       : {latest['ews_score']}")
        print(f"\n  Debt/GDP        : {latest['gov_gross_debt_pct_gdp']}%")
        print(f"  g - r           : {latest['g_minus_r']}")
        print(f"  Fiscal Space    : {latest['fiscal_space']}")
        print(f"\n  DSM Verdict     : {dsm['sustainability_verdict']}")
        print(f"  Baseline 5yr    : {dsm['baseline_path'][0]}% → {dsm['baseline_final_debt_gdp']}%")
        print(f"  Severe 5yr      : {dsm['severe_path'][0]}% → {dsm['severe_final_debt_gdp']}%")
        print(f"\n  External Risk   : {ext['overall_risk']}")
        print(f"  GFN/Reserves    : {ext['gfn_reserves_ratio']}%")
        print(f"  Currency Mis.   : {ext['currency_mismatch_index']}")
        print(f"\n  Sub-Scores:")
        for ss in latest["sub_scores"]:
            print(f"    {ss['dimension']:20s}: {ss['score']:5.1f}  (weight {ss['weight']:.0%})")
        print()


def cmd_batch(args):
    """Run pipeline for multiple countries."""
    countries = [c.strip().upper() for c in args.countries.split(",")]
    results = run_pipeline_batch(countries, args.start, args.end)
    summary = get_global_summary(results)

    if args.format == "json":
        output = {"summary": summary, "countries": results}
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n{'=' * 60}")
        print(f"  Global Debt Cycle Summary ({summary['total_countries']} countries)")
        print(f"{'=' * 60}")
        print(f"  Avg Score: {summary['avg_composite_score']}")
        print(f"\n  Top {min(5, len(summary['top_5_resilient']))} (most resilient):")
        for code, score, phase in summary["top_5_resilient"][:5]:
            print(f"    {code:4s}  {score:5.1f}  {phase}")
        print(f"\n  Bottom {min(5, len(summary['bottom_5_vulnerable']))} (most vulnerable):")
        for code, score, phase in summary["bottom_5_vulnerable"][:5]:
            print(f"    {code:4s}  {score:5.1f}  {phase}")
        print(f"\n  Phase Distribution:")
        for phase, count in summary["phase_distribution"].items():
            print(f"    {phase:10s}: {count}")
        print()


def cmd_sources(args):
    """Show data source information."""
    info = get_ingestion_summary()
    print(json.dumps(info, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Global Debt Cycle Assessment Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py country USA --start 2000 --end 2024
  python main.py batch "USA,JPN,DEU,GBR,CHN" --format json
  python main.py sources
        """,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # country command
    p_country = subparsers.add_parser("country", help="Analyze a single country")
    p_country.add_argument("country", help="ISO 3-letter country code (e.g. USA, CHN)")
    p_country.add_argument("--start", type=int, default=2000, help="Start year")
    p_country.add_argument("--end", type=int, default=2026, help="End year")
    p_country.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # batch command
    p_batch = subparsers.add_parser("batch", help="Analyze multiple countries")
    p_batch.add_argument("countries", help="Comma-separated ISO codes")
    p_batch.add_argument("--start", type=int, default=2000, help="Start year")
    p_batch.add_argument("--end", type=int, default=2026, help="End year")
    p_batch.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # sources command
    subparsers.add_parser("sources", help="Show data source info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_logging(args.verbose)

    if args.command == "country":
        cmd_country(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "sources":
        cmd_sources(args)


if __name__ == "__main__":
    main()
