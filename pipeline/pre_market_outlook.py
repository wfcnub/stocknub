from preMarketOutlook.main import generate_pre_market_outlook

if __name__ == "__main__":
    print("=" * 80)
    print("PIPELINE DESCRIPTION: PRE-MARKET MACRO OUTLOOK")
    print("=" * 80)
    print("Fetching VIX, USD/IDR, S&P 500, and Nikkei 225 data from Yahoo Finance")
    print()

    try:
        outlook = generate_pre_market_outlook()

        print("\n" + "=" * 80)
        print("PRE-MARKET OUTLOOK SUMMARY")
        print("=" * 80)
        print(f"Overall Outlook : {outlook['overall_outlook']['outlook']}")
        print(f"Composite Score : {outlook['overall_outlook']['composite_score']:+.2f}")
        print("Pipeline completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        print("=" * 80)
        raise
