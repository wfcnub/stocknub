import os
import argparse

from selectEmitenToProcess.main import select_emiten_to_process

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Select Emiten to Process"
    )
    parser.add_argument(
        "--perc_emiten_in_industry",
        type=float,
        default=0.85,
        help="The percentile threshold of the total emiten to select from each industry",
    )
    parser.add_argument(
        "--perc_emiten",
        type=float,
        default=0.9,
        help="The percentile threshold of the total emiten to select from each industry",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE DESCRIPTION: SELECT EMITEN TO PROCESS")
    print("=" * 80)

    print('Selecting all emiten that meet the following criterias')
    print(f'    Selecting all emiten in an industry having an average valution greater then the {args.perc_emiten_in_industry} percentile')
    print(f'    Selecting all emiten having an average valution greater then the {args.perc_emiten} percentile')
    print()

    selected_emiten_df = select_emiten_to_process(args.perc_emiten_in_industry, args.perc_emiten)

    print(f'Selected a total of {len(selected_emiten_df)} emiten, with a breakdown for each industry as follows')
    selected_industry_count = selected_emiten_df['Industri'].value_counts()
    for industry, count in zip(selected_industry_count.index, selected_industry_count.values):
        print(f'    Selected a total of {count} emiten from the {industry} sector')
    print()

    save_path = 'data/selected_emiten_and_industry_list.csv'
    selected_emiten_df.to_csv(save_path, index=False)
    print(f'Successfully saved the data to {save_path}')

    print("=" * 80)

if __name__ == "__main__":
    main()