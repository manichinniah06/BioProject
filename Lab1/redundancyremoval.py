import pandas as pd

def remove_redundancy(input_file):
    print("="*70)
    print("REDUNDANCY REMOVAL")
    print("="*70)

    df = pd.read_csv(input_file)
    print(f"Initial dataset size: {len(df)}")

    df = df.drop_duplicates()
    print(f"After removing exact duplicate rows: {len(df)}")

    df = df.drop_duplicates(subset=['sequence'])
    print(f"After removing duplicate sequences: {len(df)}")

    df = df.drop_duplicates(subset=['mutation'])
    print(f"After removing duplicate mutation entries: {len(df)}")

    df.to_csv("p53_dataset_non_redundant.csv", index=False)

    print("\nFinal non-redundant dataset saved.")
    print("="*70)

    return df

clean_df = remove_redundancy("p53_missense_mutation_dataset.csv")
