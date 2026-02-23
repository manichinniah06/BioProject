"""
Redundancy Removal Module
=========================

Removes duplicate sequences from cross-species dataset using k-mer similarity.
Part of: src/preprocessing/
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    REDUNDANCY_THRESHOLD, KMER_SIZE, 
    HUMAN_MUTATIONS_RAW, CROSS_SPECIES_RAW, COMBINED_DATASET
)


def generate_kmers(sequence, k=3):
    """Generate k-mers from sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence)-k+1)])


def remove_redundancy_cosine(df, threshold=REDUNDANCY_THRESHOLD, kmer_size=KMER_SIZE):
    """
    Remove redundant sequences using k-mer cosine similarity.
    
    Args:
        df: DataFrame with 'Sequence' column
        threshold: Similarity threshold (0-1)
        kmer_size: Size of k-mers
        
    Returns:
        DataFrame with only unique sequences
    """
    
    print(f"\nPerforming redundancy removal (threshold={threshold}, k={kmer_size})...")
    print(f"  Input sequences: {len(df)}")
    
    # Generate k-mers
    df["kmer"] = df["Sequence"].apply(lambda x: generate_kmers(str(x), kmer_size))
    
    # Compute similarity matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["kmer"])
    sim_matrix = cosine_similarity(X)
    
    # Find unique sequences
    keep = []
    removed = set()
    
    for i in range(len(df)):
        if i in removed:
            continue
        
        keep.append(i)
        
        for j in range(i + 1, len(df)):
            if sim_matrix[i][j] >= threshold:
                removed.add(j)
    
    result_df = df.iloc[keep].drop(columns=["kmer"])
    
    print(f"  Removed: {len(df) - len(result_df)} sequences")
    print(f"  Output sequences: {len(result_df)}")
    
    return result_df


def combine_datasets():
    """Load, combine, and deduplicate human and cross-species datasets."""
    
    print("\n" + "="*70)
    print("COMBINING AND DEDUPLICATING DATASETS")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    human_df = pd.read_csv(HUMAN_MUTATIONS_RAW)
    cross_df = pd.read_csv(CROSS_SPECIES_RAW)
    
    print(f"  Human mutations: {len(human_df)}")
    print(f"  Cross-species: {len(cross_df)}")
    
    # Standardize column names
    human_df = human_df.rename(columns={"sequence": "Sequence"}) if "sequence" in human_df.columns else human_df
    cross_df = cross_df.rename(columns={"sequence": "Sequence"}) if "sequence" in cross_df.columns else cross_df
    
    # Add sequence type
    human_df["Sequence_Type"] = "Human_Mutation"
    cross_df["Sequence_Type"] = "Cross_Species"
    
    # Remove redundancy from cross-species only
    cross_non_redundant = remove_redundancy_cosine(cross_df)
    
    # Combine
    final_df = pd.concat([human_df, cross_non_redundant], ignore_index=True)
    
    print(f"\n✓ Combined dataset size: {len(final_df)}")
    print(f"  Human: {(final_df['Sequence_Type']=='Human_Mutation').sum()}")
    print(f"  Cross-species: {(final_df['Sequence_Type']=='Cross_Species').sum()}")
    
    return final_df


if __name__ == "__main__":
    
    final_df = combine_datasets()
    
    # Save
    COMBINED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(COMBINED_DATASET, index=False)
    
    print(f"\n✓ Saved to: {COMBINED_DATASET}")
    print("="*70 + "\n")
