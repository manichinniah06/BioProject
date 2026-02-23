"""
Global Sequence Alignment Module
=================================

Performs global alignment of TP53 sequences against wild-type.
Part of: src/alignment/
"""

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    WT_P53, COMBINED_DATASET, ALIGNED_DATASET,
    ALIGNMENT_MODE, BLOSUM_MATRIX, GAP_OPEN, GAP_EXTEND
)


def compute_identity(alignment):
    """Compute percent identity from alignment."""
    
    matches = 0
    aligned_length = 0
    
    for (t_start, t_end), (q_start, q_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        target_segment = WT_P53[t_start:t_end]
        query_segment = alignment.query[q_start:q_end]
        
        for a, b in zip(target_segment, query_segment):
            aligned_length += 1
            if a == b:
                matches += 1
    
    return (matches / aligned_length) * 100 if aligned_length > 0 else 0


def perform_global_alignment(df):
    """
    Perform global alignment for all sequences.
    
    Args:
        df: DataFrame with 'Sequence' column
        
    Returns:
        DataFrame with alignment scores added
    """
    
    print("\n" + "="*70)
    print("PERFORMING GLOBAL ALIGNMENT")
    print("="*70)
    
    # Configure aligner
    aligner = PairwiseAligner()
    aligner.mode = ALIGNMENT_MODE
    aligner.substitution_matrix = substitution_matrices.load(BLOSUM_MATRIX)
    aligner.open_gap_score = GAP_OPEN
    aligner.extend_gap_score = GAP_EXTEND
    
    print(f"\nAlignment settings:")
    print(f"  Mode: {ALIGNMENT_MODE}")
    print(f"  Matrix: {BLOSUM_MATRIX}")
    print(f"  Gap open: {GAP_OPEN}")
    print(f"  Gap extend: {GAP_EXTEND}")
    
    alignment_scores = []
    identities = []
    
    print(f"\nAligning {len(df)} sequences...")
    for idx, seq in enumerate(df["Sequence"]):
        if (idx + 1) % max(1, len(df) // 10) == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")
        
        try:
            alignment = aligner.align(WT_P53, str(seq))[0]
            alignment_scores.append(float(alignment.score))
            identity = compute_identity(alignment)
            identities.append(identity)
        except Exception as e:
            print(f"    Warning: Alignment failed for sequence {idx}: {e}")
            alignment_scores.append(None)
            identities.append(None)
    
    df["Alignment_Score"] = alignment_scores
    df["Percent_Identity"] = identities
    
    # Summary
    valid_scores = [s for s in alignment_scores if s is not None]
    valid_identities = [i for i in identities if i is not None]
    
    if valid_identities:
        print(f"\n✓ Alignment complete:")
        print(f"  Min Identity: {min(valid_identities):.2f}%")
        print(f"  Max Identity: {max(valid_identities):.2f}%")
        print(f"  Mean Identity: {sum(valid_identities)/len(valid_identities):.2f}%")
    
    return df


if __name__ == "__main__":
    
    print(f"Wild-type p53 length: {len(WT_P53)} amino acids\n")
    
    # Load combined dataset
    df = pd.read_csv(COMBINED_DATASET)
    print(f"Loaded {len(df)} sequences from: {COMBINED_DATASET}")
    
    # Perform alignment
    aligned_df = perform_global_alignment(df)
    
    # Save
    ALIGNED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(ALIGNED_DATASET, index=False)
    
    print(f"\n✓ Saved to: {ALIGNED_DATASET}")
    print("="*70 + "\n")
