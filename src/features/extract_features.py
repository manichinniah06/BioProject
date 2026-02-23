"""
Handcrafted Feature Extraction Module
======================================

Extracts biologically-informed features from TP53 sequences.
Part of: src/features/
"""

import pandas as pd
from Bio.Align import substitution_matrices
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    ALIGNED_DATASET, HANDCRAFTED_FEATURES,
    DNA_BINDING_DOMAIN_START, DNA_BINDING_DOMAIN_END, WT_P53
)

# Load BLOSUM62
blosum62 = substitution_matrices.load("BLOSUM62")

# Grantham distance matrix
grantham = {
    ('A','R'):112, ('A','N'):111, ('A','D'):126, ('A','C'):195,
    ('A','Q'):91, ('A','E'):107, ('A','G'):60, ('A','H'):86,
    ('A','I'):94, ('A','L'):96, ('A','K'):106, ('A','M'):84,
    ('A','F'):113, ('A','P'):27, ('A','S'):99, ('A','T'):58,
    ('A','W'):148, ('A','Y'):112, ('A','V'):64,
}


def get_grantham(wt, mut):
    """Get Grantham distance between two amino acids."""
    if wt == mut:
        return 0
    if (wt, mut) in grantham:
        return grantham[(wt, mut)]
    if (mut, wt) in grantham:
        return grantham[(mut, wt)]
    return 100  # Default moderate penalty


def extract_handcrafted_features(df):
    """
    Extract handcrafted features from mutations.
    
    Features:
    - Position: Actual position (1-393)
    - Normalized_Position: Position / 393
    - BLOSUM62_Score: Biochemical similarity
    - Grantham_Distance: Property change measure
    - DNA_Binding_Domain: Binary flag (1 if 102-292, else 0)
    """
    
    print("\n" + "="*70)
    print("EXTRACTING HANDCRAFTED FEATURES")
    print("="*70)
    
    positions = []
    norm_positions = []
    blosum_scores = []
    grantham_scores = []
    domain_flags = []
    
    for idx, row in df.iterrows():
        
        if (idx + 1) % max(1, len(df) // 10) == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")
        
        # Skip cross-species sequences (no mutation info)
        if row.get("Sequence_Type") != "Human_Mutation":
            positions.append(None)
            norm_positions.append(None)
            blosum_scores.append(None)
            grantham_scores.append(None)
            domain_flags.append(None)
            continue
        
        # Extract mutation info
        mutation = row.get("mutation", "")
        
        if not mutation or len(mutation) < 3:
            positions.append(None)
            norm_positions.append(None)
            blosum_scores.append(None)
            grantham_scores.append(None)
            domain_flags.append(None)
            continue
        
        try:
            wt = mutation[0]
            mut = mutation[-1]
            pos = int(mutation[1:-1])
            
            positions.append(pos)
            norm_positions.append(pos / 393)
            
            # BLOSUM62 score
            try:
                bl_score = blosum62[wt][mut]
            except KeyError:
                bl_score = 0
            blosum_scores.append(bl_score)
            
            # Grantham distance
            grantham_scores.append(get_grantham(wt, mut))
            
            # DNA-binding domain flag
            if DNA_BINDING_DOMAIN_START <= pos <= DNA_BINDING_DOMAIN_END:
                domain_flags.append(1)
            else:
                domain_flags.append(0)
                
        except Exception as e:
            print(f"    Warning: Could not parse mutation {mutation}: {e}")
            positions.append(None)
            norm_positions.append(None)
            blosum_scores.append(None)
            grantham_scores.append(None)
            domain_flags.append(None)
    
    df["Position"] = positions
    df["Normalized_Position"] = norm_positions
    df["BLOSUM62_Score"] = blosum_scores
    df["Grantham_Distance"] = grantham_scores
    df["DNA_Binding_Domain"] = domain_flags
    
    print("\n✓ Features extracted:")
    print(f"  Position: {sum(1 for p in positions if p is not None)} valid")
    print(f"  BLOSUM62_Score: Added")
    print(f"  Grantham_Distance: Added")
    print(f"  DNA_Binding_Domain: Added")
    
    return df


if __name__ == "__main__":
    
    print(f"Wild-type p53 length: {len(WT_P53)} amino acids")
    print(f"DNA-binding domain: residues {DNA_BINDING_DOMAIN_START}-{DNA_BINDING_DOMAIN_END}\n")
    
    # Load aligned dataset
    df = pd.read_csv(ALIGNED_DATASET)
    print(f"Loaded {len(df)} sequences from: {ALIGNED_DATASET}")
    
    # Extract features
    feature_df = extract_handcrafted_features(df)
    
    # Save
    HANDCRAFTED_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(HANDCRAFTED_FEATURES, index=False)
    
    print(f"\n✓ Saved to: {HANDCRAFTED_FEATURES}")
    print(f"  Shape: {feature_df.shape}")
    print(f"  Columns: {list(feature_df.columns)}")
    print("="*70 + "\n")
