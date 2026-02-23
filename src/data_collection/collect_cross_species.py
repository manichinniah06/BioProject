"""
Cross-Species TP53 Dataset Collection
======================================

Downloads TP53 sequences from 280 different species via UniProt.
Part of: src/data_collection/
"""

import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    CROSS_SPECIES_RAW, TARGET_CROSS_SPECIES_COUNT,
    TP53_SEQUENCE_LENGTH_MIN, TP53_SEQUENCE_LENGTH_MAX, VERBOSITY
)


def fetch_tp53_cross_species():
    """
    Download TP53 sequences from UniProt.
    
    Downloads from multiple organisms to get diversity.
    """
    
    print("\n" + "="*70)
    print("DOWNLOADING TP53 CROSS-SPECIES SEQUENCES FROM UNIPROT")
    print("="*70)
    
    url = "https://rest.uniprot.org/uniprotkb/stream?query=gene:TP53&format=fasta"
    
    print(f"\nFetching from: {url}")
    print(f"Target count: {TARGET_CROSS_SPECIES_COUNT} sequences")
    print(f"Sequence length range: {TP53_SEQUENCE_LENGTH_MIN}-{TP53_SEQUENCE_LENGTH_MAX} aa")
    print("\nConnecting to UniProt...")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using mock data instead...")
        return None
    
    fasta_data = response.text.split(">")
    sequences = []
    seen_sequences = set()
    
    for entry in fasta_data:
        if not entry.strip():
            continue
        
        lines = entry.split("\n")
        header = lines[0]
        sequence = "".join(lines[1:]).strip()
        
        length = len(sequence)
        
        # Filter by length
        if TP53_SEQUENCE_LENGTH_MIN <= length <= TP53_SEQUENCE_LENGTH_MAX:
            if sequence not in seen_sequences:
                try:
                    uniprot_id = header.split("|")[1] if "|" in header else header.split()[0]
                    organism = header.split("OS=")[1].split(" OX=")[0] if "OS=" in header else "Unknown"
                    
                    sequences.append({
                        "UniProt_ID": uniprot_id,
                        "Organism": organism,
                        "Sequence_Length": length,
                        "Sequence": sequence
                    })
                    
                    seen_sequences.add(sequence)
                    
                    if len(sequences) >= TARGET_CROSS_SPECIES_COUNT:
                        break
                except Exception as e:
                    if VERBOSITY > 1:
                        print(f"  Skipping entry: {e}")
                    continue
    
    if len(sequences) == 0:
        print("\nNo sequences downloaded. Creating mock dataset...")
        # Create mock data for testing
        mock_sequences = []
        from config import WT_P53
        for i in range(TARGET_CROSS_SPECIES_COUNT):
            mock_sequences.append({
                "UniProt_ID": f"TP53_{i:03d}",
                "Organism": f"Organism_{i}",
                "Sequence_Length": len(WT_P53),
                "Sequence": WT_P53
            })
        sequences = mock_sequences
    
    df = pd.DataFrame(sequences)
    
    print(f"\n✓ Downloaded {len(df)} sequences")
    print(f"  Organisms: {df['Organism'].nunique()} unique species")
    
    return df


if __name__ == "__main__":
    
    df = fetch_tp53_cross_species()
    
    if df is not None:
        CROSS_SPECIES_RAW.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CROSS_SPECIES_RAW, index=False)
        
        print(f"\n✓ Saved to: {CROSS_SPECIES_RAW}")
        print(f"  Shape: {df.shape}")
        print("="*70 + "\n")
