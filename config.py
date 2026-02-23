"""
Configuration file for TP53 Mutation Prediction Project
========================================================

Centralized settings for all paths, parameters, and configurations.
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Source code directories
SRC_DIR = PROJECT_ROOT / "src"
DATA_COLLECTION_DIR = SRC_DIR / "data_collection"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
ALIGNMENT_DIR = SRC_DIR / "alignment"
FEATURES_DIR = SRC_DIR / "features"
EVALUATION_DIR = SRC_DIR / "evaluation"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILE PATHS
# ============================================================================

# Raw data files
HUMAN_MUTATIONS_RAW = RAW_DATA_DIR / "p53_missense_mutation_dataset.csv"
CROSS_SPECIES_RAW = RAW_DATA_DIR / "tp53_cross_species_dataset_280.csv"

# Processed data files (intermediate)
COMBINED_DATASET = PROCESSED_DATA_DIR / "tp53_final_combined_dataset.csv"
ALIGNED_DATASET = PROCESSED_DATA_DIR / "tp53_with_global_alignment.csv"

# Feature extraction outputs
HANDCRAFTED_FEATURES = PROCESSED_DATA_DIR / "tp53_important_features.csv"
BERT_FEATURES = PROCESSED_DATA_DIR / "tp53_with_bert_features.csv"
MUTATION_BERT_FEATURES = PROCESSED_DATA_DIR / "tp53_with_mutation_bert.csv"

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================

# UniProt data collection
UNIPROT_QUERY = "gene:TP53"
TP53_SEQUENCE_LENGTH_MIN = 350
TP53_SEQUENCE_LENGTH_MAX = 450
TARGET_CROSS_SPECIES_COUNT = 280

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Redundancy removal
REDUNDANCY_THRESHOLD = 0.90  # Cosine similarity threshold
KMER_SIZE = 3

# Global alignment
ALIGNMENT_MODE = "global"
BLOSUM_MATRIX = "BLOSUM62"
GAP_OPEN = -10
GAP_EXTEND = -0.5

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

# DNA-binding domain range
DNA_BINDING_DOMAIN_START = 102
DNA_BINDING_DOMAIN_END = 292

# BERT model configuration
# ESM-2 options: esm2_8m, esm2_35m, esm2_150m, esm2_650m (recommended: esm2_35m)
# Legacy ProtBERT (may not work on transformers 5.x): protbert
BERT_MODEL_NAME = "esm2_35m"
BERT_BATCH_SIZE = 32
BERT_DEVICE = "cpu"  # Change to "cuda" if GPU available

# ============================================================================
# WILD-TYPE TP53 SEQUENCE
# ============================================================================

WT_P53 = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAA"
    "PPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKT"
    "CPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDR"
    "NTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACP"
    "GRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEA"
    "LELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

# Verify sequence length
assert len(WT_P53) == 393, f"WT p53 sequence must be 393 aa, got {len(WT_P53)}"

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

# Verbosity level (0=silent, 1=info, 2=debug)
VERBOSITY = 1

# Print paths
PRINT_PATHS_ON_STARTUP = True

# ============================================================================
# PRINT CONFIG INFO (Optional)
# ============================================================================

def print_config():
    """Print current configuration (useful for debugging)."""
    if PRINT_PATHS_ON_STARTUP:
        print("\n" + "="*70)
        print("PROJECT CONFIGURATION")
        print("="*70)
        print(f"\nProject Root: {PROJECT_ROOT}")
        print(f"\nData Directories:")
        print(f"  Raw Data:       {RAW_DATA_DIR}")
        print(f"  Processed Data: {PROCESSED_DATA_DIR}")
        print(f"\nSource Code: {SRC_DIR}")
        print(f"\nOutput:     {OUTPUT_DIR}")
        print(f"\nWT p53 Length: {len(WT_P53)} amino acids")
        print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
