"""
Master Pipeline for TP53 Mutation Prediction
==============================================

Orchestrates the entire workflow from data collection to feature extraction.
Run this to execute the complete pipeline.

Usage:
    python pipeline.py [--skip-collection] [--skip-alignment] [--skip-bert]
"""

import subprocess
import sys
from pathlib import Path
import argparse
from config import (
    PROJECT_ROOT, HANDCRAFTED_FEATURES, BERT_FEATURES,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# ============================================================================
# PIPELINE STAGES
# ============================================================================

STAGES = [
    {
        "name": "Data Collection",
        "description": "Generate human mutations and download cross-species sequences",
        "skip_arg": "skip_collection",
        "scripts": [
            "src/data_collection/collect_mutations.py",
            "src/data_collection/collect_cross_species.py",
        ]
    },
    {
        "name": "Data Preprocessing",
        "description": "Remove redundancy from sequences",
        "skip_arg": None,
        "scripts": [
            "src/preprocessing/redundancy_removal.py",
        ]
    },
    {
        "name": "Sequence Alignment",
        "description": "Perform global alignment against wild-type p53",
        "skip_arg": "skip_alignment",
        "scripts": [
            "src/alignment/global_alignment.py",
        ]
    },
    {
        "name": "Feature Extraction",
        "description": "Extract handcrafted and BERT-based features",
        "skip_arg": None,
        "scripts": [
            "src/features/extract_features.py",
        ]
    },
    {
        "name": "BERT Feature Extraction (Optional)",
        "description": "Add deep learning-based BERT features",
        "skip_arg": "skip_bert",
        "scripts": [
            "src/features/bert_features.py",
        ]
    },
]


def run_script(script_path):
    """
    Run a Python script and return success status.
    
    Args:
        script_path: Relative path to script from project root
        
    Returns:
        True if successful, False otherwise
    """
    full_path = PROJECT_ROOT / script_path
    
    print(f"\n{'─'*70}")
    print(f"Running: {script_path}")
    print(f"Full path: {full_path}")
    print(f"{'─'*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {script_path} completed successfully")
            return True
        else:
            print(f"✗ {script_path} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_path}: {e}")
        return False


def print_header():
    """Print pipeline header."""
    print("\n" + "="*70)
    print("TP53 MUTATION IMPACT PREDICTION PIPELINE")
    print("="*70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {RAW_DATA_DIR}")
    print(f"Output Directory: {PROCESSED_DATA_DIR}")
    print("\n" + "="*70)


def print_summary(results):
    """Print pipeline execution summary."""
    print("\n\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    for stage, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {stage}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    failed = total - passed
    
    print("\n" + "-"*70)
    print(f"Total: {total} stages")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*70)
    
    if failed == 0:
        print("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"\nFinal outputs:")
        if HANDCRAFTED_FEATURES.exists():
            size = HANDCRAFTED_FEATURES.stat().st_size / (1024*1024)
            print(f"  • Handcrafted features: {HANDCRAFTED_FEATURES.name} ({size:.1f} MB)")
        if BERT_FEATURES.exists():
            size = BERT_FEATURES.stat().st_size / (1024*1024)
            print(f"  • BERT features: {BERT_FEATURES.name} ({size:.1f} MB)")
    else:
        print(f"\n⚠️  PIPELINE FAILED at some stages (see above for details)")
    
    print("\n" + "="*70 + "\n")
    
    return failed == 0


def main():
    """Main pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description="TP53 Mutation Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                    # Run complete pipeline
  python pipeline.py --skip-bert        # Skip BERT extraction (faster)
  python pipeline.py --skip-collection  # Skip data download (use existing)
  python pipeline.py --help             # Show this help message
        """
    )
    
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection stage')
    parser.add_argument('--skip-alignment', action='store_true',
                       help='Skip alignment stage')
    parser.add_argument('--skip-bert', action='store_true',
                       help='Skip BERT feature extraction (faster)')
    
    args = parser.parse_args()
    
    print_header()
    
    results = []
    
    # Execute each stage
    for stage in STAGES:
        
        # Check if we should skip this stage
        if stage["skip_arg"] and getattr(args, stage["skip_arg"]):
            print(f"\n⊘ Skipping: {stage['name']}")
            results.append((stage["name"], True))
            continue
        
        print(f"\n\n{'='*70}")
        print(f"STAGE: {stage['name']}")
        print(f"{'='*70}")
        print(f"Description: {stage['description']}")
        
        stage_success = True
        for script in stage["scripts"]:
            if not run_script(script):
                stage_success = False
                break  # Stop on first failure
        
        results.append((stage["name"], stage_success))
        
        if not stage_success:
            print(f"\n✗ Stage '{stage['name']}' failed. Stopping pipeline.")
            break
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
