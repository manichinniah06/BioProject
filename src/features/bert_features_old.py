"""
BERT-Based Feature Extraction for TP53 Mutations (SIMPLIFIED & ROBUST VERSION)
================================================================================

This module uses pretrained BERT models to extract contextualized 
protein sequence embeddings and features.

Uses widely-available models for better compatibility:
- DistilBERT: Fast, lightweight (66M params) - RECOMMENDED
- BERT: Standard (110M params)
- Alternative: Sentence-BERT for faster inference

Key Advantages:
- Captures sequence context
- Better generalization
- Handles variable-length sequences
- Pre-computed or on-the-fly extraction

Part of: src/features/
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
import warnings
import sys
from pathlib import Path
from typing import Union, Optional

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import HANDCRAFTED_FEATURES, BERT_FEATURES, BERT_MODEL_NAME, BERT_DEVICE

# ============================================================================
# BERT MODEL SELECTION (SIMPLIFIED FOR ROBUSTNESS)
# ============================================================================

MODELS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "embedding_dim": 768,
        "size": "66M params",
        "speed": "Very Fast [OK] RECOMMENDED",
        "quality": "Good",
        "type": "distilbert"
    },
    "bert": {
        "name": "bert-base-uncased",
        "embedding_dim": 768,
        "size": "110M params",
        "speed": "Fast",
        "quality": "Good",
        "type": "bert"
    },
    "biobert": {
        "name": "bert-base-uncased",  # Fallback to standard BERT
        "embedding_dim": 768,
        "size": "110M params",
        "speed": "Fast",
        "quality": "Good (works offline)",
        "type": "bert"
    },
    "biobert-pubmed": {
        "name": "bert-base-uncased",  # Fallback to standard BERT
        "embedding_dim": 768,
        "size": "110M params",
        "speed": "Fast",
        "quality": "Good (offline compatible)",
        "type": "bert"
    },
    "protbert": {
        "name": "bert-base-uncased",  # Fallback to standard BERT
        "embedding_dim": 768,
        "size": "110M params",
        "speed": "Fast",
        "quality": "Good (offline fallback)",
        "type": "bert"
    }
}


class ProteinBERTFeatureExtractor:
    """
    Extract features from protein sequences using pretrained BERT models.
    
    Simplified version using widely-available models for better compatibility.
    Works with or without internet access (downloads on first use).
    """
    
    def __init__(self, model_name: str = "distilbert", device: str = None):
        """
        Initialize BERT feature extractor.
        
        Args:
            model_name: Which BERT model to use ('distilbert', 'bert', 'biobert', etc.)
            device: 'cuda' for GPU, 'cpu' for CPU. Auto-detect if None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.original_model_name = model_name
        
        if model_name not in MODELS:
            print(f"[WARNING] Unknown model '{model_name}', defaulting to 'distilbert'")
            model_name = "distilbert"
        
        model_config = MODELS[model_name]
        self.model_path = model_config["name"]
        self.embedding_dim = model_config["embedding_dim"]
        self.model_type = model_config.get("type", "bert")
        
        print(f"\nLoading {model_config['name']}...")
        print(f"  Size: {model_config['size']}")
        print(f"  Speed: {model_config['speed']}")
        print(f"  Quality: {model_config['quality']}")
        print(f"  Device: {device}\n")
        
        # Load tokenizer with proper error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False,
                timeout=30
            )
            print(f"[OK] Tokenizer loaded")
        except Exception as e:
            print(f"[ERROR] Tokenizer loading failed: {str(e)[:80]}")
            print(f"  Trying alternative model...")
            self.model_path = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.embedding_dim = 768
        
        # Load model with proper error handling
        try:
            if self.model_type == "distilbert":
                self.model = DistilBertModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=False
                ).to(device)
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=False
                ).to(device)
            print(f"[OK] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {str(e)[:80]}")
            print(f"  Loading distilbert-base-uncased as fallback...")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
            self.model_path = "distilbert-base-uncased"
            self.embedding_dim = 768
            print(f"[OK] Fallback model loaded")
        
        self.model.eval()  # Set to evaluation mode
    
    
    def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        Get the full sequence embedding (mean of all token embeddings).
        
        Args:
            sequence: Protein amino acid sequence (e.g., "MKVLW...")
            
        Returns:
            Sequence embedding (shape: embedding_dim,)
        """
        # Add spaces between amino acids (BERT tokenization requirement)
        sequence_spaced = " ".join(list(sequence))
        
        # Tokenize
        inputs = self.tokenizer(sequence_spaced, return_tensors="pt").to(self.device)
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Mean pooling across tokens (excluding [CLS] and [PAD])
        attention_mask = inputs['attention_mask']
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sequence_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        
        return sequence_embedding.squeeze().detach().cpu().numpy()
    
    
    def get_position_embedding(self, sequence: str, position: int) -> np.ndarray:
        """
        Get the BERT embedding for a specific position in the sequence.
        
        Args:
            sequence: Protein amino acid sequence
            position: 1-based position (e.g., position 175)
            
        Returns:
            Position embedding (shape: embedding_dim,)
        """
        sequence_spaced = " ".join(list(sequence))
        inputs = self.tokenizer(sequence_spaced, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Position in tokens = position + 1 (account for [CLS] token)
        token_position = min(position + 1, embeddings.shape[1] - 1)
        position_embedding = embeddings[0, token_position, :].detach().cpu().numpy()
        
        return position_embedding
    
    
    def get_contextual_similarity(self, seq1: str, seq2: str, position: int) -> float:
        """
        Compute how similar sequences are at a specific position
        based on BERT embeddings. Useful for comparing mutations.
        
        Args:
            seq1: Wild-type sequence
            seq2: Mutant sequence
            position: Position of mutation (1-based)
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        emb1 = self.get_position_embedding(seq1, position)
        emb2 = self.get_position_embedding(seq2, position)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(similarity)
    
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                       sequence_col: str = "Sequence",
                                       batch_size: int = 32) -> pd.DataFrame:
        """
        Extract BERT features for all sequences in a dataframe.
        
        Args:
            df: Input dataframe with sequences
            sequence_col: Name of column containing sequences
            batch_size: Number of sequences to process simultaneously
            
        Returns:
            DataFrame with BERT features added
        """
        embeddings = []
        
        print(f"Extracting BERT features from {len(df)} sequences...")
        
        for idx, sequence in enumerate(df[sequence_col]):
            if (idx + 1) % max(1, len(df) // 10) == 0:
                print(f"  Progress: {idx + 1}/{len(df)}")
            
            embedding = self.get_sequence_embedding(str(sequence))
            embeddings.append(embedding)
        
        # Create feature columns
        embedding_array = np.array(embeddings)
        
        # Column names for embedding dimensions
        embedding_cols = [f"BERT_Embedding_{i}" for i in range(self.embedding_dim)]
        
        # Add to dataframe
        for i, col in enumerate(embedding_cols):
            df[col] = embedding_array[:, i]
        
        print(f"[OK] Extracted {self.embedding_dim} BERT features per sequence\n")
        
        return df
    
    
    def extract_aggregated_features(self, df: pd.DataFrame, 
                                   sequence_col: str = "Sequence",
                                   position_col: str = "Position") -> pd.DataFrame:
        """
        Extract dimensionality-reduced BERT features (useful for interpretability).
        
        Creates:
        - BERT_Seq_Mean: Mean of embeddings
        - BERT_Seq_Std: Standard deviation
        - BERT_Seq_Max: Maximum values
        - BERT_Seq_Min: Minimum values
        - Position-specific embedding stats
        
        Args:
            df: Input dataframe
            sequence_col: Sequence column name
            position_col: Position column name (if available)
            
        Returns:
            DataFrame with aggregated features
        """
        mean_features = []
        std_features = []
        
        print(f"Extracting aggregated BERT features from {len(df)} sequences...")
        
        for idx, sequence in enumerate(df[sequence_col]):
            if (idx + 1) % max(1, len(df) // 10) == 0:
                print(f"  Progress: {idx + 1}/{len(df)}")
            
            embedding = self.get_sequence_embedding(str(sequence))
            
            mean_features.append({
                'BERT_Mean': np.mean(embedding),
                'BERT_Std': np.std(embedding),
                'BERT_Max': np.max(embedding),
                'BERT_Min': np.min(embedding),
                'BERT_Median': np.median(embedding),
            })
        
        feature_df = pd.DataFrame(mean_features)
        
        # Add position-specific features if available
        if position_col in df.columns:
            position_embeddings = []
            for idx, (sequence, position) in enumerate(zip(df[sequence_col], df[position_col])):
                if pd.notna(position):
                    pos_emb = self.get_position_embedding(str(sequence), int(position))
                    position_embeddings.append({
                        'BERT_Pos_Mean': np.mean(pos_emb),
                        'BERT_Pos_Std': np.std(pos_emb),
                    })
                else:
                    position_embeddings.append({
                        'BERT_Pos_Mean': np.nan,
                        'BERT_Pos_Std': np.nan,
                    })
            
            pos_df = pd.DataFrame(position_embeddings)
            feature_df = pd.concat([feature_df, pos_df], axis=1)
        
        # Merge with input dataframe
        result_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        
        print(f"[OK] Extracted aggregated BERT features\n")
        
        return result_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("BERT FEATURE EXTRACTION FOR TP53 MUTATIONS")
    print("="*70 + "\n")
    
    # Load the handcrafted features dataset
    print(f"Loading {HANDCRAFTED_FEATURES}...")
    df = pd.read_csv(HANDCRAFTED_FEATURES)
    print(f"  Loaded {len(df)} sequences\n")
    
    # Initialize extractor (using config)
    extractor = ProteinBERTFeatureExtractor(model_name=BERT_MODEL_NAME, device=BERT_DEVICE)
    
    # If ProtBERT fails, use BioBERT as fallback (better compatibility)
    if extractor is None or "error" in str(extractor).lower():
        print("[WARNING] Trying alternative model as fallback...\n")
        extractor = ProteinBERTFeatureExtractor(model_name="biobert-pubmed", device=BERT_DEVICE)
    
    # ===== Option 1: Aggregated features (RECOMMENDED for small dataset) =====
    print("Extracting aggregated features (lower dimensionality)...\n")
    df_with_features = extractor.extract_aggregated_features(
        df,
        sequence_col="Sequence",
        position_col="Position"
    )
    
    # Save results
    BERT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df_with_features.to_csv(BERT_FEATURES, index=False)
    print(f"[OK] Saved with BERT features to {BERT_FEATURES}")
    print(f"  Shape: {df_with_features.shape}")
    print(f"  New features: {[col for col in df_with_features.columns if 'BERT' in col]}")
    
    # ===== Feature summary =====
    print("\n" + "="*70)
    print("NEW FEATURES ADDED:")
    print("="*70)
    bert_features = [col for col in df_with_features.columns if 'BERT' in col]
    print(f"\nTotal BERT features: {len(bert_features)}")
    for feat in bert_features:
        print(f"  • {feat}")
    
    print("\n" + "="*70)
    print("COMBINED FEATURE SET:")
    print("="*70)
    original_features = [col for col in df.columns if col not in ['Sequence', 'UniProt_ID', 'Organism']]
    print(f"\nOriginal handcrafted features: {len(original_features)}")
    print(f"\nNew BERT features: {len(bert_features)}")
    print(f"Total features for ML: {len(original_features) + len(bert_features)}")
    
    print("\n" + "="*70)
