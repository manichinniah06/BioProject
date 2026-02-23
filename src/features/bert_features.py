"""
ESM-2 Protein Language Model Feature Extraction for TP53 Mutations
==================================================================

Uses Facebook/Meta's ESM-2 (Evolutionary Scale Modeling) model - the
state-of-the-art protein language model (equivalent to BioBERT for proteins).

Key Advantages over handcrafted features:
- Captures long-range amino acid dependencies
- Pre-trained on 250M+ protein sequences (UniRef50)
- Understands evolutionary conservation
- Better generalization to unseen mutations

ESM-2 Model Options (by size / quality tradeoff):
1. esm2_8m   - facebook/esm2_t6_8M_UR50D   (8M  params) - Fastest, CPU-friendly
2. esm2_35m  - facebook/esm2_t12_35M_UR50D (35M params) - Recommended for this project
3. esm2_150m - facebook/esm2_t30_150M_UR50D(150M params) - Higher quality, slower
4. esm2_650m - facebook/esm2_t33_650M_UR50D(650M params) - Best, needs GPU / RAM

Part of: src/features/
"""

import torch
import numpy as np
import pandas as pd
from transformers import EsmTokenizer, EsmModel
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import HANDCRAFTED_FEATURES, BERT_FEATURES, BERT_MODEL_NAME, BERT_DEVICE

# ============================================================================
# ESM-2 MODEL REGISTRY
# ============================================================================

MODELS = {
    "esm2_8m": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "embedding_dim": 320,
        "size": "8M params",
        "speed": "Very Fast",
        "quality": "Good (CPU-friendly)"
    },
    "esm2_35m": {
        "name": "facebook/esm2_t12_35M_UR50D",
        "embedding_dim": 480,
        "size": "35M params",
        "speed": "Fast",
        "quality": "Recommended for this project"
    },
    "esm2_150m": {
        "name": "facebook/esm2_t30_150M_UR50D",
        "embedding_dim": 640,
        "size": "150M params",
        "speed": "Moderate",
        "quality": "High"
    },
    "esm2_650m": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "embedding_dim": 1280,
        "size": "650M params",
        "speed": "Slow (needs GPU)",
        "quality": "Best"
    },
    # Legacy ProtBERT fallback (may have tokenizer issues on transformers 5.x)
    "protbert": {
        "name": "Rostlab/prot_bert",
        "embedding_dim": 1024,
        "size": "92M params",
        "speed": "Moderate",
        "quality": "Good (legacy)"
    },
}


class ProteinBERTFeatureExtractor:
    """
    Extract features from protein sequences using Facebook ESM-2 protein
    language model (state-of-the-art, equivalent to BioBERT for proteins).

    Advantages:
    - Pre-trained on 250M+ UniRef50 sequences
    - Understands evolutionary conservation
    - Captures long-range amino acid dependencies
    - Compatible with transformers 5.x out-of-the-box
    """

    # ESM-2 model flag (ProtBERT is legacy and needs different handling)
    _ESM_MODELS = {"esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"}

    def __init__(self, model_name: str = "esm2_35m", device: str = None):
        """
        Initialize the protein language model feature extractor.

        Args:
            model_name: Model key from MODELS dict. Default 'esm2_35m'.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model_name = model_name

        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

        model_config = MODELS[model_name]
        self.model_path = model_config["name"]
        self.embedding_dim = model_config["embedding_dim"]
        self._is_esm = model_name in self._ESM_MODELS

        print(f"Loading {model_name} model...")
        print(f"  Model  : {model_config['name']}")
        print(f"  Size   : {model_config['size']}")
        print(f"  Quality: {model_config['quality']}")
        print(f"  Device : {device}")

        if self._is_esm:
            # ESM-2 - clean API, works with transformers 5.x
            self.tokenizer = EsmTokenizer.from_pretrained(self.model_path)
            self.model = EsmModel.from_pretrained(self.model_path).to(device)
        else:
            # Legacy ProtBERT - needs use_fast=False to avoid sentencepiece error
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_path, do_lower_case=False, use_fast=False
            )
            self.model = BertModel.from_pretrained(self.model_path).to(device)

        self.model.eval()
        print("Model loaded successfully\n")
    
    
    def _prepare_input(self, sequence: str):
        """
        Prepare and tokenize a sequence for the loaded model.
        ESM-2 accepts raw sequence; ProtBERT needs space-separated amino acids.
        Sequences longer than 1022 tokens are truncated safely.
        """
        if self._is_esm:
            seq = sequence[:1022]  # ESM-2 max length
        else:
            seq = " ".join(list(sequence[:510]))  # ProtBERT: space-separated, max 510 AA

        return self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

    def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        Get the full sequence embedding (mean of all token embeddings).

        Args:
            sequence: Protein amino acid sequence (e.g., "MKVLW...")

        Returns:
            Sequence embedding (shape: embedding_dim,)
        """
        inputs = self._prepare_input(sequence)

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
        Get the model embedding for a specific position in the sequence.

        Args:
            sequence: Protein amino acid sequence
            position: 1-based position (e.g., position 175)

        Returns:
            Position embedding (shape: embedding_dim,)
        """
        inputs = self._prepare_input(sequence)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        # Token index = position + 1 (offset for [CLS] token)
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
        
        print(f"✓ Extracted {self.embedding_dim} BERT features per sequence\n")
        
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
        
        print(f"✓ Extracted aggregated BERT features\n")
        
        return result_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("ESM-2 PROTEIN LANGUAGE MODEL FEATURE EXTRACTION FOR TP53")
    print("="*70 + "\n")
    
    # Load the handcrafted features dataset
    print(f"Loading {HANDCRAFTED_FEATURES}...")
    df = pd.read_csv(HANDCRAFTED_FEATURES)
    print(f"  Loaded {len(df)} sequences\n")
    
    # Initialize extractor (using config)
    extractor = ProteinBERTFeatureExtractor(model_name=BERT_MODEL_NAME, device=BERT_DEVICE)
    
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
    print(f"✓ Saved with BERT features to {BERT_FEATURES}")
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
