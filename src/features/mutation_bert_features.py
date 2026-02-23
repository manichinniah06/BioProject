"""
Advanced BERT Features for Mutation Impact Prediction
=======================================================

Extract BERT features that specifically capture mutation effects:
1. Wild-type vs Mutant embedding differences
2. Local context around mutation site
3. Embedding-based physicochemical property changes
4. Structural perturbation scores
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


class MutationSpecificBERTExtractor:
    """
    Extract mutation-specific features by comparing wild-type and mutant sequences.
    """
    
    def __init__(self, model_name: str = "protbert", wt_sequence: str = None, device: str = None):
        """
        Initialize mutation-specific extractor.
        
        Args:
            model_name: Which BERT model to use
            wt_sequence: Wild-type TP53 sequence (for all mutations)
            device: 'cuda' or 'cpu'
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = model_name
        self.wt_sequence = wt_sequence
        
        # Load model
        models = {
            "protbert": "Rostlab/prot_bert",
            "protbert_bfd": "Rostlab/prot_bert_bfd",
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(models[model_name], do_lower_case=False)
        self.model = AutoModel.from_pretrained(models[model_name]).to(device)
        self.model.eval()
    
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """Get BERT embedding for a sequence."""
        sequence_spaced = " ".join(list(sequence))
        inputs = self.tokenizer(sequence_spaced, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        avg_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        
        return avg_embedding.squeeze().detach().cpu().numpy()
    
    
    def get_local_context_embedding(self, sequence: str, position: int, window: int = 5) -> np.ndarray:
        """
        Get embedding of local context around mutation site.
        
        Args:
            sequence: Full sequence
            position: Mutation position (1-based)
            window: Number of residues on each side
            
        Returns:
            Local embedding (shape: embedding_dim,)
        """
        start = max(0, position - window - 1)
        end = min(len(sequence), position + window)
        local_seq = sequence[start:end]
        
        return self.get_embedding(local_seq)
    
    
    def extract_mutation_features(self, df: pd.DataFrame, 
                                 sequence_col: str = "Sequence",
                                 position_col: str = "Position",
                                 mutation_col: str = "mutation") -> pd.DataFrame:
        """
        Extract comprehensive mutation-specific features.
        
        Features extracted:
        1. Full sequence embedding distance
        2. Position-specific embedding distance
        3. Local context embedding distance
        4. Mutation severity based on embedding
        5. Context complexity score
        """
        
        features_list = []
        
        # Precompute wild-type embeddings
        print("Computing wild-type baseline embeddings...")
        wt_full_emb = self.get_embedding(self.wt_sequence)
        wt_local_embs = {}
        
        for pos in range(1, len(self.wt_sequence) + 1):
            wt_local_embs[pos] = self.get_local_context_embedding(self.wt_sequence, pos)
        
        print(f"Processing {len(df)} mutations...")
        
        for idx, row in df.iterrows():
            if (idx + 1) % max(1, len(df) // 10) == 0:
                print(f"  Progress: {idx + 1}/{len(df)}")
            
            sequence = str(row[sequence_col])
            position = int(row[position_col]) if pd.notna(row[position_col]) else None
            
            features = {}
            
            if position is not None and 1 <= position <= len(self.wt_sequence):
                # 1. Full sequence embedding distance
                mut_full_emb = self.get_embedding(sequence)
                full_seq_dist = np.linalg.norm(wt_full_emb - mut_full_emb)
                
                # 2. Local context embedding distance
                mut_local_emb = self.get_local_context_embedding(sequence, position)
                wt_local_emb = wt_local_embs[position]
                local_dist = np.linalg.norm(wt_local_emb - mut_local_emb)
                
                # 3. Cosine similarity
                wt_norm = wt_full_emb / (np.linalg.norm(wt_full_emb) + 1e-8)
                mut_norm = mut_full_emb / (np.linalg.norm(mut_full_emb) + 1e-8)
                cosine_sim = np.dot(wt_norm, mut_norm)
                
                # 4. Local context complexity (entropy-like measure)
                context_std = np.std(mut_local_emb)
                context_complexity = context_std
                
                # 5. Perturbation score (combination of effects)
                perturbation_score = full_seq_dist * (1 - cosine_sim) / (local_dist + 1e-8)
                
                features = {
                    'BERT_Full_Seq_Distance': full_seq_dist,
                    'BERT_Local_Distance': local_dist,
                    'BERT_Cosine_Similarity': cosine_sim,
                    'BERT_Context_Complexity': context_complexity,
                    'BERT_Perturbation_Score': perturbation_score,
                    'BERT_Effect_Magnitude': (full_seq_dist + local_dist) / 2,
                }
            else:
                features = {
                    'BERT_Full_Seq_Distance': np.nan,
                    'BERT_Local_Distance': np.nan,
                    'BERT_Cosine_Similarity': np.nan,
                    'BERT_Context_Complexity': np.nan,
                    'BERT_Perturbation_Score': np.nan,
                    'BERT_Effect_Magnitude': np.nan,
                }
            
            features_list.append(features)
        
        # Add to dataframe
        features_df = pd.DataFrame(features_list)
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        print(f"✓ Extracted mutation-specific BERT features\n")
        
        return result


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

def add_bert_to_pipeline(input_file: str = "tp53_important_features.csv",
                        output_file: str = "tp53_with_mutation_bert.csv",
                        wt_sequence: str = None):
    """
    Complete pipeline to add BERT features to existing dataset.
    
    WT sequence should be the wild-type TP53 from your codebase.
    """
    
    # Load the final features dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    # Get wild-type sequence (or use provided one)
    if wt_sequence is None:
        # Extract from human wild-type row(s)
        wt_rows = df[df['Sequence_Type'] == 'Human_Mutation'].iloc[0:1]
        if len(wt_rows) > 0:
            wt_sequence = wt_rows.iloc[0]['Sequence']
        else:
            raise ValueError("Could not find wild-type sequence. Please provide one.")
    
    print(f"Wild-type sequence length: {len(wt_sequence)}")
    
    # Extract mutation-specific features
    extractor = MutationSpecificBERTExtractor(
        model_name="protbert",
        wt_sequence=wt_sequence,
        device="cpu"
    )
    
    result_df = extractor.extract_mutation_features(df)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to {output_file}")
    print(f"  Shape: {result_df.shape}")
    
    # Print feature statistics
    print("\nMutation-Specific BERT Feature Statistics:")
    print("=" * 70)
    bert_cols = [col for col in result_df.columns if 'BERT' in col]
    print(result_df[bert_cols].describe())
    
    return result_df


if __name__ == "__main__":
    # This will be called from the main pipeline or standalone
    
    # Wild-type TP53 sequence (from your codebase)
    WT_P53 = (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAA"
        "PPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKT"
        "CPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDR"
        "NTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACP"
        "GRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEA"
        "LELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    )
    
    print("="*70)
    print("MUTATION-SPECIFIC BERT FEATURE EXTRACTION")
    print("="*70 + "\n")
    
    # Run pipeline
    result_df = add_bert_to_pipeline(
        input_file="tp53_important_features.csv",
        output_file="tp53_with_mutation_bert.csv",
        wt_sequence=WT_P53
    )
