"""
BEFORE vs AFTER: Comparison of Model Performance
================================================

This script shows how to compare performance before and after adding BERT features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath, keep_bert=False):
    """
    Load data and prepare for ML training.
    
    Args:
        filepath: Path to CSV file
        keep_bert: If True, include BERT features; if False, exclude them
        
    Returns:
        X, y: Features and labels
        feature_names: Names of features used
    """
    df = pd.read_csv(filepath)
    
    # Exclude non-ml columns
    exclude_cols = ['Sequence', 'mutation', 'UniProt_ID', 'Organism', 
                    'sequence_id', 'Sequence_Type']
    
    if not keep_bert:
        # Exclude BERT features - only use handcrafted features
        exclude_cols += [col for col in df.columns if 'BERT' in col]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'label']
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    return X, y, feature_cols


def train_and_evaluate_model(X, y, model_type='RandomForest', model_name='Model'):
    """
    Train model and return detailed metrics.
    
    Args:
        X, y: Features and labels
        model_type: Type of model ('RandomForest', 'GradientBoosting', 'LogisticRegression')
        model_name: Display name for results
        
    Returns:
        Dictionary with all metrics
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Metrics
    metrics = {
        'model_name': model_name,
        'model_type': model_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': (y_pred[y_test == 1] == 1).sum() / max((y_pred == 1).sum(), 1),  # Avoid division by zero
        'recall': (y_pred[y_test == 1] == 1).sum() / max((y_test == 1).sum(), 1),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model,
        'scaler': scaler
    }
    
    return metrics


def print_comparison(results_before, results_after):
    """
    Print detailed comparison between before and after BERT.
    
    Args:
        results_before: Dict with metrics before BERT
        results_after: Dict with metrics after BERT
    """
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: HANDCRAFTED FEATURES vs HANDCRAFTED + BERT FEATURES")
    print("="*80)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'cv_mean']
    
    print(f"\n{'Metric':<20} {'Before BERT':<15} {'After BERT':<15} {'Improvement':<15}")
    print("-"*80)
    
    for metric in metrics_to_compare:
        before = results_before[metric]
        after = results_after[metric]
        improvement = ((after - before) / before * 100) if before != 0 else 0
        
        if metric == 'cv_mean':
            print(f"{metric:<20} {before:.4f} ± {results_before['cv_std']:.4f}  {after:.4f} ± {results_after['cv_std']:.4f}  {improvement:+.2f}%")
        else:
            print(f"{metric:<20} {before:.4f}         {after:.4f}         {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT (After BERT)")
    print("="*80)
    print(classification_report(results_after['y_test'], results_after['y_pred'],
                               target_names=['Non-Damaging (0)', 'Damaging (1)']))
    
    print("\nCONFUSION MATRIX (After BERT):")
    print(confusion_matrix(results_after['y_test'], results_after['y_pred']))


def print_feature_importance(model, feature_names, top_n=15):
    """
    Print top important features from the model.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance extraction.")
        return
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTOP {top_n} MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for idx, row in importance_df.head(top_n).iterrows():
        bar_length = int(row['Importance'] * 100)
        bar = '█' * bar_length
        print(f"{row['Feature']:<30} {row['Importance']:>7.4f}  {bar}")


def generate_comparison_report(handcrafted_file, bert_file=None):
    """
    Generate complete comparison report.
    
    Args:
        handcrafted_file: Path to original features CSV
        bert_file: Path to BERT-enhanced features CSV (optional)
    """
    
    print("\n" + "🔬 "*40)
    print("TP53 MUTATION PREDICTION: MODEL PERFORMANCE COMPARISON")
    print("🔬 "*40)
    
    # ===== SCENARIO 1: Handcrafted Features Only =====
    print("\n\n1️⃣  BASELINE: Using Handcrafted Features Only")
    print("-" * 80)
    
    X_hand, y, feature_names_hand = load_and_prepare_data(handcrafted_file, keep_bert=False)
    print(f"Features used: {len(feature_names_hand)}")
    print(f"  {feature_names_hand}")
    print(f"\nDataset: {len(X_hand)} samples")
    print(f"  Class 0 (Non-damaging): {(y==0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")
    print(f"  Class 1 (Damaging): {(y==1).sum()} ({100*(y==1).sum()/len(y):.1f}%)")
    
    results_handcrafted = {}
    for model_type, model_display in [('RandomForest', 'Random Forest Classifier'),
                                       ('GradientBoosting', 'Gradient Boosting Classifier')]:
        print(f"\nTraining {model_display}...")
        results = train_and_evaluate_model(X_hand, y, model_type=model_type, 
                                          model_name=f"{model_display} (Handcrafted)")
        results_handcrafted[model_type] = results
        print(f"✓ Accuracy: {results['accuracy']:.4f}, AUC-ROC: {results['auc_roc']:.4f}")
    
    # ===== SCENARIO 2: Handcrafted + BERT Features =====
    if bert_file:
        print("\n\n2️⃣  ENHANCED: Using Handcrafted + BERT Features")
        print("-" * 80)
        
        X_bert, _, feature_names_bert = load_and_prepare_data(bert_file, keep_bert=True)
        print(f"Features used: {len(feature_names_bert)}")
        bert_only = [f for f in feature_names_bert if 'BERT' in f]
        print(f"  Handcrafted: {len(feature_names_bert) - len(bert_only)} features")
        print(f"  BERT: {len(bert_only)} features")
        print(f"    {bert_only}")
        
        results_bert = {}
        for model_type, model_display in [('RandomForest', 'Random Forest Classifier'),
                                          ('GradientBoosting', 'Gradient Boosting Classifier')]:
            print(f"\nTraining {model_display}...")
            results = train_and_evaluate_model(X_bert, y, model_type=model_type,
                                              model_name=f"{model_display} (With BERT)")
            results_bert[model_type] = results
            print(f"✓ Accuracy: {results['accuracy']:.4f}, AUC-ROC: {results['auc_roc']:.4f}")
        
        # ===== COMPARISON =====
        print("\n" + "="*80)
        print("DETAILED COMPARISON: RANDOM FOREST CLASSIFIER")
        print("="*80)
        print_comparison(results_handcrafted['RandomForest'], results_bert['RandomForest'])
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS (Random Forest with BERT)")
        print("="*80)
        print_feature_importance(results_bert['RandomForest']['model'], feature_names_bert, top_n=15)
        
        # Summary
        improvement_auc = (results_bert['RandomForest']['auc_roc'] - 
                          results_handcrafted['RandomForest']['auc_roc'])
        improvement_acc = (results_bert['RandomForest']['accuracy'] - 
                          results_handcrafted['RandomForest']['accuracy'])
        
        print("\n" + "🎯 "*40)
        print("SUMMARY")
        print("🎯 "*40)
        print(f"\nAccuracy Improvement: {improvement_acc:+.4f} ({improvement_acc/results_handcrafted['RandomForest']['accuracy']*100:+.1f}%)")
        print(f"AUC-ROC Improvement: {improvement_auc:+.4f} ({improvement_auc/results_handcrafted['RandomForest']['auc_roc']*100:+.1f}%)")
        
        if improvement_auc > 0.02:
            print("\n✅ BERT features significantly improved performance!")
            print("   Recommendation: Use the BERT-enhanced model in production")
        elif improvement_auc > 0:
            print("\n✅ BERT features provided modest improvement")
            print("   Recommendation: BERT features worth using if computational cost is acceptable")
        else:
            print("\n⚠️  BERT features did not improve performance")
            print("   Recommendation: Try mutation-specific BERT features (mutation_bert_features.py)")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Option 1: Compare handcrafted features only
    print("Note: This script requires your CSV files to exist.")
    print("\nUsage:")
    print("  1. Run: python bert_feature_extraction.py")
    print("  2. Run: python comparison.py")
    print("\n" + "="*80)
    
    # Example comparison (uncomment when you have both files)
    try:
        generate_comparison_report(
            handcrafted_file='tp53_important_features.csv',
            bert_file='tp53_with_bert_features.csv'
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nFirst, run: python bert_feature_extraction.py")
        print("This will create tp53_with_bert_features.csv")
        print("\nThen run: python comparison.py")
