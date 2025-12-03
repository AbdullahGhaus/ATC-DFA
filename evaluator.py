"""
Evaluation and Analysis Module
Provides comprehensive evaluation metrics and analysis
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import numpy as np
from collections import defaultdict


def evaluate_model(y_true, y_pred, pos_label="spam"):
    """
    Comprehensive model evaluation
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    }
    
    return metrics


def analyze_keyword_performance(df, keywords, classifier_func):
    """
    Analyze which keywords are most effective
    """
    keyword_stats = defaultdict(lambda: {'detections': 0, 'true_positives': 0, 'false_positives': 0})
    
    for idx, row in df.iterrows():
        text = row['v2']
        true_label = row['v1']
        predicted = classifier_func(text)
        
        # Check which keyword triggered the detection
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                keyword_stats[keyword]['detections'] += 1
                if predicted == 'spam' and true_label == 'spam':
                    keyword_stats[keyword]['true_positives'] += 1
                elif predicted == 'spam' and true_label == 'ham':
                    keyword_stats[keyword]['false_positives'] += 1
    
    return dict(keyword_stats)


def train_test_evaluation(df, classifier_func, test_size=0.3, random_state=42):
    """
    Perform train-test split evaluation
    """
    X = df['v2']
    y = df['v1']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Predict on test set
    y_pred = X_test.apply(classifier_func)
    
    metrics = evaluate_model(y_test, y_pred)
    
    return {
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def cross_validation_evaluation(df, classifier_func, cv_folds=5):
    """
    Perform k-fold cross-validation
    """
    X = df['v2'].values
    y = df['v1'].values
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create temporary dataframe for this fold
        test_df = pd.DataFrame({'v2': X_test, 'v1': y_test})
        y_pred = test_df['v2'].apply(classifier_func)
        
        fold_metrics = evaluate_model(y_test, y_pred)
        
        for metric in cv_scores:
            cv_scores[metric].append(fold_metrics[metric])
    
    # Calculate mean and std
    cv_results = {}
    for metric in cv_scores:
        cv_results[f'{metric}_mean'] = np.mean(cv_scores[metric])
        cv_results[f'{metric}_std'] = np.std(cv_scores[metric])
    
    return cv_results


def generate_detailed_report(df, y_pred, metrics, keyword_stats=None):
    """
    Generate a detailed text report
    """
    report = []
    report.append("=" * 70)
    report.append("DFA SPAM DETECTOR - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Dataset Information
    report.append("DATASET INFORMATION")
    report.append("-" * 70)
    report.append(f"Total Messages: {len(df)}")
    report.append(f"Ham Messages: {len(df[df['v1'] == 'ham'])} ({len(df[df['v1'] == 'ham'])/len(df)*100:.1f}%)")
    report.append(f"Spam Messages: {len(df[df['v1'] == 'spam'])} ({len(df[df['v1'] == 'spam'])/len(df)*100:.1f}%)")
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 70)
    report.append(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    report.append(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    report.append(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    report.append("")
    
    # Confusion Matrix
    cm = confusion_matrix(df['v1'], y_pred, labels=['ham', 'spam'])
    report.append("CONFUSION MATRIX")
    report.append("-" * 70)
    report.append(f"                Predicted")
    report.append(f"              Ham    Spam")
    report.append(f"Actual  Ham   {cm[0][0]:4d}   {cm[0][1]:4d}")
    report.append(f"        Spam  {cm[1][0]:4d}   {cm[1][1]:4d}")
    report.append("")
    
    # Classification Report
    report.append("DETAILED CLASSIFICATION REPORT")
    report.append("-" * 70)
    report.append(classification_report(df['v1'], y_pred, target_names=['Ham', 'Spam']))
    
    # Keyword Analysis
    if keyword_stats:
        report.append("KEYWORD PERFORMANCE ANALYSIS")
        report.append("-" * 70)
        for keyword, stats in keyword_stats.items():
            report.append(f"Keyword: {keyword.upper()}")
            report.append(f"  Total Detections: {stats['detections']}")
            report.append(f"  True Positives: {stats['true_positives']}")
            report.append(f"  False Positives: {stats['false_positives']}")
            if stats['detections'] > 0:
                precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
                report.append(f"  Precision: {precision:.2%}")
            report.append("")
    
    # Sample Predictions
    report.append("SAMPLE PREDICTIONS")
    report.append("-" * 70)
    for idx, row in df.head(10).iterrows():
        status = "[OK]" if row['v1'] == y_pred.iloc[idx] else "[X]"
        report.append(f"{status} [{row['v1']:4s}] -> [{y_pred.iloc[idx]:4s}] | {row['v2'][:50]}...")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)

