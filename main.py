"""
DFA Spam Detector - Main Application
Comprehensive spam detection using Deterministic Finite Automata
"""
import pandas as pd
import os
from classifier import build_keyword_dfa, classify_message
from evaluator import (evaluate_model, analyze_keyword_performance, 
                       train_test_evaluation, cross_validation_evaluation,
                       generate_detailed_report)
from visualizer import (visualize_dfa, plot_confusion_matrix, 
                       plot_metrics_comparison, plot_keyword_analysis,
                       plot_classification_distribution)


def main():
    """
    Main execution function
    """
    print("=" * 70)
    print("DFA SPAM DETECTOR - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")
    print(f"Dataset loaded: {len(df)} messages")
    print()
    
    # Define keywords (can be expanded)
    keywords = ["win", "free", "congratulations", "prize", "urgent", "click", "limited"]
    print(f"Keywords for detection: {', '.join([kw.upper() for kw in keywords])}")
    print()
    
    # Build DFAs
    print("Building DFAs...")
    dfas = [build_keyword_dfa(k) for k in keywords]
    print(f"Created {len(dfas)} DFAs")
    print()
    
    # Create classifier function
    def classifier_func(text):
        return classify_message(dfas, text)
    
    # Full dataset evaluation
    print("=" * 70)
    print("FULL DATASET EVALUATION")
    print("=" * 70)
    df["predicted"] = df["v2"].apply(classifier_func)
    
    metrics = evaluate_model(df["v1"], df["predicted"])
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print()
    
    # Train-Test Split Evaluation
    print("=" * 70)
    print("TRAIN-TEST SPLIT EVALUATION (70/30)")
    print("=" * 70)
    tt_results = train_test_evaluation(df, classifier_func, test_size=0.3)
    print(f"\nTrain Size: {tt_results['train_size']}")
    print(f"Test Size: {tt_results['test_size']}")
    print("\nTest Set Performance:")
    for metric, value in tt_results['metrics'].items():
        print(f"  {metric.capitalize()}: {value:.4f} ({value*100:.2f}%)")
    print()
    
    # Cross-Validation
    print("=" * 70)
    print("CROSS-VALIDATION EVALUATION (5-Fold)")
    print("=" * 70)
    cv_results = cross_validation_evaluation(df, classifier_func, cv_folds=5)
    print("\nCross-Validation Results (Mean ± Std):")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean = cv_results[f'{metric}_mean']
        std = cv_results[f'{metric}_std']
        print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f} ({mean*100:.2f}% ± {std*100:.2f}%)")
    print()
    
    # Keyword Analysis
    print("=" * 70)
    print("KEYWORD PERFORMANCE ANALYSIS")
    print("=" * 70)
    keyword_stats = analyze_keyword_performance(df, keywords, classifier_func)
    for keyword, stats in keyword_stats.items():
        print(f"\n{keyword.upper()}:")
        print(f"  Total Detections: {stats['detections']}")
        print(f"  True Positives: {stats['true_positives']}")
        print(f"  False Positives: {stats['false_positives']}")
        if stats['detections'] > 0:
            precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
            print(f"  Precision: {precision:.2%}")
    print()
    
    # Generate Visualizations
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize DFAs for first 3 keywords
    print("\n1. Creating DFA state diagrams...")
    for keyword in keywords[:3]:
        visualize_dfa(dfas[keywords.index(keyword)], keyword)
    
    # Confusion Matrix
    print("\n2. Creating confusion matrix...")
    plot_confusion_matrix(df["v1"], df["predicted"])
    
    # Metrics Comparison
    print("\n3. Creating metrics comparison chart...")
    plot_metrics_comparison(metrics)
    
    # Keyword Analysis
    print("\n4. Creating keyword analysis charts...")
    plot_keyword_analysis(keywords, keyword_stats)
    
    # Label Distribution
    print("\n5. Creating label distribution charts...")
    plot_classification_distribution(df["v1"], df["predicted"])
    
    print("\nAll visualizations saved to 'visualizations/' directory")
    print()
    
    # Generate Detailed Report
    print("=" * 70)
    print("GENERATING DETAILED REPORT")
    print("=" * 70)
    report = generate_detailed_report(df, df["predicted"], metrics, keyword_stats)
    
    # Save report
    report_path = "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Display report (handle encoding for Windows console)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print("\n" + report.encode('ascii', 'ignore').decode('ascii'))
    
    # Export results to CSV
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    results_df = df[['v1', 'v2', 'predicted']].copy()
    results_df['correct'] = results_df['v1'] == results_df['predicted']
    results_path = "classification_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results exported to: {results_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nFiles generated:")
    print("  - visualizations/dfa_*.png (DFA state diagrams)")
    print("  - visualizations/confusion_matrix.png")
    print("  - visualizations/metrics_comparison.png")
    print("  - visualizations/keyword_analysis.png")
    print("  - visualizations/label_distribution.png")
    print("  - analysis_report.txt (detailed text report)")
    print("  - classification_results.csv (results with predictions)")
    print()


if __name__ == "__main__":
    main()
