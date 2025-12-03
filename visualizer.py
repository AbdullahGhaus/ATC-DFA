"""
DFA Visualization Module
Creates visual representations of DFAs and classification results
"""
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    import seaborn as sns
    import numpy as np
    VISUALIZATION_AVAILABLE = True
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    # Try to import graphviz, but don't fail if executable not available
    try:
        from graphviz import Digraph
        GRAPHVIZ_AVAILABLE = True
    except (ImportError, Exception):
        GRAPHVIZ_AVAILABLE = False
except ImportError:
    VISUALIZATION_AVAILABLE = False
    GRAPHVIZ_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Visualizations will be skipped.")
    print("Install with: pip install matplotlib seaborn")

from sklearn.metrics import confusion_matrix, classification_report
import os


def visualize_dfa(dfa, keyword, output_dir="visualizations"):
    """
    Visualize a DFA using matplotlib (fallback if Graphviz not available)
    """
    if not VISUALIZATION_AVAILABLE:
        print(f"Skipping DFA visualization for '{keyword}' (matplotlib not available)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    # Try Graphviz first, fallback to matplotlib
    if GRAPHVIZ_AVAILABLE:
        try:
            dot = Digraph(comment=f'DFA for keyword: {keyword}')
            dot.attr(rankdir='LR', size='8,5')
            dot.attr('node', shape='circle')
            
            # Add all states
            for state in dfa.states:
                if state in dfa.accept_states:
                    dot.node(str(state), str(state), shape='doublecircle', style='filled', fillcolor='lightgreen')
                elif state == dfa.start_state:
                    dot.node(str(state), str(state), style='filled', fillcolor='lightblue')
                else:
                    dot.node(str(state), str(state))
            
            # Add start arrow
            dot.node('start', '', shape='point')
            dot.edge('start', str(dfa.start_state))
            
            # Add transitions (only show important ones to avoid clutter)
            shown_transitions = set()
            for (state, char), next_state in dfa.transitions.items():
                if next_state != 0 or state == 0:
                    key = (state, next_state, char)
                    if key not in shown_transitions:
                        if state == 0 and next_state == 1:
                            dot.edge(str(state), str(next_state), label=char, color='red', penwidth='2')
                        elif next_state == state + 1:
                            dot.edge(str(state), str(next_state), label=char, color='blue')
                        elif next_state != 0:
                            dot.edge(str(state), str(next_state), label=char, style='dashed', color='gray')
                        shown_transitions.add(key)
            
            output_path = os.path.join(output_dir, f'dfa_{keyword}.png')
            dot.render(output_path.replace('.png', ''), format='png', cleanup=True)
            print(f"DFA visualization saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Graphviz rendering failed, using matplotlib fallback: {e}")
    
    # Fallback to matplotlib visualization
    return visualize_dfa_matplotlib(dfa, keyword, output_dir)


def visualize_dfa_matplotlib(dfa, keyword, output_dir="visualizations"):
    """
    Visualize DFA using matplotlib (no Graphviz required)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, len(dfa.states) * 2 + 1)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title(f'DFA State Diagram for Keyword: "{keyword.upper()}"', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Position states horizontally
    state_positions = {}
    for i, state in enumerate(dfa.states):
        x = i * 2
        y = 0
        state_positions[state] = (x, y)
    
    # Draw states
    for state in dfa.states:
        x, y = state_positions[state]
        
        if state in dfa.accept_states:
            # Double circle for accept state
            circle1 = Circle((x, y), 0.4, color='green', fill=True, alpha=0.3, zorder=2)
            circle2 = Circle((x, y), 0.35, color='white', fill=True, zorder=3)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            color = 'green'
        elif state == dfa.start_state:
            # Highlighted start state
            circle = Circle((x, y), 0.4, color='lightblue', fill=True, alpha=0.5, zorder=2)
            ax.add_patch(circle)
            color = 'blue'
        else:
            circle = Circle((x, y), 0.4, color='black', fill=False, linewidth=2, zorder=2)
            ax.add_patch(circle)
            color = 'black'
        
        # State label
        ax.text(x, y, str(state), ha='center', va='center', 
               fontsize=14, fontweight='bold', zorder=4, color='black' if state in dfa.accept_states else 'black')
    
    # Draw start arrow
    start_x, start_y = state_positions[dfa.start_state]
    ax.annotate('', xy=(start_x - 0.5, start_y), xytext=(start_x - 1.2, start_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(start_x - 1.5, start_y, 'Start', ha='right', va='center', fontsize=10)
    
    # Draw transitions (show forward transitions and key ones)
    shown_transitions = set()
    for (state, char), next_state in dfa.transitions.items():
        if next_state != 0 or state == 0:
            key = (state, next_state)
            if key not in shown_transitions and state < len(dfa.states) - 1:
                x1, y1 = state_positions[state]
                x2, y2 = state_positions[next_state]
                
                # Forward transition
                if next_state == state + 1:
                    color = 'blue'
                    style = 'solid'
                    lw = 2
                elif state == 0:
                    color = 'red'
                    style = 'solid'
                    lw = 2.5
                else:
                    color = 'gray'
                    style = 'dashed'
                    lw = 1
                
                # Draw arrow
                arrow = FancyArrowPatch((x1 + 0.4, y1), (x2 - 0.4, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       color=color, linestyle=style, linewidth=lw, zorder=1)
                ax.add_patch(arrow)
                
                # Label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y + 0.3, char, ha='center', va='bottom',
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                shown_transitions.add(key)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', alpha=0.5, label='Start State'),
        mpatches.Patch(facecolor='lightgreen', alpha=0.3, label='Accept State'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Forward Transition'),
        plt.Line2D([0], [0], color='red', lw=2, label='First Character'),
        plt.Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Other Transitions')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'dfa_{keyword}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"DFA visualization saved to: {output_path}")
    plt.close()
    return output_path


def plot_confusion_matrix(y_true, y_pred, output_dir="visualizations"):
    """
    Create and save a confusion matrix heatmap
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping confusion matrix (matplotlib not available)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - DFA Spam Detector', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
             transform=plt.gca().transAxes, 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()
    return output_path


def plot_metrics_comparison(metrics_dict, output_dir="visualizations"):
    """
    Create a bar chart comparing different metrics
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping metrics comparison (matplotlib not available)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('DFA Spam Detector - Performance Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 1.0
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {output_path}")
    plt.close()
    return output_path


def plot_keyword_analysis(keywords, keyword_stats, output_dir="visualizations"):
    """
    Visualize which keywords are most effective
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping keyword analysis (matplotlib not available)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Detection count
    detection_counts = [keyword_stats.get(kw, {}).get('detections', 0) for kw in keywords]
    ax1.barh(keywords, detection_counts, color='steelblue')
    ax1.set_xlabel('Number of Detections', fontsize=11)
    ax1.set_title('Keyword Detection Frequency', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (kw, count) in enumerate(zip(keywords, detection_counts)):
        ax1.text(count, i, f' {count}', va='center', fontsize=10)
    
    # Plot 2: True positives vs false positives
    true_positives = [keyword_stats.get(kw, {}).get('true_positives', 0) for kw in keywords]
    false_positives = [keyword_stats.get(kw, {}).get('false_positives', 0) for kw in keywords]
    
    x = np.arange(len(keywords))
    width = 0.35
    
    ax2.bar(x - width/2, true_positives, width, label='True Positives', color='green', alpha=0.7)
    ax2.bar(x + width/2, false_positives, width, label='False Positives', color='red', alpha=0.7)
    ax2.set_xlabel('Keywords', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Keyword Performance Analysis', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(keywords, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'keyword_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Keyword analysis saved to: {output_path}")
    plt.close()
    return output_path


def plot_classification_distribution(y_true, y_pred, output_dir="visualizations"):
    """
    Visualize the distribution of predictions vs actual labels
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping label distribution (matplotlib not available)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Actual distribution
    unique, counts = np.unique(y_true, return_counts=True)
    ax1.pie(counts, labels=unique, autopct='%1.1f%%', startangle=90, 
            colors=['#3498db', '#e74c3c'])
    ax1.set_title('Actual Label Distribution', fontsize=13, fontweight='bold')
    
    # Predicted distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax2.pie(counts_pred, labels=unique_pred, autopct='%1.1f%%', startangle=90,
            colors=['#3498db', '#e74c3c'])
    ax2.set_title('Predicted Label Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Label distribution saved to: {output_path}")
    plt.close()
    return output_path

