# DFA Spam Detector - Advanced Theory of Computation Project

A comprehensive spam detection system using **Deterministic Finite Automata (DFA)** for pattern matching and keyword detection. This project demonstrates the practical application of automata theory in text classification.

## 🎯 Features

### Core Functionality
- **DFA-based Keyword Detection**: Uses KMP-like automata for efficient substring matching
- **Multiple Keywords Support**: Detects spam based on multiple suspicious keywords
- **Case-Insensitive Matching**: Handles variations in text case

### Advanced Features
- **📊 Comprehensive Visualizations**:
  - DFA state transition diagrams
  - Confusion matrix heatmaps
  - Performance metrics comparison charts
  - Keyword effectiveness analysis
  - Label distribution pie charts

- **📈 Evaluation Metrics**:
  - Full dataset evaluation
  - Train-test split validation (70/30)
  - K-fold cross-validation (5-fold)
  - Detailed classification reports

- **🔍 Analysis Tools**:
  - Keyword performance analysis
  - Detection frequency tracking
  - True/False positive analysis per keyword

- **💻 Interactive Demo**:
  - Real-time message classification
  - Batch processing mode
  - Keyword triggering visualization

- **📄 Report Generation**:
  - Detailed text reports
  - CSV export of results
  - Comprehensive statistics

## 📁 Project Structure

```
.
├── main.py                 # Main application with full analysis
├── dfa.py                  # DFA class implementation
├── classifier.py           # DFA builder and classifier
├── evaluator.py            # Evaluation and analysis functions
├── visualizer.py           # Visualization functions
├── interactive_demo.py     # Interactive classification demo
├── dataset.csv             # Training/test dataset
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Note**: For Graphviz visualizations, you may need to install Graphviz system package:
   - **Windows**: Download from [Graphviz website](https://graphviz.org/download/)
   - **Linux**: `sudo apt-get install graphviz`
   - **Mac**: `brew install graphviz`

## 💻 Usage

### Main Analysis (Recommended)
Run the comprehensive analysis with all visualizations and reports:

```bash
python main.py
```

This will:
- Evaluate the model on the full dataset
- Perform train-test split evaluation
- Run 5-fold cross-validation
- Generate all visualizations
- Create detailed reports
- Export results to CSV

### Interactive Demo
Test messages in real-time:

```bash
python interactive_demo.py
```

Or classify multiple messages at once:
```bash
python interactive_demo.py "Your message here" "Another message"
```

## 📊 Output Files

After running `main.py`, you'll get:

### Visualizations (in `visualizations/` folder):
- `dfa_*.png` - State transition diagrams for each keyword
- `confusion_matrix.png` - Confusion matrix heatmap
- `metrics_comparison.png` - Bar chart of performance metrics
- `keyword_analysis.png` - Keyword effectiveness analysis
- `label_distribution.png` - Distribution of labels

### Reports:
- `analysis_report.txt` - Comprehensive text report
- `classification_results.csv` - All predictions with results

## 🔬 How It Works

### DFA Construction
1. For each keyword, a DFA is built using a KMP-like algorithm
2. States represent progress through the keyword
3. Transitions handle character matching and failure cases
4. Accept states indicate successful keyword detection

### Classification Process
1. Input text is normalized (lowercase)
2. Each DFA processes the text sequentially
3. If any DFA reaches an accept state, message is classified as spam
4. Otherwise, message is classified as ham

### Evaluation
- **Accuracy**: Overall correctness
- **Precision**: Spam predictions that are actually spam
- **Recall**: Actual spam messages that were detected
- **F1 Score**: Harmonic mean of precision and recall

## 📈 Example Output

```
======================================================================
DFA SPAM DETECTOR - COMPREHENSIVE ANALYSIS
======================================================================

Performance Metrics:
  Accuracy:  1.0000 (100.00%)
  Precision: 1.0000 (100.00%)
  Recall:    1.0000 (100.00%)
  F1 Score:  1.0000 (100.00%)
```

## 🎓 Educational Value

This project demonstrates:
- **Automata Theory**: Practical DFA implementation
- **Pattern Matching**: KMP-like substring matching
- **Machine Learning**: Evaluation metrics and validation
- **Data Visualization**: Multiple chart types
- **Software Engineering**: Modular code structure

## 🔧 Customization

### Adding More Keywords
Edit the `keywords` list in `main.py`:
```python
keywords = ["win", "free", "congratulations", "your_keyword"]
```

### Adjusting Evaluation
Modify test size or CV folds in `main.py`:
```python
train_test_evaluation(df, classifier_func, test_size=0.2)  # 80/20 split
cross_validation_evaluation(df, classifier_func, cv_folds=10)  # 10-fold CV
```

## 📝 Dataset Format

The CSV file should have the format:
```csv
v1,v2
ham,"Message text here"
spam,"Another message"
```

Where:
- `v1`: Label (ham or spam)
- `v2`: Message text

## 🤝 Contributing

Feel free to extend this project with:
- More sophisticated DFA patterns
- Additional evaluation metrics
- Web interface
- Real-time API
- More visualization types

## 📄 License

This project is for educational purposes as part of Advanced Theory of Computation course.

## 👨‍💻 Author

Created for Advanced Theory of Computation course project.

---

**Note**: This is an educational project demonstrating DFA applications. For production spam detection, consider more sophisticated methods like machine learning models.

