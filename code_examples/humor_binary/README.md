# Humor Detection Binary Classification

A machine learning system for detecting humor in text using traditional ML techniques. This binary classifier can distinguish between humorous and non-humorous sentences.

## Features

- Supports multiple classifiers (SVM, Random Forest, Logistic Regression, etc.)
- Ensemble learning with voting classifier
- Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Command-line interface with customizable parameters
- Handles both numeric and text labels (0/1, humor/not humor)

## Requirements

```
scikit-learn==1.7.0
pandas==2.3.0
numpy==2.3.0
tabulate==0.9.0
```

***ကိုယ့်စက်ထဲမှာက အထက်ပါ ဗားရှင်နံပါတ်အတိအကျနဲ့ အဆင်မပြေရင် ဗာရှင်းကို မသတ်မှတ်ပဲ install လုပ်ပါ***  

## Usage

### Basic Command  

```
python ml_humor_detection.py --train_file train.csv --test_file test.csv
```



