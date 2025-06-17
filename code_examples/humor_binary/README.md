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

**ကိုယ့်စက်ထဲမှာက အထက်ပါ ဗားရှင်နံပါတ်အတိအကျနဲ့ အဆင်မပြေရင် ဗာရှင်းကို မသတ်မှတ်ပဲ install လုပ်ပါ**  

## Usage

### Basic Command  

```
python ml_humor_detection.py --train_file train.csv --test_file test.csv
```

### Command Line Options  

```
(humor) ye@lst-hpc3090:~/intern3/humor$ python3.13 ./ml_humor_detection.py --help
usage: ml_humor_detection.py [-h] [--train_file TRAIN_FILE] [--test_file TEST_FILE]
                             [--classifier {all,svm,random_forest,logistic_regression,naive_bayes,knn,decision_tree,adaboost,gradient_boosting,voting}]
                             [--svm_kernel {linear,rbf,poly,sigmoid}] [--svm_c SVM_C] [--rf_n_estimators RF_N_ESTIMATORS]
                             [--rf_max_depth RF_MAX_DEPTH] [--lr_c LR_C] [--lr_penalty {l1,l2,elasticnet,none}]
                             [--nb_alpha NB_ALPHA] [--knn_n_neighbors KNN_N_NEIGHBORS] [--dt_max_depth DT_MAX_DEPTH]
                             [--ab_n_estimators AB_N_ESTIMATORS] [--ab_learning_rate AB_LEARNING_RATE]
                             [--gb_n_estimators GB_N_ESTIMATORS] [--gb_learning_rate GB_LEARNING_RATE]

Humor Detection using Traditional ML Techniques

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        Path to training CSV file (default: train.csv)
  --test_file TEST_FILE
                        Path to testing CSV file (default: test.csv)
  --classifier {all,svm,random_forest,logistic_regression,naive_bayes,knn,decision_tree,adaboost,gradient_boosting,voting}
                        Classifier to use (default: all)
  --svm_kernel {linear,rbf,poly,sigmoid}
                        Kernel type for SVM (default: linear)
  --svm_c SVM_C         Regularization parameter for SVM (default: 1.0)
  --rf_n_estimators RF_N_ESTIMATORS
                        Number of trees in Random Forest (default: 100)
  --rf_max_depth RF_MAX_DEPTH
                        Maximum depth of trees in Random Forest (default: None)
  --lr_c LR_C           Inverse of regularization strength for Logistic Regression (default: 1.0)
  --lr_penalty {l1,l2,elasticnet,none}
                        Penalty norm for Logistic Regression (default: l2)
  --nb_alpha NB_ALPHA   Additive smoothing parameter for Naive Bayes (default: 1.0)
  --knn_n_neighbors KNN_N_NEIGHBORS
                        Number of neighbors for KNN (default: 5)
  --dt_max_depth DT_MAX_DEPTH
                        Maximum depth for Decision Tree (default: None)
  --ab_n_estimators AB_N_ESTIMATORS
                        Number of estimators for AdaBoost (default: 50)
  --ab_learning_rate AB_LEARNING_RATE
                        Learning rate for AdaBoost (default: 1.0)
  --gb_n_estimators GB_N_ESTIMATORS
                        Number of estimators for Gradient Boosting (default: 100)
  --gb_learning_rate GB_LEARNING_RATE
                        Learning rate for Gradient Boosting (default: 0.1)
(humor) ye@lst-hpc3090:~/intern3/humor$
```

## Example Runs  

### Run Only SVM

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --classifier svm --train_file ./kaggle
/data/train.csv --test_file ./kaggle/data/test.csv
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Training Svm...

Evaluation Results:
Accuracy: 0.8618
Precision: 0.8619
Recall: 0.8677
F1 Score: 0.8648

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.86      0.86      2453
           1       0.86      0.87      0.86      2547

    accuracy                           0.86      5000
   macro avg       0.86      0.86      0.86      5000
weighted avg       0.86      0.86      0.86      5000


real    3m59.265s
user    4m4.133s
sys     0m0.209s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

### Run Only KNN

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --classifier knn --train_file ./kaggle
/data/train.csv --test_file ./kaggle/data/test.csv
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Training Knn...

Evaluation Results:
Accuracy: 0.6034
Precision: 0.5673
Recall: 0.9333
F1 Score: 0.7057

Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.26      0.39      2453
           1       0.57      0.93      0.71      2547

    accuracy                           0.60      5000
   macro avg       0.68      0.60      0.55      5000
weighted avg       0.68      0.60      0.55      5000


real    0m2.517s
user    0m6.819s
sys     0m0.799s
```

### Run Only Adaboost

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --classifier adaboost --train_file ./k
aggle/data/train.csv --test_file ./kaggle/data/test.csv
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Training Adaboost...

Evaluation Results:
Accuracy: 0.6028
Precision: 0.5639
Recall: 0.9717
F1 Score: 0.7137

Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.22      0.35      2453
           1       0.56      0.97      0.71      2547

    accuracy                           0.60      5000
   macro avg       0.72      0.60      0.53      5000
weighted avg       0.72      0.60      0.54      5000


real    0m2.335s
user    0m7.338s
sys     0m0.099s
```

## Run Only Random-Forest

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --classifier random_forest --train_fil
e ./kaggle/data/train.csv --test_file ./kaggle/data/test.csv
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Training Random Forest...

Evaluation Results:
Accuracy: 0.8416
Precision: 0.8295
Recall: 0.8673
F1 Score: 0.8480

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.81      0.83      2453
           1       0.83      0.87      0.85      2547

    accuracy                           0.84      5000
   macro avg       0.84      0.84      0.84      5000
weighted avg       0.84      0.84      0.84      5000


real    0m30.554s
user    0m35.508s
sys     0m0.147s
```

```

```


