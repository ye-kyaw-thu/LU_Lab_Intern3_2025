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

## Input File Format

```
text,label
"funny sentence here",1
"serious sentence here",0
```

Labels can be:  
- Numeric: 0 (not humor) or 1 (humor)  
- Text: "humor"/"not humor", "true"/"false", etc.  

## Output

The program will display:  
- Training progress for each classifier
- Evaluation metrics (accuracy, precision, recall, F1)  
- Classification report  
- Comparison table when running all classifiers  
- Identification of the best performing model  
    
## Example Runs  

### Dataset Information

Link: [https://www.kaggle.com/datasets/amaanmansuri/humor-detection](https://www.kaggle.com/datasets/amaanmansuri/humor-detection)  

```
humor_50k.csv  humor_50k.shuf.csv  test.csv  train.csv
(humor) ye@lst-hpc3090:~/intern3/humor/kaggle/data$ wc {train,test}.csv
  45001  319336 2061615 train.csv
   5000   35510  228886 test.csv
  50001  354846 2290501 total
```

```
(humor) ye@lst-hpc3090:~/intern3/humor/kaggle/data$ head ./train.csv
truli headscratch moment donald trump press confer infrastructur,0
indian chief friend got indian chief tattoo arm arm never work,1
destin wed tip bridesmaid budget,0
dark alley johnni optimist beat half life,1
horni pirat worst nightmar sunken chest booti,1
mathematician work home function domain,1
bought drug shoe dealer unlac still got high heel,1
virginia museum open costa concordia exhibit memori day weekend photo,0
peopl nt realiz chickpea get everi manpea make,1
betabrand hire graduat student model result pretti great,0
(humor) ye@lst-hpc3090:~/intern3/humor/kaggle/data$
```

```
(humor) ye@lst-hpc3090:~/intern3/humor/kaggle/data$ tail ./test.csv
rob gray man bun ncaa tournament,0
texa put undu burden women choic abort clinic tell suprem court,0
alabama gop told gay sheriff candid run republican,0
nt phone sex might get hear aid,1
not lazi energi save mode,1
q nt blond like butter toast ca nt figur side butter goe,1
like coffe like like tea hot splash milk,1
hope nt take joke liter pleas return later,1
syrian increasingli desper escap wartorn countri,0
herb chicken thigh roast paper bag,0
(humor) ye@lst-hpc3090:~/intern3/humor/kaggle/data$
```

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

### Run Only Random-Forest

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

### Run Only Random-Forest with Specific Settings

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --classifier random_forest --train_file ./kaggle/data/train.csv --test_file ./kaggle/data/test.csv --rf_n_estimators 200 --rf_max_depth 10
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Training Random Forest...

Evaluation Results:
Accuracy: 0.7466
Precision: 0.8299
Recall: 0.6321
F1 Score: 0.7176

Classification Report:
              precision    recall  f1-score   support

           0       0.69      0.87      0.77      2453
           1       0.83      0.63      0.72      2547

    accuracy                           0.75      5000
   macro avg       0.76      0.75      0.74      5000
weighted avg       0.76      0.75      0.74      5000


real    0m1.901s
user    0m6.903s
sys     0m0.101s
```

### Run All Classifiers (default)

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection.py --train_file ./kaggle/data/train.csv -
-test_file ./kaggle/data/test.csv
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Running all classifiers...

Training Svm...
Completed in 242.97 seconds
Accuracy: 0.8618
F1 Score: 0.8648

Training Random Forest...
Completed in 29.69 seconds
Accuracy: 0.8416
F1 Score: 0.8480

Training Logistic Regression...
Completed in 0.04 seconds
Accuracy: 0.8652
F1 Score: 0.8680

Training Naive Bayes...
Completed in 0.01 seconds
Accuracy: 0.8658
F1 Score: 0.8687

Training Knn...
Completed in 1.55 seconds
Accuracy: 0.6062
F1 Score: 0.7073

Training Decision Tree...
Completed in 6.03 seconds
Accuracy: 0.7998
F1 Score: 0.8038

Training Adaboost...
Completed in 1.22 seconds
Accuracy: 0.6028
F1 Score: 0.7137

Training Gradient Boosting...
Completed in 5.93 seconds
Accuracy: 0.7214
F1 Score: 0.6734

Training Voting...
Completed in 270.63 seconds
Accuracy: 0.8716
F1 Score: 0.8751


Comparison of All Classifiers:
+---------------------+------------+-------------+----------+------------+---------+
| Model               |   Accuracy |   Precision |   Recall |   F1 Score | Time    |
+=====================+============+=============+==========+============+=========+
| Svm                 |     0.8618 |      0.8619 |   0.8677 |     0.8648 | 242.97s |
+---------------------+------------+-------------+----------+------------+---------+
| Random Forest       |     0.8416 |      0.8295 |   0.8673 |     0.848  | 29.69s  |
+---------------------+------------+-------------+----------+------------+---------+
| Logistic Regression |     0.8652 |      0.866  |   0.87   |     0.868  | 0.04s   |
+---------------------+------------+-------------+----------+------------+---------+
| Naive Bayes         |     0.8658 |      0.8658 |   0.8716 |     0.8687 | 0.01s   |
+---------------------+------------+-------------+----------+------------+---------+
| Knn                 |     0.6062 |      0.5691 |   0.934  |     0.7073 | 1.55s   |
+---------------------+------------+-------------+----------+------------+---------+
| Decision Tree       |     0.7998 |      0.8024 |   0.8053 |     0.8038 | 6.03s   |
+---------------------+------------+-------------+----------+------------+---------+
| Adaboost            |     0.6028 |      0.5639 |   0.9717 |     0.7137 | 1.22s   |
+---------------------+------------+-------------+----------+------------+---------+
| Gradient Boosting   |     0.7214 |      0.8359 |   0.5638 |     0.6734 | 5.93s   |
+---------------------+------------+-------------+----------+------------+---------+
| Voting              |     0.8716 |      0.8671 |   0.8834 |     0.8751 | 270.63s |
+---------------------+------------+-------------+----------+------------+---------+

Best model: Voting with F1 score: 0.8751

real    9m19.121s
user    9m25.260s
sys     0m1.258s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

## Limitations of Version 0.9

**Note**: This code serves as an educational example for internship students and has several technical limitations:

- **No model persistence**: Trained models are not saved to disk for future use
- **No modular execution**: Cannot run only training or only testing separately
- **Basic text preprocessing**: Limited to TF-IDF with minimal text cleaning
- **Fixed hyperparameters**: No built-in hyperparameter tuning (grid/random search)
- **Memory intensive**: Loads entire datasets into memory (not suitable for very large datasets)
- **No cross-validation**: Uses simple train-test split without k-fold validation
- **English-only**: Text processing assumes English language content

## Version 1.0 Updates

Major improvements over previous version:

### New Features
- **Multilingual Support**:
  - Removed English-only limitations
  - Added UTF-8 encoding for text processing (i.e. Support for Myanmar, Thai, Lao and other languages)  
  - Basic token pattern suitable for multiple scripts

- **Model Persistence**:
  - Save trained models to disk (.joblib format)
  - Load pre-trained models for evaluation
  - Includes both classifier and vectorizer in saved files

- **Enhanced Testing Mode**:
  - New `--test` flag for test-only execution
  - Specify model file with `--model_file`
  - Supports testing individual saved models

## Usage

```
(humor) ye@lst-hpc3090:~/intern3/humor$ python3.13 ./ml_humor_detection_ver1.0.py --help
usage: ml_humor_detection_ver1.0.py [-h] [--train_file TRAIN_FILE] --test_file TEST_FILE [--test] [--model_file MODEL_FILE]
                                    [--classifier {all,svm,random_forest,logistic_regression,naive_bayes,knn,decision_tree,adaboost,gradient_boosting,voting}]
                                    [--save_dir SAVE_DIR] [--svm_kernel {linear,rbf,poly,sigmoid}] [--svm_c SVM_C]
                                    [--rf_n_estimators RF_N_ESTIMATORS] [--rf_max_depth RF_MAX_DEPTH] [--lr_c LR_C]
                                    [--lr_penalty {l1,l2,elasticnet,none}] [--nb_alpha NB_ALPHA] [--knn_n_neighbors KNN_N_NEIGHBORS]
                                    [--dt_max_depth DT_MAX_DEPTH] [--ab_n_estimators AB_N_ESTIMATORS]
                                    [--ab_learning_rate AB_LEARNING_RATE] [--gb_n_estimators GB_N_ESTIMATORS]
                                    [--gb_learning_rate GB_LEARNING_RATE]

Humor Detection using Traditional ML Techniques (Version 1.0)

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        Path to training CSV file (default: None)
  --test_file TEST_FILE
                        Path to testing CSV file (default: None)
  --test                Run in test-only mode (default: False)
  --model_file MODEL_FILE
                        Path to model file for test-only mode (default: None)
  --classifier {all,svm,random_forest,logistic_regression,naive_bayes,knn,decision_tree,adaboost,gradient_boosting,voting}
                        Classifier to use (ignored in test-only mode) (default: all)
  --save_dir SAVE_DIR   Directory to save trained models (default: models)
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

## Example Runs for Version 1.0 

### 1. Training and Testing All Classifiers

```
time python3.13 ./ml_humor_detection_ver1.0.py --train_file kaggle/data/train.csv --test_file kaggle/data/test.csv | tee ver1.train_all.log2
Unique labels in training set: {np.int64(0), np.int64(1)}
Unique labels in test set: {np.int64(0), np.int64(1)}
Running all classifiers...

Training Svm...
Model saved to models/svm_model.joblib
Completed in 250.94 seconds
Accuracy: 0.8786
F1 Score: 0.8811

Training Random Forest...
Model saved to models/random_forest_model.joblib
Completed in 28.09 seconds
Accuracy: 0.8400
F1 Score: 0.8480

Training Logistic Regression...
Model saved to models/logistic_regression_model.joblib
Completed in 0.10 seconds
Accuracy: 0.8810
F1 Score: 0.8829

Training Naive Bayes...
Model saved to models/naive_bayes_model.joblib
Completed in 0.04 seconds
Accuracy: 0.8744
F1 Score: 0.8775

Training Knn...
Model saved to models/knn_model.joblib
Completed in 1.58 seconds
Accuracy: 0.6644

Training Decision Tree...
Model saved to models/decision_tree_model.joblib
Completed in 8.69 seconds
Accuracy: 0.7824
F1 Score: 0.7917

Training Adaboost...
Model saved to models/adaboost_model.joblib
Completed in 1.39 seconds
Accuracy: 0.6444
F1 Score: 0.5251

Training Gradient Boosting...
Model saved to models/gradient_boosting_model.joblib
Completed in 6.63 seconds
Accuracy: 0.7526
F1 Score: 0.7261

Training Voting...
Model saved to models/voting_model.joblib
Completed in 282.18 seconds
Accuracy: 0.8826
F1 Score: 0.8860


Comparison of All Classifiers:
+---------------------+------------+-------------+----------+------------+---------+
| Model               |   Accuracy |   Precision |   Recall |   F1 Score | Time    |
+=====================+============+=============+==========+============+=========+
| Svm                 |     0.8786 |      0.8789 |   0.8834 |     0.8811 | 250.94s |
+---------------------+------------+-------------+----------+------------+---------+
| Random Forest       |     0.84   |      0.8217 |   0.8759 |     0.848  | 28.09s  |
+---------------------+------------+-------------+----------+------------+---------+
| Logistic Regression |     0.881  |      0.8852 |   0.8806 |     0.8829 | 0.10s   |
+---------------------+------------+-------------+----------+------------+---------+
| Naive Bayes         |     0.8744 |      0.872  |   0.883  |     0.8775 | 0.04s   |
+---------------------+------------+-------------+----------+------------+---------+
| Knn                 |     0.6644 |      0.8932 |   0.3875 |     0.5405 | 1.58s   |
+---------------------+------------+-------------+----------+------------+---------+
| Decision Tree       |     0.7824 |      0.7725 |   0.8119 |     0.7917 | 8.69s   |
+---------------------+------------+-------------+----------+------------+---------+
| Adaboost            |     0.6444 |      0.8212 |   0.3859 |     0.5251 | 1.39s   |
+---------------------+------------+-------------+----------+------------+---------+
| Gradient Boosting   |     0.7526 |      0.8325 |   0.6439 |     0.7261 | 6.63s   |
+---------------------+------------+-------------+----------+------------+---------+
| Voting              |     0.8826 |      0.8763 |   0.896  |     0.886  | 282.18s |
+---------------------+------------+-------------+----------+------------+---------+

Best model: Voting with F1 score: 0.8860
Model saved to: models/voting_model.joblib

real    9m40.673s
user    9m47.655s
sys     0m1.141s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

### Testing with Pre-trained Random Forest Model  

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection_ver1.0.py --test --model_file models/random_forest_model.joblib --test_file kaggle/data/test.csv
Loading model from models/random_forest_model.joblib...
Unique labels in test set: {np.int64(0), np.int64(1)}

Evaluation Results:
Accuracy: 0.8400
Precision: 0.8217
Recall: 0.8759
F1 Score: 0.8480

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.80      0.83      2453
           1       0.82      0.88      0.85      2547

    accuracy                           0.84      5000
   macro avg       0.84      0.84      0.84      5000
weighted avg       0.84      0.84      0.84      5000


real    0m0.944s
user    0m5.865s
sys     0m0.181s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

### Testing with Pre-trained SVM Model 

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection_ver1.0.py --test --model_file models/svm_model.joblib --test_file kaggle/data/test.csv
Loading model from models/svm_model.joblib...
Unique labels in test set: {np.int64(0), np.int64(1)}

Evaluation Results:
Accuracy: 0.8786
Precision: 0.8789
Recall: 0.8834
F1 Score: 0.8811

Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.87      0.88      2453
           1       0.88      0.88      0.88      2547

    accuracy                           0.88      5000
   macro avg       0.88      0.88      0.88      5000
weighted avg       0.88      0.88      0.88      5000


real    0m3.077s
user    0m8.115s
sys     0m0.065s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

### Testing with Pre-trained KNN Model  

```
(humor) ye@lst-hpc3090:~/intern3/humor$ time python3.13 ./ml_humor_detection_ver1.0.py --test --model_file models/knn_model.joblib --te
st_file kaggle/data/test.csv
Loading model from models/knn_model.joblib...
Unique labels in test set: {np.int64(0), np.int64(1)}

Evaluation Results:
Accuracy: 0.6644
Precision: 0.8932
Recall: 0.3875
F1 Score: 0.5405

Classification Report:
              precision    recall  f1-score   support

           0       0.60      0.95      0.74      2453
           1       0.89      0.39      0.54      2547

    accuracy                           0.66      5000
   macro avg       0.75      0.67      0.64      5000
weighted avg       0.75      0.66      0.64      5000


real    0m2.104s
user    0m6.405s
sys     0m0.801s
(humor) ye@lst-hpc3090:~/intern3/humor$
```

