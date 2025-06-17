# Written by Ye Kyaw Thu, LU Lab., Myanmar
# Humor binary classification with ML approaches
# Version 0.9
# Last updated: 15 June 2025
#
# How to run:
# $ python ./ml_humor_detection.py --help
#
# $ time python3.13 ./ml_humor_detection.py --classifier svm --train_file ./kaggle/data/train.csv --test_file ./kaggle/data/test.csv
#
# $ time python3.13 ./ml_humor_detection.py --train_file ./kaggle/data/train.csv --test_file ./kaggle/data/test.csv

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import time

class HumorDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.classifiers = {
            'svm': {
                'model': SVC(kernel='linear', probability=True),
                'params': {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(n_neighbors=5),
                'params': {
                    'n_neighbors': [3, 5, 7]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [None, 5, 10]
                }
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.5]
                }
            }
        }
        self.ensemble_classifiers = {
            'voting': {
                'model': VotingClassifier(
                    estimators=[
                        ('svm', SVC(kernel='linear', probability=True)),
                        ('rf', RandomForestClassifier(n_estimators=100)),
                        ('lr', LogisticRegression(max_iter=1000))
                    ],
                    voting='soft'
                ),
                'params': {}
            }
        }

    def load_data(self, train_file, test_file):
        """Load and preprocess the data"""
        train_df = pd.read_csv(train_file, header=None, names=['text', 'label'])
        test_df = pd.read_csv(test_file, header=None, names=['text', 'label'])
    
        # Convert labels to integers, handling string labels
        def convert_label(label):
            if isinstance(label, str):
                label = label.lower()
                if label in ['1', 'true', 'yes', 'humor', 'humour']:
                    return 1
                elif label in ['0', 'false', 'no', 'not humor', 'not_humor']:
                    return 0
                else:
                    try:
                        return int(label)
                    except ValueError:
                        return 0  # Default to 0 if unknown string
            else:
                return int(label)
    
        train_df['label'] = train_df['label'].apply(convert_label)
        test_df['label'] = test_df['label'].apply(convert_label)
    
        # Check for missing values
        if train_df.isnull().sum().any() or test_df.isnull().sum().any():
            raise ValueError("Dataset contains missing values")
        
        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_test = test_df['text'].values
        y_test = test_df['label'].values
    
        # Debug print to verify labels
        print("Unique labels in training set:", set(y_train))
        print("Unique labels in test set:", set(y_test))
    
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, X_train, X_test):
        """Vectorize the text data"""
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    def train_classifier(self, classifier_name, X_train, y_train, **kwargs):
        """Train a specific classifier"""
        if classifier_name in self.classifiers:
            model = self.classifiers[classifier_name]['model']
            # Update model parameters if provided
            if kwargs:
                model.set_params(**kwargs)
        elif classifier_name in self.ensemble_classifiers:
            model = self.ensemble_classifiers[classifier_name]['model']
            if kwargs:
                model.set_params(**kwargs)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return metrics"""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': classification_report(y_test, y_pred)
        }

    def run_all_classifiers(self, X_train, y_train, X_test, y_test):
        """Run all classifiers and return results"""
        results = []
        
        for name in self.classifiers:
            start_time = time.time()
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            model = self.train_classifier(name, X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            
            elapsed = time.time() - start_time
            metrics['time'] = elapsed
            metrics['model'] = name.replace('_', ' ').title()
            
            results.append(metrics)
            
            print(f"Completed in {elapsed:.2f} seconds")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Run ensemble classifiers
        for name in self.ensemble_classifiers:
            start_time = time.time()
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            model = self.train_classifier(name, X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            
            elapsed = time.time() - start_time
            metrics['time'] = elapsed
            metrics['model'] = name.replace('_', ' ').title()
            
            results.append(metrics)
            
            print(f"Completed in {elapsed:.2f} seconds")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Humor Detection using Traditional ML Techniques',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--train_file', type=str, default='train.csv',
                      help='Path to training CSV file')
    parser.add_argument('--test_file', type=str, default='test.csv',
                      help='Path to testing CSV file')
    parser.add_argument('--classifier', type=str, default='all',
                      choices=['all', 'svm', 'random_forest', 'logistic_regression', 
                              'naive_bayes', 'knn', 'decision_tree', 'adaboost',
                              'gradient_boosting', 'voting'],
                      help='Classifier to use')
    
    # SVM parameters
    parser.add_argument('--svm_kernel', type=str, default='linear',
                      choices=['linear', 'rbf', 'poly', 'sigmoid'],
                      help='Kernel type for SVM')
    parser.add_argument('--svm_c', type=float, default=1.0,
                      help='Regularization parameter for SVM')
    
    # Random Forest parameters
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                      help='Number of trees in Random Forest')
    parser.add_argument('--rf_max_depth', type=int, default=None,
                      help='Maximum depth of trees in Random Forest')
    
    # Logistic Regression parameters
    parser.add_argument('--lr_c', type=float, default=1.0,
                      help='Inverse of regularization strength for Logistic Regression')
    parser.add_argument('--lr_penalty', type=str, default='l2',
                      choices=['l1', 'l2', 'elasticnet', 'none'],
                      help='Penalty norm for Logistic Regression')
    
    # Naive Bayes parameters
    parser.add_argument('--nb_alpha', type=float, default=1.0,
                      help='Additive smoothing parameter for Naive Bayes')
    
    # KNN parameters
    parser.add_argument('--knn_n_neighbors', type=int, default=5,
                      help='Number of neighbors for KNN')
    
    # Decision Tree parameters
    parser.add_argument('--dt_max_depth', type=int, default=None,
                      help='Maximum depth for Decision Tree')
    
    # AdaBoost parameters
    parser.add_argument('--ab_n_estimators', type=int, default=50,
                      help='Number of estimators for AdaBoost')
    parser.add_argument('--ab_learning_rate', type=float, default=1.0,
                      help='Learning rate for AdaBoost')
    
    # Gradient Boosting parameters
    parser.add_argument('--gb_n_estimators', type=int, default=100,
                      help='Number of estimators for Gradient Boosting')
    parser.add_argument('--gb_learning_rate', type=float, default=0.1,
                      help='Learning rate for Gradient Boosting')
    
    args = parser.parse_args()
    
    # Initialize the humor detector
    detector = HumorDetector()
    
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test = detector.load_data(args.train_file, args.test_file)
        X_train_vec, X_test_vec = detector.preprocess_data(X_train, X_test)
        
        if args.classifier == 'all':
            # Run all classifiers
            print("Running all classifiers...")
            results = detector.run_all_classifiers(X_train_vec, y_train, X_test_vec, y_test)
            
            # Create comparison table
            comparison = []
            for res in results:
                comparison.append([
                    res['model'],
                    f"{res['accuracy']:.4f}",
                    f"{res['precision']:.4f}",
                    f"{res['recall']:.4f}",
                    f"{res['f1']:.4f}",
                    f"{res['time']:.2f}s"
                ])
            
            print("\n\nComparison of All Classifiers:")
            print(tabulate(
                comparison,
                headers=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Time'],
                tablefmt='grid'
            ))
            
            # Find best model by F1 score
            best_model = max(results, key=lambda x: x['f1'])
            print(f"\nBest model: {best_model['model']} with F1 score: {best_model['f1']:.4f}")
            
        else:
            # Run specific classifier
            classifier_params = {}
            
            if args.classifier == 'svm':
                classifier_params = {
                    'kernel': args.svm_kernel,
                    'C': args.svm_c
                }
            elif args.classifier == 'random_forest':
                classifier_params = {
                    'n_estimators': args.rf_n_estimators,
                    'max_depth': args.rf_max_depth
                }
            elif args.classifier == 'logistic_regression':
                classifier_params = {
                    'C': args.lr_c,
                    'penalty': args.lr_penalty
                }
            elif args.classifier == 'naive_bayes':
                classifier_params = {
                    'alpha': args.nb_alpha
                }
            elif args.classifier == 'knn':
                classifier_params = {
                    'n_neighbors': args.knn_n_neighbors
                }
            elif args.classifier == 'decision_tree':
                classifier_params = {
                    'max_depth': args.dt_max_depth
                }
            elif args.classifier == 'adaboost':
                classifier_params = {
                    'n_estimators': args.ab_n_estimators,
                    'learning_rate': args.ab_learning_rate
                }
            elif args.classifier == 'gradient_boosting':
                classifier_params = {
                    'n_estimators': args.gb_n_estimators,
                    'learning_rate': args.gb_learning_rate
                }
            
            print(f"Training {args.classifier.replace('_', ' ').title()}...")
            model = detector.train_classifier(args.classifier, X_train_vec, y_train, **classifier_params)
            metrics = detector.evaluate_model(model, X_test_vec, y_test)
            
            print("\nEvaluation Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print("\nClassification Report:")
            print(metrics['report'])
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == '__main__':
    main()

