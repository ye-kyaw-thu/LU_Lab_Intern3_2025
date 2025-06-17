# Written by Ye Kyaw Thu, LU Lab., Myanmar
# Humor binary classification with ML approaches
# Version 1.0
# Last updated: 17 June 2025
#
# Major Updates:
# - Multilingual support (removed English-only limitation)
# - Model saving/loading functionality
# - Test-only mode with model file specification
#
# How to run:
# $ python ./ml_humor_detection.py --help
#
# Training and testing:
# $ python ml_humor_detection.py --train_file train.csv --test_file test.csv
#
# Test-only mode:
# $ python ml_humor_detection.py --test --model_file svm_model.joblib --test_file test.csv

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
import joblib
import os
from pathlib import Path

class HumorDetector:
    def __init__(self):
        # Multilingual TF-IDF vectorizer (no English-specific preprocessing)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=False,  # Disable automatic lowercase for multilingual support
            token_pattern=r'\b\w+\b',  # Basic token pattern for multiple languages
            analyzer='word'
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

    def load_data(self, train_file=None, test_file=None):
        """Load and preprocess the data with UTF-8 encoding"""
        if train_file:
            train_df = pd.read_csv(train_file, header=None, names=['text', 'label'], encoding='utf-8')
        if test_file:
            test_df = pd.read_csv(test_file, header=None, names=['text', 'label'], encoding='utf-8')
    
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
    
        if train_file:
            train_df['label'] = train_df['label'].apply(convert_label)
            X_train = train_df['text'].values
            y_train = train_df['label'].values
        else:
            X_train, y_train = None, None
            
        if test_file:
            test_df['label'] = test_df['label'].apply(convert_label)
            X_test = test_df['text'].values
            y_test = test_df['label'].values
        else:
            X_test, y_test = None, None
    
        # Debug print to verify labels
        if y_train is not None:
            print("Unique labels in training set:", set(y_train))
        if y_test is not None:
            print("Unique labels in test set:", set(y_test))
    
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, X_train=None, X_test=None):
        """Vectorize the text data"""
        if X_train is not None:
            X_train_vec = self.vectorizer.fit_transform(X_train)
        else:
            X_train_vec = None
        
        if X_test is not None:
            X_test_vec = self.vectorizer.transform(X_test)
        else:
            X_test_vec = None
        
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

    def save_model(self, model, model_file):
        """Save trained model to file"""
        # Create directory if it doesn't exist
        Path(os.path.dirname(model_file)).mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': model,
            'vectorizer': self.vectorizer
        }, model_file)
        print(f"Model saved to {model_file}")

    def load_model(self, model_file):
        """Load trained model from file"""
        data = joblib.load(model_file)
        return data  # Return the entire dictionary

    def run_all_classifiers(self, X_train, y_train, X_test, y_test, save_dir='models'):
        """Run all classifiers and return results"""
        results = []
        
        for name in self.classifiers:
            start_time = time.time()
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            model = self.train_classifier(name, X_train, y_train)
            
            # Save each model
            model_file = os.path.join(save_dir, f"{name}_model.joblib")
            self.save_model(model, model_file)
            
            metrics = self.evaluate_model(model, X_test, y_test)
            
            elapsed = time.time() - start_time
            metrics['time'] = elapsed
            metrics['model'] = name.replace('_', ' ').title()
            metrics['model_file'] = model_file
            
            results.append(metrics)
            
            print(f"Completed in {elapsed:.2f} seconds")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Run ensemble classifiers
        for name in self.ensemble_classifiers:
            start_time = time.time()
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            model = self.train_classifier(name, X_train, y_train)
            
            # Save ensemble model
            model_file = os.path.join(save_dir, f"{name}_model.joblib")
            self.save_model(model, model_file)
            
            metrics = self.evaluate_model(model, X_test, y_test)
            
            elapsed = time.time() - start_time
            metrics['time'] = elapsed
            metrics['model'] = name.replace('_', ' ').title()
            metrics['model_file'] = model_file
            
            results.append(metrics)
            
            print(f"Completed in {elapsed:.2f} seconds")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Humor Detection using Traditional ML Techniques (Version 1.0)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_file', type=str, help='Path to training CSV file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to testing CSV file')
    
    # Mode selection
    parser.add_argument('--test', action='store_true', help='Run in test-only mode')
    parser.add_argument('--model_file', type=str, help='Path to model file for test-only mode')
    
    # Classifier selection
    parser.add_argument('--classifier', type=str, default='all',
                      choices=['all', 'svm', 'random_forest', 'logistic_regression', 
                              'naive_bayes', 'knn', 'decision_tree', 'adaboost',
                              'gradient_boosting', 'voting'],
                      help='Classifier to use (ignored in test-only mode)')
    
    # Model saving
    parser.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save trained models')
    
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
    
    # Validate arguments
    if not args.test and not args.train_file:
        parser.error("--train_file is required unless in test-only mode (--test)")
    if args.test and not args.model_file:
        parser.error("--model_file is required in test-only mode")
    
    # Initialize the humor detector
    detector = HumorDetector()
    
    try:
        if args.test:
            # Test-only mode
            print(f"Loading model from {args.model_file}...")
            model_data = detector.load_model(args.model_file)
            model = model_data['model']
            detector.vectorizer = model_data['vectorizer']  # Set the loaded vectorizer
    
            # Load test data
            _, _, X_test, y_test = detector.load_data(test_file=args.test_file)
            X_test_vec = detector.vectorizer.transform(X_test)  # Transform test data
    
            # Evaluate
            metrics = detector.evaluate_model(model, X_test_vec, y_test)
    
            print("\nEvaluation Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print("\nClassification Report:")
            print(metrics['report'])            
        else:
            # Training + testing mode
            X_train, y_train, X_test, y_test = detector.load_data(args.train_file, args.test_file)
            X_train_vec, X_test_vec = detector.preprocess_data(X_train, X_test)
            
            if args.classifier == 'all':
                # Run all classifiers
                print("Running all classifiers...")
                results = detector.run_all_classifiers(X_train_vec, y_train, X_test_vec, y_test, args.save_dir)
                
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
                print(f"Model saved to: {best_model['model_file']}")
                
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
                
                # Save the trained model
                model_file = os.path.join(args.save_dir, f"{args.classifier}_model.joblib")
                detector.save_model(model, model_file)
                
                metrics = detector.evaluate_model(model, X_test_vec, y_test)
                
                print("\nEvaluation Results:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print("\nClassification Report:")
                print(metrics['report'])
                print(f"Model saved to: {model_file}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == '__main__':
    main()

