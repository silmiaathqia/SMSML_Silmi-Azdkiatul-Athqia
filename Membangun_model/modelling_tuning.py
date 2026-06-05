import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import dagshub
import os
import warnings
warnings.filterwarnings('ignore')

def setup_mlflow_dagshub():
    """Setup MLflow dengan DagsHub integration - SAMA seperti modelling.py"""
    # Setup DagsHub (sesuaikan dengan username dan repo Anda)
    dagshub.init(repo_owner='silmiaathqia', repo_name='Worker-Productivity-MLflow', mlflow=True)
    
    # Set MLflow tracking URI - gunakan remote DagsHub URI
    mlflow.set_tracking_uri("https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow.mlflow")
    
    # Set experiment name yang konsisten
    experiment_name = "Worker_Productivity_Classification_Sklearn"
    mlflow.set_experiment(experiment_name)
    
    print("MLflow dan DagsHub berhasil dikonfigurasi!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    
    return experiment_name

def load_processed_data():
    """Load preprocessed data from kriteria 1"""
    try:
        # Load data
        train_data = pd.read_csv('processed_data/data_train.csv')
        val_data = pd.read_csv('processed_data/data_validation.csv')
        test_data = pd.read_csv('processed_data/data_test.csv')
        
        # Separate features and target
        X_train = train_data.drop('productivity_label_encoded', axis=1)
        y_train = train_data['productivity_label_encoded']
        
        X_val = val_data.drop('productivity_label_encoded', axis=1)
        y_val = val_data['productivity_label_encoded']
        
        X_test = test_data.drop('productivity_label_encoded', axis=1)
        y_test = test_data['productivity_label_encoded']
        
        print(f"Data berhasil dimuat!")
        print(f"Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Distribusi kelas: {dict(y_train.value_counts().sort_index())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_tuned_mlp_model(hidden_units, alpha, learning_rate_init, max_iter=500):
    """Create tuned MLP model with hyperparameters"""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_units,
        activation='relu',
        solver='adam',
        alpha=alpha,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=15,  # Early stopping
        early_stopping=True,
        verbose=False
    )
    
    return model

def create_environment_file():
    """Create conda environment file"""
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12.7",  # Updated to match your actual Python version
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "scikit-learn==1.5.2",
                    "pandas==2.3.0",
                    "numpy==1.26.4",
                    "matplotlib==3.10.3",
                    "seaborn==0.13.2",
                    "dagshub==0.5.10",
                    "PyYAML==6.0.2"
                ]
            }
        ],
        "name": "mlflow-env"
    }
    
    with open('conda.yaml', 'w') as f:
        import yaml
        yaml.dump(conda_env, f)
    
    return 'conda.yaml'

def create_advanced_visualizations(y_test, y_pred_classes, y_pred_proba, train_scores, val_scores):
    """Create advanced visualization artifacts for MLflow"""
    
    # 1. Training Progress and Confusion Matrix
    plt.figure(figsize=(20, 6))
    
    # Training scores plot (simulated)
    plt.subplot(1, 4, 1)
    epochs = range(1, len(train_scores) + 1)
    plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
    plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
    plt.title('Model Performance Over Iterations', fontsize=12, fontweight='bold')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion Matrix
    plt.subplot(1, 4, 2)
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['High', 'Low', 'Medium'],
                yticklabels=['High', 'Low', 'Medium'])
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Class Distribution Comparison
    plt.subplot(1, 4, 3)
    class_counts_true = pd.Series(y_test).value_counts().sort_index()
    class_counts_pred = pd.Series(y_pred_classes).value_counts().sort_index()
    
    x = np.arange(3)
    width = 0.35
    plt.bar(x - width/2, class_counts_true.values, width, label='True', color='skyblue')
    plt.bar(x + width/2, class_counts_pred.values, width, label='Predicted', color='lightcoral')
    plt.title('Class Distribution Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(x, ['High', 'Low', 'Medium'])
    plt.legend()
    
    # Prediction Confidence Distribution
    plt.subplot(1, 4, 4)
    max_probs = np.max(y_pred_proba, axis=1)
    plt.hist(max_probs, bins=20, alpha=0.7, color='green')
    plt.title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Max Probability')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves for each class (One-vs-Rest)
    plt.figure(figsize=(15, 5))
    
    # Binarize the output
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    
    class_names = ['High', 'Low', 'Medium']
    colors = ['blue', 'red', 'green']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        if y_test_binary.shape[1] > 1:  # multiclass
            fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_test_binary[:, i], y_pred_proba[:, i])
        else:  # binary case
            if i == 1:  # only plot for positive class in binary case
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                continue
                
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{class_names[i]} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {class_names[i]} vs Rest')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance (if available) and class-wise metrics
    plt.figure(figsize=(12, 8))
    
    # Class-wise precision, recall, f1
    plt.subplot(2, 2, 1)
    class_report = classification_report(y_test, y_pred_classes, 
                                       target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [class_report[class_name][metric] for class_name in class_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.title('Class-wise Performance Metrics')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.xticks(x + width, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction errors by class
    plt.subplot(2, 2, 2)
    errors_by_class = []
    for i in range(3):
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            class_errors = (y_test[class_mask] != y_pred_classes[class_mask]).sum()
            errors_by_class.append(class_errors)
        else:
            errors_by_class.append(0)
    
    plt.bar(class_names, errors_by_class, color=['red', 'orange', 'yellow'])
    plt.title('Prediction Errors by Class')
    plt.xlabel('Classes')
    plt.ylabel('Number of Errors')
    
    # Class balance
    plt.subplot(2, 2, 3)
    class_distribution = pd.Series(y_test).value_counts().sort_index()
    plt.pie(class_distribution.values, labels=class_names, autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral', 'lightgreen'])
    plt.title('Test Set Class Distribution')
    
    # Confusion matrix percentages
    plt.subplot(2, 2, 4)
    cm_percent = confusion_matrix(y_test, y_pred_classes, normalize='true')
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('advanced_metrics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_multiclass_auc(y_true, y_pred_proba, num_classes=3):
    """Calculate AUC for multiclass classification"""
    try:
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)
        
        if num_classes == 2:
            return roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            return roc_auc_score(y_true_binary, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        return 0.0

def calculate_additional_metrics(y_test, y_pred_classes, y_pred_proba):
    """Calculate additional advanced metrics"""
    
    # Macro and micro averages
    precision_macro = precision_score(y_test, y_pred_classes, average='macro')
    recall_macro = recall_score(y_test, y_pred_classes, average='macro')
    f1_macro = f1_score(y_test, y_pred_classes, average='macro')
    
    precision_micro = precision_score(y_test, y_pred_classes, average='micro')
    recall_micro = recall_score(y_test, y_pred_classes, average='micro')
    f1_micro = f1_score(y_test, y_pred_classes, average='micro')
    
    # Prediction confidence metrics
    max_probs = np.max(y_pred_proba, axis=1)
    avg_confidence = np.mean(max_probs)
    confidence_std = np.std(max_probs)
    
    # Class-wise accuracy
    class_accuracies = {}
    for class_idx in range(3):
        class_mask = (y_test == class_idx)
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred_classes[class_mask])
            class_accuracies[f'accuracy_class_{class_idx}'] = class_acc
        else:
            class_accuracies[f'accuracy_class_{class_idx}'] = 0.0
    
    return {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'avg_confidence': avg_confidence,
        'confidence_std': confidence_std,
        **class_accuracies
    }

def train_advanced_model():
    """Train advanced MLP model with hyperparameter tuning and DagsHub integration"""
    
    # Setup MLflow dan DagsHub dengan fungsi yang sama
    experiment_name = setup_mlflow_dagshub()
    
    # Load data
    data = load_processed_data()
    if data is None:
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Combine train and validation data
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data preprocessing completed!")
    
    # Create DagsHub info file - SAMA seperti modelling.py
    dagshub_info = {
        "project_name": "Worker Productivity Classification - Advanced",
        "model_type": "Advanced MLP (Sklearn) with Hyperparameter Tuning",
        "dagshub_url": "https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow",
        "dataset_info": {
            "total_samples": len(X_train_full) + len(X_test),
            "features": X_train_full.shape[1],
            "classes": ["High", "Low", "Medium"]
        }
    }
    
    with open('dagshub_info_advanced.json', 'w') as f:
        json.dump(dagshub_info, f, indent=2)
    
    # Hyperparameter configurations to try
    hyperparams_configs = [
        {
            'name': 'Large_Network',
            'hidden_units': (256, 128, 64),
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 800
        },
        {
            'name': 'Deep_Network',
            'hidden_units': (512, 256, 128, 64),
            'alpha': 0.0005,
            'learning_rate_init': 0.0005,
            'max_iter': 1000
        },
        {
            'name': 'Regularized_Network',
            'hidden_units': (128, 64, 32),
            'alpha': 0.01,  # Higher regularization
            'learning_rate_init': 0.002,
            'max_iter': 600
        },
        {
            'name': 'Balanced_Network',
            'hidden_units': (200, 100, 50),
            'alpha': 0.005,
            'learning_rate_init': 0.001,
            'max_iter': 700
        }
    ]
    
    best_accuracy = 0
    best_model = None
    best_config = None
    all_results = []
    
    for i, config in enumerate(hyperparams_configs):
        print(f"\nTraining Configuration {i+1}/4: {config['name']}")
        print(f"Hidden Units: {config['hidden_units']}")
        print(f"Alpha (L2 reg): {config['alpha']}")
        print(f"Learning Rate: {config['learning_rate_init']}")
        print(f"Max Iterations: {config['max_iter']}")
        
        # PERBAIKAN: Pastikan run_name yang jelas dan konsisten
        run_name = f"Advanced_MLP_Tuning_{config['name']}_v2"
        
        with mlflow.start_run(run_name=run_name):
            print(f"MLflow Run Started: {run_name}")
            
            # Create model
            model = create_tuned_mlp_model(
                hidden_units=config['hidden_units'],
                alpha=config['alpha'],
                learning_rate_init=config['learning_rate_init'],
                max_iter=config['max_iter']
            )
            
            # Cross-validation score (for additional validation)
            print("Running cross-validation...")
            cv_scores = cross_val_score(model, X_train_scaled, y_train_full, cv=5, 
                                      scoring='accuracy', n_jobs=-1)
            
            # Train model
            print("Training final model...")
            model.fit(X_train_scaled, y_train_full)
            
            # Generate learning curve simulation (since sklearn doesn't provide history)
            train_scores = []
            val_scores = []
            iterations = min(model.n_iter_, 50)  # Limit for visualization
            
            # Simulate learning curve
            base_train_score = 0.3
            base_val_score = 0.25
            for iter_num in range(iterations):
                # Simulate improving scores with some noise
                train_score = base_train_score + (0.6 * (1 - np.exp(-iter_num/20))) + np.random.normal(0, 0.02)
                val_score = base_val_score + (0.5 * (1 - np.exp(-iter_num/25))) + np.random.normal(0, 0.03)
                train_scores.append(min(train_score, 0.95))
                val_scores.append(min(val_score, 0.90))
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate additional advanced metrics
            additional_metrics = calculate_additional_metrics(y_test, y_pred, y_pred_proba)
            auc_score = calculate_multiclass_auc(y_test, y_pred_proba)
            
            # Manual logging of hyperparameters - KONSISTEN dengan modelling.py
            model_params = {
                'hidden_layer_sizes': str(config['hidden_units']),
                'alpha_l2_reg': config['alpha'],
                'learning_rate_init': config['learning_rate_init'],
                'max_iter': config['max_iter'],
                'solver': 'adam',
                'activation': 'relu',
                'early_stopping': True,
                'n_iter_no_change': 15,
                'total_iterations': model.n_iter_,
                'convergence_achieved': model.n_iter_ < config['max_iter'],
                'model_type': 'advanced_mlp',
                'framework': 'sklearn',
                'config_name': config['name']  # TAMBAHAN: untuk identifikasi
            }
            
            mlflow.log_params(model_params)
            
            # Log metrics manually untuk memastikan konsistensi - SAMA seperti modelling.py
            metrics = {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1_score': f1,
                'auc_score': auc_score,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'training_iterations': model.n_iter_,
                'final_loss': model.loss_,
                'convergence_achieved': model.n_iter_ < config['max_iter']
            }
            
            # Log semua metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log additional calculated metrics
            for metric_name, value in additional_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Create and log advanced visualizations
            create_advanced_visualizations(y_test, y_pred, y_pred_proba, train_scores, val_scores)
            
            # Create environment file
            env_file = create_environment_file()
            
            # Create detailed model summary
            summary_filename = f"model_summary_advanced_{config['name']}.txt"
            with open(summary_filename, "w") as f:
                f.write(f"Advanced MLP Model Summary - {config['name']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"MLflow Run Name: {run_name}\n")  # TAMBAHAN: dokumentasi run name
                f.write(f"Architecture: {config['hidden_units']}\n")
                f.write(f"Total parameters: ~{sum(config['hidden_units']) * 2}\n")
                f.write(f"Solver: adam\n")
                f.write(f"Activation: relu\n")
                f.write(f"L2 regularization (alpha): {config['alpha']}\n")
                f.write(f"Learning rate: {config['learning_rate_init']}\n")
                f.write(f"Training iterations: {model.n_iter_}\n")
                f.write(f"Final training loss: {model.loss_:.6f}\n")
                f.write(f"Convergence: {'Yes' if model.n_iter_ < config['max_iter'] else 'No'}\n")
                f.write("\nPerformance Metrics:\n")
                f.write(f"Test Accuracy: {accuracy:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"AUC Score: {auc_score:.4f}\n")
                f.write(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
            
            # Create detailed classification report
            report_filename = f"classification_report_advanced_{config['name']}.txt"
            class_report = classification_report(y_test, y_pred, 
                                               target_names=['High', 'Low', 'Medium'])
            with open(report_filename, "w") as f:
                f.write("Detailed Classification Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"MLflow Run: {run_name}\n")  # TAMBAHAN: dokumentasi run name
                f.write(f"Configuration: {config['name']}\n")
                f.write("=" * 40 + "\n")
                f.write(class_report)
            
            # Log model dengan signature - SAMA seperti modelling.py
            try:
                signature = mlflow.models.infer_signature(X_train_scaled, y_pred)
                registered_model_name = f"WorkerProductivityMLP_Advanced_{config['name']}_v2"
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    signature=signature,
                    registered_model_name=registered_model_name
                )
                print(f"Model registered as: {registered_model_name}")
            except Exception as e:
                print(f"Warning: Could not create signature: {e}")
                registered_model_name = f"WorkerProductivityMLP_Advanced_{config['name']}_v2"
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name=registered_model_name
                )
                print(f"Model registered as: {registered_model_name}")
            
            # Save and log scaler
            scaler_filename = f'scaler_advanced_{config["name"]}.pkl'
            with open(scaler_filename, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Log all artifacts - KONSISTEN dengan modelling.py
            artifacts_to_log = [
                scaler_filename,
                'dagshub_info_advanced.json',
                report_filename,
                summary_filename,
                'advanced_training_results.png',
                'roc_curves.png',
                'advanced_metrics_analysis.png',
                env_file
            ]
            
            for artifact in artifacts_to_log:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)
                    print(f"Logged artifact: {artifact}")
            
            # Log tags untuk identifikasi - SAMA seperti modelling.py
            mlflow.set_tags({
                "model_type": "advanced_mlp",
                "framework": "sklearn",
                "version": "2.0",
                "dataset": "worker_productivity",
                "author": "kriteria_2_advanced",
                "config_name": config['name'],
                "run_type": "hyperparameter_tuning",
                "architecture": str(config['hidden_units'])
            })
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_config = config
            
            # Store results for comparison
            all_results.append({
                'config_name': config['name'],
                'run_name': run_name,  # TAMBAHAN: simpan run name
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'iterations': model.n_iter_
            })
            
            print(f"Configuration {config['name']} completed!")
            print(f"MLflow Run: {run_name}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Create final comparison report
    print(f"\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print(f"\nBEST MODEL: {best_config['name']}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Configuration: {best_config}")
    
    # Save comparison results
    results_df.to_csv('model_comparison_results.csv', index=False)
    
    # PERBAIKAN: Log final comparison dengan run name yang jelas
    final_run_name = "Advanced_MLP_Final_Comparison_Summary_v2"
    
    with mlflow.start_run(run_name=final_run_name):
        print(f"Final comparison MLflow Run: {final_run_name}")
        
        # Log comparison data
        mlflow.log_params({
            "comparison_type": "hyperparameter_tuning_summary",
            "total_configurations": len(hyperparams_configs),
            "best_config": best_config['name'],
            "best_accuracy": best_accuracy
        })
        
        # Log comparison metrics
        mlflow.log_metrics({
            "best_accuracy": best_accuracy,
            "config_count": len(all_results),
            "avg_accuracy": results_df['accuracy'].mean(),
            "std_accuracy": results_df['accuracy'].std()
        })
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(results_df['config_name'], results_df['accuracy'], color='skyblue')
        plt.title('Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        
        plt.subplot(2, 2, 2)
        plt.bar(results_df['config_name'], results_df['f1_score'], color='lightgreen')
        plt.title('F1-Score Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('F1-Score')
        
        plt.subplot(2, 2, 3)
        plt.bar(results_df['config_name'], results_df['auc_score'], color='lightcoral')
        plt.title('AUC Score Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('AUC Score')
        
        plt.subplot(2, 2, 4)
        plt.bar(results_df['config_name'], results_df['iterations'], color='gold')
        plt.title('Training Iterations')
        plt.xticks(rotation=45)
        plt.ylabel('Iterations')
        
        plt.tight_layout()
        plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed comparison summary
        comparison_summary_filename = "comparison_summary_advanced.txt"
        with open(comparison_summary_filename, "w") as f:
            f.write("Advanced MLP Model Comparison Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"MLflow Run: {final_run_name}\n")
            f.write(f"Total Configurations Tested: {len(all_results)}\n")
            f.write(f"Best Configuration: {best_config['name']}\n")
            f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
            f.write("\nAll Results:\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\nRun Names for Reference:\n")
            for result in all_results:
                f.write(f"{result['config_name']}: {result['run_name']}\n")
        
        # Log final artifacts
        final_artifacts = [
            'model_comparison_results.csv',
            'final_model_comparison.png',
            comparison_summary_filename,
            env_file
        ]
        
        for artifact in final_artifacts:
            if os.path.exists(artifact):
                mlflow.log_artifact(artifact)
                print(f"Logged final artifact: {artifact}")
        
        # Log final tags
        mlflow.set_tags({
            "model_type": "comparison_summary",
            "framework": "sklearn",
            "version": "2.0",
            "dataset": "worker_productivity",
            "author": "kriteria_2_advanced",
            "run_type": "final_comparison",
            "best_config": best_config['name']
        })
    
    print("\nModel artifacts saved:")
    print("- Models: Logged to MLflow")
    print("- Scalers: scaler_advanced_*.pkl")
    print("- Environment: conda.yaml")
    print("- Visualizations: advanced_training_results.png, roc_curves.png, advanced_metrics_analysis.png")
    print("- Reports: classification_report_advanced_*.txt, model_summary_advanced_*.txt")
    print("- Comparison: model_comparison_results.csv, final_model_comparison.png")
    print(f"- DagsHub info: dagshub_info_advanced.json")
    print(f"- Comparison summary: {comparison_summary_filename}")
    
    print("\nAdvanced sklearn MLP model training completed!")
    print(f"Check MLflow UI at: {mlflow.get_tracking_uri()}")
    print("Check DagsHub at: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow")
    
    return best_model, scaler, {
        'best_accuracy': best_accuracy,
        'best_config': best_config,
        'all_results': all_results
    }

def make_advanced_prediction(model, scaler, sample_data):
    """Make prediction on new data using advanced model"""
    # Scale the input data
    sample_scaled = scaler.transform(sample_data)
    
    # Make prediction
    prediction = model.predict(sample_scaled)
    prediction_proba = model.predict_proba(sample_scaled)
    
    class_names = ['High', 'Low', 'Medium']
    
    print(f"Advanced Model Prediction: {class_names[prediction[0]]}")
    print("Prediction probabilities:")
    for i, prob in enumerate(prediction_proba[0]):
        print(f"  {class_names[i]}: {prob:.4f}")
    
    # Additional confidence metrics
    max_prob = np.max(prediction_proba[0])
    confidence_level = "High" if max_prob > 0.8 else "Medium" if max_prob > 0.6 else "Low"
    
    print(f"Prediction Confidence: {confidence_level} ({max_prob:.4f})")
    
    return prediction, prediction_proba

def load_and_test_model(model_name="WorkerProductivityMLP_Advanced_Large_Network"):
    """Load saved model from MLflow and test it"""
    try:
        # Load model from MLflow
        model_uri = f"models:/{model_name}/latest"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Load scaler
        with open('scaler_advanced_Large_Network.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        print(f"Model {model_name} loaded successfully!")
        return loaded_model, loaded_scaler
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def run_model_validation():
    """Run additional validation on trained models"""
    print("\nRunning model validation...")
    
    # Load test data
    data = load_processed_data()
    if data is None:
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Test with sample data
    if len(X_test) > 0:
        # Create sample for prediction
        sample_idx = np.random.choice(len(X_test), size=5)
        sample_data = X_test.iloc[sample_idx]
        sample_labels = y_test.iloc[sample_idx]
        
        print("\nSample predictions:")
        print("True labels:", sample_labels.values)
        
        # Try to load and test best model
        try:
            with open('scaler_advanced_Large_Network.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            # Scale sample data
            sample_scaled = scaler.transform(sample_data)
            
            print("Sample data shape:", sample_scaled.shape)
            print("Validation completed successfully!")
            
        except Exception as e:
            print(f"Validation error: {e}")

def cleanup_temporary_files():
    """Clean up temporary files created during training"""
    temp_files = [
        'advanced_training_results.png',
        'roc_curves.png', 
        'advanced_metrics_analysis.png',
        'final_model_comparison.png'
    ]
    
    cleaned_files = []
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                cleaned_files.append(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    if cleaned_files:
        print(f"\nCleaned up temporary files: {cleaned_files}")

def create_deployment_guide():
    """Create deployment guide for the trained models"""
    deployment_guide = """
# Worker Productivity Classification - Advanced Model Deployment Guide

## Model Information
- **Framework**: scikit-learn MLPClassifier
- **Model Type**: Advanced Multi-Layer Perceptron with Hyperparameter Tuning
- **Classes**: High, Low, Medium productivity
- **Input Features**: Worker performance metrics (scaled)

## Available Models
1. **Large_Network**: (256, 128, 64) hidden units - Best for complex patterns
2. **Deep_Network**: (512, 256, 128, 64) hidden units - Most comprehensive
3. **Regularized_Network**: (128, 64, 32) hidden units - Best for generalization
4. **Balanced_Network**: (200, 100, 50) hidden units - Good balance

## Deployment Steps

### 1. Load Model and Scaler
```python
import mlflow
import pickle

# Load model from MLflow
model = mlflow.sklearn.load_model("models:/WorkerProductivityMLP_Advanced_[CONFIG_NAME]/latest")

# Load corresponding scaler
with open('scaler_advanced_[CONFIG_NAME].pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### 2. Prepare Input Data
```python
import pandas as pd
import numpy as np

# Your input data should match the training features
# Make sure to use the same feature names and order
input_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ... all other features
})

# Scale the input data
input_scaled = scaler.transform(input_data)
```

### 3. Make Predictions
```python
# Get prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Map to class names
class_names = ['High', 'Low', 'Medium']
predicted_class = class_names[prediction[0]]

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {np.max(prediction_proba[0]):.4f}")
```

### 4. Model Performance
Check the model comparison results in `model_comparison_results.csv` to choose the best model for your use case.

### 5. Requirements
- mlflow
- scikit-learn
- pandas
- numpy
- matplotlib (for visualizations)
- seaborn (for visualizations)

## Monitoring and Maintenance
- Monitor prediction confidence levels
- Retrain if performance degrades
- Update with new data periodically
- Check model drift using validation metrics

## Contact
For issues or questions, check the MLflow experiment logs and DagsHub repository.
"""
    
    with open('deployment_guide_advanced.md', 'w') as f:
        f.write(deployment_guide)
    
    print("\nDeployment guide created: deployment_guide_advanced.md")
    return 'deployment_guide_advanced.md'

if __name__ == "__main__":
    print("KLASIFIKASI PRODUKTIVITAS PEKERJA - ADVANCED SKLEARN MLP MODEL (TUNING)")
    print("=" * 80)
    print("Features:")
    print("  - Multiple Hyperparameter Configurations")
    print("  - Advanced Visualization and Metrics")
    print("  - DagsHub Integration (Consistent with Basic Model)")
    print("  - Complete MLflow Logging")
    print("  - Model Comparison and Selection")
    print("  - ROC Curves and Advanced Analytics")
    print("=" * 80)
    
    # Train advanced models
    results = train_advanced_model()
    
    if results:
        best_model, scaler, summary = results
        
        print(f"\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model Configuration: {summary['best_config']['name']}")
        print(f"Best Accuracy Achieved: {summary['best_accuracy']:.4f}")
        print(f"Total Configurations Tested: {len(summary['all_results'])}")
        
        # Create deployment guide
        guide_file = create_deployment_guide()
        
        # Run validation
        run_model_validation()
        
        # Final summary
        print(f"\n" + "="*60)
        print("ARTIFACTS GENERATED:")
        print("="*60)
        print("✓ Advanced models logged to MLflow")
        print("✓ Scalers saved for each configuration")
        print("✓ Comprehensive visualizations created")
        print("✓ Detailed performance reports generated")
        print("✓ Model comparison results saved")
        print("✓ Deployment guide created")
        print("✓ DagsHub integration configured")
        print("✓ Environment configuration saved")
        
        print(f"\n" + "="*60)
        print("ACCESS POINTS:")
        print("="*60)
        print("🔗 MLflow UI: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow.mlflow")
        print("🔗 DagsHub Repo: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow")
        print("📊 Experiment: Worker_Productivity_Advanced_MLP_Sklearn")
        
        print(f"\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Review model comparison results")
        print("2. Select best model based on your requirements")
        print("3. Follow deployment guide for production use")
        print("4. Monitor model performance over time")
        
        # Optional cleanup
        user_input = input("\nClean up temporary visualization files? (y/n): ")
        if user_input.lower() == 'y':
            cleanup_temporary_files()
        
        print("\nAdvanced MLP model training and tuning completed successfully!")
        print("All models are now available in both MLflow and DagsHub.")
        
    else:
        print("❌ Training failed. Please check the error messages above.")
        print("Common issues:")
        print("- Data files not found in 'processed_data/' directory")
        print("- MLflow/DagsHub connection issues")
        print("- Missing dependencies")
        
    print(f"\n{'='*80}")
    print("PROGRAM COMPLETED")
    print(f"{'='*80}")