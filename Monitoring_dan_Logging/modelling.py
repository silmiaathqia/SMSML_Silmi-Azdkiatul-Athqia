import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import dagshub
import os
import warnings
import argparse
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

def setup_mlflow_dagshub():
    """Setup MLflow dengan DagsHub integration untuk CI/CD"""
    try:
        # Setup DagsHub dengan environment variables (untuk CI/CD)
        dagshub.init(repo_owner='silmiaathqia', repo_name='Worker-Productivity-MLflow', mlflow=True)
        
        # Set MLflow tracking URI - prioritas environment variable untuk CI/CD
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', "https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow.mlflow")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment name - gunakan try/except untuk handling experiment
        experiment_name = "Worker_Productivity_Classification_Sklearn"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(experiment_name)
                print(f"Using existing experiment: {experiment_name}")
        except Exception as exp_error:
            print(f"Warning: Experiment setup error: {exp_error}")
            # Fallback ke default experiment
            experiment_name = "Default"
            mlflow.set_experiment(experiment_name)
        
        print("MLflow dan DagsHub berhasil dikonfigurasi!")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {experiment_name}")
        
        return experiment_name
        
    except Exception as e:
        print(f"Warning: MLflow setup error: {e}")
        print("Akan menggunakan local tracking...")
        mlflow.set_tracking_uri("file:./mlruns")
        experiment_name = "Worker_Productivity_Local"
        try:
            mlflow.set_experiment(experiment_name)
        except:
            experiment_name = "Default"
            mlflow.set_experiment(experiment_name)
        return experiment_name

def load_processed_data(data_path="processed_data"):
    """Load preprocessed data dengan path yang fleksibel"""
    try:
        # Cek apakah path relatif atau absolut
        data_dir = Path(data_path)
        if not data_dir.exists():
            # Coba path alternatif
            alt_paths = [
                Path("processed_data"),
                Path("MLProject/processed_data"),
                Path("../processed_data"),
                Path("./data")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    data_dir = alt_path
                    break
            else:
                raise FileNotFoundError(f"Processed data directory not found. Tried: {[str(p) for p in [data_dir] + alt_paths]}")
        
        print(f"Loading data from: {data_dir.absolute()}")
        
        # Load data files
        train_file = data_dir / 'data_train.csv'
        val_file = data_dir / 'data_validation.csv'
        test_file = data_dir / 'data_test.csv'
        
        # Validate files exist
        for file_path in [train_file, val_file, test_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load data
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)
        
        # Separate features and target
        target_col = 'productivity_label_encoded'
        
        # Validate target column exists
        if target_col not in train_data.columns:
            print(f"Available columns: {list(train_data.columns)}")
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X_train = train_data.drop(target_col, axis=1)
        y_train = train_data[target_col]
        
        X_val = val_data.drop(target_col, axis=1)
        y_val = val_data[target_col]
        
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]
        
        print(f"Data berhasil dimuat!")
        print(f"Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Features: {list(X_train.columns)}")
        print(f"Kelas target: {sorted(y_train.unique())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return None

def create_mlp_model():
    """Create MLP model using sklearn for worker productivity classification"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,  # Early stopping
        early_stopping=True,
        verbose=False  # Set to False untuk CI/CD
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names=['High', 'Low', 'Medium'], output_dir="."):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Basic Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'confusion_matrix_basic.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {e}")
        return None

def plot_metrics_comparison(metrics_dict, output_dir="."):
    """Plot metrics comparison"""
    try:
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Metrics - Basic Model')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'metrics_comparison_basic.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    except Exception as e:
        print(f"Warning: Could not create metrics plot: {e}")
        return None

def create_environment_file(output_dir="."):
    """Create conda environment file"""
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12.7",
            "pip<=25.1",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle==3.1.1",
                    "numpy==1.26.4",
                    "pandas==2.3.0",
                    "psutil==7.0.0",
                    "scikit-learn==1.5.2",
                    "scipy==1.15.3",
                    "matplotlib==3.10.3",
                    "seaborn==0.13.2",
                    "dagshub==0.5.10",
                    "PyYAML==6.0.2"
                ]
            }
        ],
        "name": "mlflow-env"
    }
    
    env_path = Path(output_dir) / 'conda.yaml'
    try:
        import yaml
        with open(env_path, 'w') as f:
            yaml.dump(conda_env, f)
        return str(env_path)
    except ImportError:
        # Fallback jika PyYAML tidak ada
        import json
        json_path = Path(output_dir) / 'environment.json'
        with open(json_path, 'w') as f:
            json.dump(conda_env, f, indent=2)
        return str(json_path)

def cleanup_temporary_files(output_dir="."):
    """Clean up temporary files"""
    temp_files = [
        'confusion_matrix_basic.png',
        'metrics_comparison_basic.png'
    ]
    
    cleaned_files = []
    for file in temp_files:
        file_path = Path(output_dir) / file
        if file_path.exists():
            try:
                file_path.unlink()
                cleaned_files.append(str(file_path))
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
    
    if cleaned_files:
        print(f"\nCleaned up temporary files: {cleaned_files}")
    else:
        print("\nNo temporary files to clean up.")

def create_deployment_info(output_dir="."):
    """Create deployment information file"""
    deployment_info = {
        "model_name": "Basic MLP Worker Productivity Classifier",
        "model_type": "sklearn.neural_network.MLPClassifier",
        "version": "1.0",
        "created_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "architecture": "(128, 64, 32)",
            "activation": "relu",
            "solver": "adam",
            "regularization": "L2 (alpha=0.001)",
            "early_stopping": True
        },
        "deployment": {
            "required_files": [
                "model (from MLflow)",
                "scaler_basic.pkl",
                "conda.yaml"
            ],
            "mlflow_model_name": "WorkerProductivityMLP_Basic",
            "prediction_classes": ["High", "Low", "Medium"]
        },
        "ci_cd": {
            "mlflow_project": True,
            "docker_ready": True,
            "github_actions": True
        },
        "performance_note": "Check MLflow experiment for detailed metrics"
    }
    
    info_path = Path(output_dir) / 'deployment_info_basic.json'
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return str(info_path)

def train_basic_model(data_path="processed_data", experiment_name=None):
    """Train basic MLP model dengan MLflow Project compatibility"""
    
    print("="*60)
    print("STARTING MLflow PROJECT TRAINING")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Data path: {data_path}")
    
    # Setup MLflow dan DagsHub
    if experiment_name:
        try:
            mlflow.set_experiment(experiment_name)
            exp_name = experiment_name
        except Exception as e:
            print(f"Warning: Could not set experiment {experiment_name}: {e}")
            exp_name = setup_mlflow_dagshub()
    else:
        exp_name = setup_mlflow_dagshub()
    
    # Load data
    data = load_processed_data(data_path)
    if data is None:
        print("❌ Failed to load data. Exiting...")
        return None
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Combine train and validation for sklearn MLPClassifier
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data scaling completed!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Create output directory
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    
    # Create DagsHub info file
    dagshub_info = {
        "project_name": "Worker Productivity Classification",
        "model_type": "Basic MLP (Sklearn)",
        "dagshub_url": "https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow",
        "mlflow_project": True,
        "ci_cd_ready": True,
        "dataset_info": {
            "total_samples": len(X_train_full) + len(X_test),
            "features": X_train_full.shape[1],
            "classes": ["High", "Low", "Medium"]
        }
    }
    
    with open('dagshub_info.json', 'w') as f:
        json.dump(dagshub_info, f, indent=2)

    # Run experiment dengan manual logging dan error handling
    try:
        with mlflow.start_run(run_name="MLProject_Basic_MLP"):
            print("\n" + "="*60)
            print("STARTING MLflow RUN")
            print("="*60)
            
            # DISABLE autolog untuk manual control
            mlflow.sklearn.autolog(disable=True)
            
            # Log parameters manually
            model_params = {
                'hidden_layer_sizes': str((128, 64, 32)),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'shuffle': True,
                'random_state': 42,
                'tol': 1e-4,
                'validation_fraction': 0.1,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-8,
                'n_iter_no_change': 10,
                'early_stopping': True,
                'batch_size': 'auto',
                'verbose': False,
                'model_type': 'basic_mlp',
                'framework': 'sklearn',
                'experiment_group': 'mlproject',
                'data_path': str(data_path),
                'ci_cd_mode': True
            }
            
            # Log all parameters
            for param_name, param_value in model_params.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception as param_error:
                    print(f"Warning: Could not log parameter {param_name}: {param_error}")
            
            # Create and train model
            model = create_mlp_model()
            
            print("\nModel Configuration:")
            print(f"Hidden layers: {model.hidden_layer_sizes}")
            print(f"Solver: {model.solver}")
            print(f"Alpha (L2 reg): {model.alpha}")
            
            # Train model
            print("\nStarting training...")
            model.fit(X_train_scaled, y_train_full)
            
            print(f"Training completed!")
            print(f"Iterations: {model.n_iter_}")
            print(f"Final loss: {model.loss_:.6f}")
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate per-class metrics
            precision_per_class = precision_score(y_test, y_pred, average=None)
            recall_per_class = recall_score(y_test, y_pred, average=None)
            f1_per_class = f1_score(y_test, y_pred, average=None)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            class_names = ['High', 'Low', 'Medium']
            
            # Log all metrics manually
            metrics = {
                'test_accuracy': accuracy,
                'test_precision_weighted': precision,
                'test_recall_weighted': recall,
                'test_f1_score_weighted': f1,
                'training_loss': model.loss_,
                'n_iterations': model.n_iter_,
                'convergence_achieved': model.n_iter_ < model.max_iter,
                # Per-class metrics
                'test_precision_high': precision_per_class[0],
                'test_precision_low': precision_per_class[1],
                'test_precision_medium': precision_per_class[2],
                'test_recall_high': recall_per_class[0],
                'test_recall_low': recall_per_class[1],
                'test_recall_medium': recall_per_class[2],
                'test_f1_high': f1_per_class[0],
                'test_f1_low': f1_per_class[1],
                'test_f1_medium': f1_per_class[2],
                # Dataset info
                'train_samples': len(X_train_full),
                'test_samples': len(X_test),
                'n_features': X_train_full.shape[1],
                'n_classes': len(class_names)
            }
            
            # Log metrics with error handling
            for metric_name, metric_value in metrics.items():
                try:
                    mlflow.log_metric(metric_name, metric_value)
                except Exception as metric_error:
                    print(f"Warning: Could not log metric {metric_name}: {metric_error}")
            
            print(f"\nTest Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Classification report
            class_report = classification_report(y_test, y_pred, target_names=class_names)
            print("\nClassification Report:")
            print(class_report)
            
            # Save classification report
            with open('classification_report_basic.txt', 'w') as f:
                f.write("MLflow Project - Basic MLP Model\n")
                f.write("=" * 35 + "\n")
                f.write(class_report)
            
            # Create visualizations
            cm_file = plot_confusion_matrix(y_test, y_pred, class_names)
            metrics_file = plot_metrics_comparison({
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            # Create environment file
            env_file = create_environment_file()
            
            # Create deployment info
            deployment_file = create_deployment_info()
            
            # Create model summary
            with open('model_summary_basic.txt', 'w') as f:
                f.write("MLflow Project - Basic MLP Model Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Architecture: {model.hidden_layer_sizes}\n")
                f.write(f"Solver: {model.solver}\n")
                f.write(f"Training iterations: {model.n_iter_}\n")
                f.write(f"Final loss: {model.loss_:.6f}\n")
                f.write(f"Convergence: {'Yes' if model.n_iter_ < model.max_iter else 'No'}\n")
                f.write(f"CI/CD Mode: True\n")
                f.write(f"MLflow Project: True\n")
                f.write("\nPerformance:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
            
            # Log model dengan signature dan error handling
            try:
                signature = mlflow.models.infer_signature(X_train_scaled, y_pred)
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    signature=signature,
                    registered_model_name="WorkerProductivityMLP_Basic"
                )
                print("✅ Model logged with signature")
            except Exception as model_error:
                print(f"Warning: Model logging with signature failed: {model_error}")
                try:
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        registered_model_name="WorkerProductivityMLP_Basic"
                    )
                    print("✅ Model logged without signature")
                except Exception as model_error2:
                    print(f"Warning: Model logging failed: {model_error2}")
                    # Log model without registration
                    mlflow.sklearn.log_model(model, "model")
                    print("✅ Model logged without registration")
            
            # Save and log scaler
            with open('scaler_basic.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Log all artifacts
            artifacts_to_log = [
                'scaler_basic.pkl',
                'dagshub_info.json',
                'classification_report_basic.txt',
                'model_summary_basic.txt',
                deployment_file
            ]
            
            # Add visualization files if they exist
            if cm_file and Path(cm_file).exists():
                artifacts_to_log.append(cm_file)
            if metrics_file and Path(metrics_file).exists():
                artifacts_to_log.append(metrics_file)
            if env_file and Path(env_file).exists():
                artifacts_to_log.append(env_file)
            
            for artifact in artifacts_to_log:
                if os.path.exists(artifact):
                    try:
                        mlflow.log_artifact(artifact)
                        print(f"✅ Logged: {artifact}")
                    except Exception as artifact_error:
                        print(f"Warning: Could not log artifact {artifact}: {artifact_error}")
            
            # Set tags
            try:
                mlflow.set_tags({
                    "model_type": "basic_mlp",
                    "framework": "sklearn",
                    "version": "1.0",
                    "dataset": "worker_productivity",
                    "author": "mlflow_project",
                    "experiment_group": "ci_cd",
                    "logging_type": "manual",
                    "mlflow_project": "true",
                    "docker_ready": "true"
                })
            except Exception as tag_error:
                print(f"Warning: Could not set tags: {tag_error}")
            
            print("\n" + "="*60)
            print("✅ MLflow PROJECT TRAINING COMPLETED!")
            print("="*60)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Model: WorkerProductivityMLP_Basic")
            print(f"Artifacts logged: {len(artifacts_to_log)}")
            
            return model, scaler, metrics
            
    except Exception as run_error:
        print(f"❌ MLflow run failed: {run_error}")
        print("Training will continue without MLflow logging...")
        
        # Continue training without MLflow
        model = create_mlp_model()
        model.fit(X_train_scaled, y_train_full)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'test_accuracy': accuracy,
            'test_precision_weighted': precision,
            'test_recall_weighted': recall,
            'test_f1_score_weighted': f1
        }
        
        print(f"Training completed without MLflow:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return model, scaler, metrics

def main():
    """Main function untuk MLflow Project"""
    parser = argparse.ArgumentParser(description='Train Worker Productivity MLP Model')
    parser.add_argument('--data_path', type=str, default='processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='MLflow experiment name')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up temporary files after training')
    
    args = parser.parse_args()
    
    print("🚀 MLflow PROJECT: Worker Productivity Classification")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Experiment: {args.experiment_name or 'Default'}")
    print(f"Cleanup: {args.cleanup}")
    print("=" * 60)
    
    # Train model
    results = train_basic_model(args.data_path, args.experiment_name)
    
    if results:
        model, scaler, metrics = results
        print(f"\n✅ TRAINING SUCCESS!")
        print(f"Final Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Final F1-Score: {metrics['test_f1_score_weighted']:.4f}")
        
        if args.cleanup:
            cleanup_temporary_files()
            print("✅ Cleanup completed")
        
        return 0
    else:
        print("\n❌ TRAINING FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
