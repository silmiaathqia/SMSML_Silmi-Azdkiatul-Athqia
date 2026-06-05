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
import os
import warnings
warnings.filterwarnings('ignore')


def setup_mlflow():
    """Setup MLflow dengan tracking lokal (localhost)"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    experiment_name = "Worker_Productivity_Classification_Sklearn"
    mlflow.set_experiment(experiment_name)

    print("MLflow berhasil dikonfigurasi!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")

    return experiment_name


def load_processed_data():
    """Load preprocessed data dari hasil kriteria 1"""
    try:
        train_data = pd.read_csv('processed_data/data_train.csv')
        val_data   = pd.read_csv('processed_data/data_validation.csv')
        test_data  = pd.read_csv('processed_data/data_test.csv')

        X_train = train_data.drop('productivity_label_encoded', axis=1)
        y_train = train_data['productivity_label_encoded']

        X_val = val_data.drop('productivity_label_encoded', axis=1)
        y_val = val_data['productivity_label_encoded']

        X_test = test_data.drop('productivity_label_encoded', axis=1)
        y_test = test_data['productivity_label_encoded']

        print(f"Data berhasil dimuat!")
        print(f"Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Kelas target: {sorted(y_train.unique())}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_mlp_model():
    """Buat model MLP Classifier untuk klasifikasi produktivitas"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
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
        n_iter_no_change=10,
        early_stopping=True,
        verbose=False
    )
    return model


def plot_confusion_matrix(y_true, y_pred, class_names=['High', 'Low', 'Medium']):
    """Plot dan simpan confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Basic Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'confusion_matrix_basic.png'


def plot_metrics_comparison(metrics_dict):
    """Plot perbandingan metriks evaluasi"""
    metrics = list(metrics_dict.keys())
    values  = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics - Basic Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('metrics_comparison_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'metrics_comparison_basic.png'


def train_basic_model():
    """Latih model MLP menggunakan MLflow autolog"""

    setup_mlflow()

    data = load_processed_data()
    if data is None:
        return None

    X_train, X_val, X_test, y_train, y_val, y_test = data

    # Gabungkan train + val (MLPClassifier mengelola internal validation sendiri)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # Scaling (penting untuk neural network)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")


    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Basic_MLP_Baseline"):

        # Buat dan latih model
        model = create_mlp_model()

        print("\nModel Configuration:")
        print(f"  Hidden layers : {model.hidden_layer_sizes}")
        print(f"  Activation    : {model.activation}")
        print(f"  Solver        : {model.solver}")
        print(f"  Alpha (L2)    : {model.alpha}")
        print(f"  Max iterations: {model.max_iter}")

        print("\nStarting training...")
        model.fit(X_train_scaled, y_train_full)

        print(f"\nTraining completed!")
        print(f"  Iterations : {model.n_iter_}")
        print(f"  Final loss : {model.loss_:.6f}")

        # Evaluasi di test set
        y_pred = model.predict(X_test_scaled)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')
        f1        = f1_score(y_test, y_pred, average='weighted')

        class_names = ['High', 'Low', 'Medium']

        print(f"\nTest Results:")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Buat visualisasi dan log sebagai artefak tambahan
        cm_file      = plot_confusion_matrix(y_test, y_pred, class_names)
        metrics_file = plot_metrics_comparison({
            'Accuracy': accuracy, 'Precision': precision,
            'Recall': recall,     'F1-Score': f1
        })

        # Simpan classification report
        report_file = 'classification_report_basic.txt'
        with open(report_file, 'w') as f:
            f.write("Basic MLP Model - Classification Report\n")
            f.write("=" * 45 + "\n")
            f.write(classification_report(y_test, y_pred, target_names=class_names))

        # Simpan scaler
        scaler_file = 'scaler_basic.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Log artefak tambahan (autolog sudah log model secara otomatis)
        for artifact in [cm_file, metrics_file, report_file, scaler_file]:
            if os.path.exists(artifact):
                mlflow.log_artifact(artifact)
                print(f"Logged artifact: {artifact}")

        # Set tags untuk identifikasi
        mlflow.set_tags({
            "model_type"  : "basic_mlp",
            "framework"   : "sklearn",
            "dataset"     : "worker_productivity",
            "logging_type": "autolog"
        })

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED — MLflow autolog aktif")
        print("=" * 60)
        print(f"Tracking URI : {mlflow.get_tracking_uri()}")
        print("Buka MLflow UI: mlflow ui  (di terminal)")

        metrics_result = {
            'test_accuracy' : accuracy,
            'test_precision': precision,
            'test_recall'   : recall,
            'test_f1'       : f1
        }

        return model, scaler, metrics_result


if __name__ == "__main__":
    print("KLASIFIKASI PRODUKTIVITAS PEKERJA - BASIC MLP MODEL")
    print("=" * 60)
    print("  - MLflow autolog: AKTIF")
    print("  - Tracking URI  : localhost (127.0.0.1:5000)")
    print("=" * 60)

    results = train_basic_model()

    if results:
        model, scaler, metrics = results
        print(f"\nFinal Accuracy : {metrics['test_accuracy']:.4f}")
        print(f"Final F1-Score : {metrics['test_f1']:.4f}")
        print("\n✅ Training selesai. Jalankan 'mlflow ui' untuk melihat hasil.")
    else:
        print("\n❌ Training gagal. Cek pesan error di atas.")
        print("Pastikan folder 'processed_data/' berisi file CSV yang benar.")
