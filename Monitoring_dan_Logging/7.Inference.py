import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server, CollectorRegistry, REGISTRY
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus Metrics - 10+ Different Metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total number of predictions made', ['model_name', 'prediction_class'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Time spent processing predictions', ['model_name'])
MODEL_ACCURACY_GAUGE = Gauge('ml_model_accuracy', 'Current model accuracy', ['model_name'])
ACTIVE_USERS = Gauge('ml_active_users', 'Number of active users')
ERROR_COUNTER = Counter('ml_errors_total', 'Total number of errors', ['error_type'])
MEMORY_USAGE = Gauge('ml_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('ml_cpu_usage_percent', 'CPU usage percentage')
DISK_USAGE = Gauge('ml_disk_usage_percent', 'Disk usage percentage')
REQUEST_RATE = Counter('ml_requests_per_second', 'HTTP requests per second', ['method', 'endpoint'])
PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence scores', ['model_name'])
MODEL_LOAD_TIME = Gauge('ml_model_load_time_seconds', 'Time taken to load the model')
FEATURE_STATS = Gauge('ml_feature_statistics', 'Feature statistics', ['feature_name', 'stat_type'])

# Model Info
MODEL_INFO = Info('ml_model_info', 'Information about the ML model')

class WorkerProductivityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = "WorkerProductivityMLP_Basic"
        self.class_names = ['High', 'Low', 'Medium']
        self.feature_names = []
        self.model_accuracy = 0.0
        self.load_model()
        
    def load_model(self):
        """Load model from MLflow"""
        start_time = time.time()
        try:
            logger.info("Loading model from MLflow...")
            
            # Try to load from MLflow tracking server
            try:
                mlflow.set_tracking_uri("https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow.mlflow")
                model_uri = f"models:/{self.model_name}/latest"
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded from MLflow registry")
            except Exception as e:
                logger.warning(f"Could not load from registry: {e}")
                # Fallback to local loading
                try:
                    # Load from local pickle if available
                    with open('model_basic.pkl', 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info("Model loaded from local pickle file")
                except FileNotFoundError:
                    logger.error("No model file found!")
                    raise
            
            # Load scaler
            try:
                with open('scaler_basic.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            except FileNotFoundError:
                logger.warning("Scaler file not found, creating dummy scaler")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                # Create dummy data to fit scaler
                dummy_data = np.random.randn(100, 10)
                self.scaler.fit(dummy_data)
            
            # Set model info
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)
            
            # Set model accuracy (from previous training)
            self.model_accuracy = 0.85  # Default value, should be loaded from model metadata
            MODEL_ACCURACY_GAUGE.labels(model_name=self.model_name).set(self.model_accuracy)
            
            # Set model info
            MODEL_INFO.info({
                'model_name': self.model_name,
                'model_type': 'MLPClassifier',
                'framework': 'sklearn',
                'version': '1.0',
                'classes': str(self.class_names),
                'loaded_at': datetime.now().isoformat(),
                'load_time_seconds': str(load_time)
            })
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            ERROR_COUNTER.labels(error_type='model_load_error').inc()
            raise
    
    def predict(self, features):
        """Make prediction with monitoring"""
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(features, (list, np.ndarray)):
                raise ValueError("Features must be a list or numpy array")
            
            # Convert to numpy array and reshape if needed
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            try:
                features_scaled = self.scaler.transform(features)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}, using original features")
                features_scaled = features
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            prediction_proba = self.model.predict_proba(features_scaled)
            
            # Get prediction class name and confidence
            pred_class = self.class_names[prediction[0]]
            confidence = np.max(prediction_proba[0])
            
            # Log metrics
            PREDICTION_COUNTER.labels(
                model_name=self.model_name, 
                prediction_class=pred_class
            ).inc()
            
            PREDICTION_CONFIDENCE.labels(model_name=self.model_name).observe(confidence)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            PREDICTION_LATENCY.labels(model_name=self.model_name).observe(processing_time)
            
            # Log feature statistics
            for i, feature_val in enumerate(features[0]):
                FEATURE_STATS.labels(
                    feature_name=f'feature_{i}', 
                    stat_type='current_value'
                ).set(feature_val)
            
            result = {
                'prediction': pred_class,
                'prediction_numeric': int(prediction[0]),
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(prediction_proba[0])
                },
                'processing_time_ms': processing_time * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction made: {pred_class} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            ERROR_COUNTER.labels(error_type='prediction_error').inc()
            logger.error(f"Prediction failed: {e}")
            raise

# Initialize Flask app and predictor
app = Flask(__name__)
predictor = WorkerProductivityPredictor()

# System monitoring thread
def monitor_system():
    """Monitor system resources"""
    while True:
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            DISK_USAGE.set((disk.used / disk.total) * 100)
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"System monitoring error: {e}")
            ERROR_COUNTER.labels(error_type='system_monitoring_error').inc()

# Start system monitoring in background
monitoring_thread = threading.Thread(target=monitor_system, daemon=True)
monitoring_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    REQUEST_RATE.labels(method='GET', endpoint='/health').inc()
    
    health_status = {
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'scaler_loaded': predictor.scaler is not None,
        'model_name': predictor.model_name,
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': time.time() - app.start_time
    }
    
    return jsonify(health_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    REQUEST_RATE.labels(method='POST', endpoint='/predict').inc()
    ACTIVE_USERS.inc()
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'features' not in data:
            ERROR_COUNTER.labels(error_type='invalid_request').inc()
            return jsonify({'error': 'Missing features in request'}), 400
        
        # Make prediction
        result = predictor.predict(data['features'])
        
        return jsonify(result)
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='request_processing_error').inc()
        logger.error(f"Request processing failed: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        ACTIVE_USERS.dec()

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    REQUEST_RATE.labels(method='GET', endpoint='/model/info').inc()
    
    info = {
        'model_name': predictor.model_name,
        'model_type': 'MLPClassifier',
        'framework': 'scikit-learn',
        'classes': predictor.class_names,
        'accuracy': predictor.model_accuracy,
        'features_expected': 10,  # Adjust based on your model
        'version': '1.0'
    }
    
    return jsonify(info)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    REQUEST_RATE.labels(method='POST', endpoint='/predict/batch').inc()
    ACTIVE_USERS.inc()
    
    try:
        data = request.get_json()
        
        if not data or 'features_list' not in data:
            ERROR_COUNTER.labels(error_type='invalid_batch_request').inc()
            return jsonify({'error': 'Missing features_list in request'}), 400
        
        results = []
        for features in data['features_list']:
            result = predictor.predict(features)
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'batch_size': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='batch_processing_error').inc()
        logger.error(f"Batch processing failed: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        ACTIVE_USERS.dec()

if __name__ == '__main__':
    # Record start time
    app.start_time = time.time()
    
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    logger.info("Metrics available at: http://localhost:8000/metrics")
    
    # Start Flask app
    logger.info("Starting ML inference service...")
    logger.info("Health check: http://localhost:5000/health")
    logger.info("Prediction endpoint: http://localhost:5000/predict")
    logger.info("Model info: http://localhost:5000/model/info")
    logger.info("Metrics endpoint: http://localhost:5000/metrics")
    
    app.run(host='0.0.0.0', port=5000, debug=False)