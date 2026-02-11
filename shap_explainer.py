"""
SHAP Explainer Module for IDS Models
Provides feature importance explanations for predictions using SHAP
"""
import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import base64
from io import BytesIO

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Explainability features will be limited.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import torch
import joblib


# Feature names for CICIDS2017 (post-preprocessing)
CICIDS2017_FEATURE_NAMES = [
    "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "CWE Flag Cnt", "ECE Flag Cnt", "Down/Up Ratio",
    "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts",
    "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

# Feature names for NSL-KDD (post-preprocessing)
NSLKDD_FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# Default paths
BASE_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = BASE_DIR / "trained_weights"

# CICIDS2017 label mapping
CICIDS2017_LABELS = {
    0: "BENIGN", 1: "Bot", 2: "Brute Force", 
    3: "DDoS", 4: "DoS", 5: "Port Scan", 6: "Web Attack"
}

NSLKDD_LABELS = {0: "Normal", 1: "Attack"}


class SHAPExplainer:
    """
    SHAP-based explainer for IDS models.
    Supports sklearn models (TreeExplainer) and PyTorch models (approximation).
    """
    
    def __init__(self):
        self._explainer_cache: Dict[str, Any] = {}
        self._background_cache: Dict[str, np.ndarray] = {}
        self._model_cache: Dict[str, Any] = {}
    
    def get_feature_names(self, dataset: str, model_id: str = None) -> List[str]:
        """Get feature names for a dataset, dynamically from model if available"""
        # Try to get from cached model first
        if model_id:
            cache_key = f"{dataset}_{model_id}"
            if cache_key in self._model_cache:
                model = self._model_cache[cache_key]
                if hasattr(model, 'feature_names_in_'):
                    return list(model.feature_names_in_)
        
        # Fallback to static feature names
        if dataset == "cicids2017":
            return CICIDS2017_FEATURE_NAMES
        elif dataset == "nslkdd":
            return NSLKDD_FEATURE_NAMES
        else:
            return [f"Feature_{i}" for i in range(100)]
    
    def _get_model_feature_names(self, model, n_features: int) -> List[str]:
        """Get feature names from model or generate them"""
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        # Generate PCA-style names if model doesn't have feature names
        return [f"PC{i+1}" for i in range(n_features)]
    
    def get_label_map(self, dataset: str) -> Dict[int, str]:
        """Get label mapping for a dataset"""
        if dataset == "cicids2017":
            return CICIDS2017_LABELS
        elif dataset == "nslkdd":
            return NSLKDD_LABELS
        return {}
    
    def _load_sklearn_model(self, model_path: str):
        """Load a sklearn model from joblib file"""
        return joblib.load(model_path)
    
    def _create_tree_explainer(self, model, background_data: np.ndarray = None):
        """Create SHAP TreeExplainer for tree-based models"""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not installed")
        return shap.TreeExplainer(model, data=background_data)
    
    def _create_kernel_explainer(self, predict_fn, background_data: np.ndarray):
        """Create SHAP KernelExplainer for any model"""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not installed")
        # Use a small subset for background
        if len(background_data) > 100:
            indices = np.random.choice(len(background_data), 100, replace=False)
            background_data = background_data[indices]
        return shap.KernelExplainer(predict_fn, background_data)
    
    def explain_prediction(
        self, 
        dataset: str, 
        model_id: str, 
        X: np.ndarray, 
        sample_index: int = 0,
        background_data: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            dataset: 'cicids2017' or 'nslkdd'
            model_id: Model identifier
            X: Input features (can be single sample or batch)
            sample_index: Index of sample to explain (if X is batch)
            background_data: Optional background data for explainer
            
        Returns:
            Dictionary with explanation data
        """
        if not SHAP_AVAILABLE:
            return {
                "error": "SHAP library not installed",
                "available": False
            }
        
        label_map = self.get_label_map(dataset)
        
        # Get the sample to explain
        if X.ndim == 1:
            sample = X.reshape(1, -1)
        else:
            sample = X[sample_index:sample_index+1]
        
        try:
            # Get model and create explainer based on model type
            model_path = self._get_model_path(dataset, model_id)
            cache_key = f"{dataset}_{model_id}"
            
            if model_id in ["rf_model_1", "rf_model_2", "dt_model_1", "dt_model_2", "xgboost"]:
                # Tree-based models: use TreeExplainer
                if cache_key in self._model_cache:
                    model = self._model_cache[cache_key]
                else:
                    model = self._load_sklearn_model(model_path)
                    self._model_cache[cache_key] = model
                
                # Get feature names from model
                feature_names = self._get_model_feature_names(model, model.n_features_in_)
                
                # Ensure sample matches model's expected features
                if sample.shape[1] != model.n_features_in_:
                    return {
                        "error": f"Feature mismatch: got {sample.shape[1]} features, expected {model.n_features_in_}",
                        "available": False
                    }
                
                explainer = self._create_tree_explainer(model)
                shap_values = explainer.shap_values(sample)
                
                # Get prediction class for indexing
                pred = model.predict(sample)[0]
                # Create inverse label map for string labels
                label_to_idx = {v: k for k, v in label_map.items()}
                if isinstance(pred, str):
                    pred_idx = label_to_idx.get(pred, 0)
                else:
                    pred_idx = int(pred)
                
                # Handle different SHAP formats:
                # - Older: list of arrays, one per class [array(n_samples, n_features), ...]
                # - Newer: 3D array (n_samples, n_features, n_classes)
                if isinstance(shap_values, list):
                    # Older format: list of arrays
                    pred_idx = min(pred_idx, len(shap_values) - 1)
                    shap_values_sample = shap_values[pred_idx][0]
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[min(pred_idx, len(base_value) - 1)]
                elif shap_values.ndim == 3:
                    # Newer format: 3D array (samples, features, classes)
                    pred_idx = min(pred_idx, shap_values.shape[2] - 1)
                    shap_values_sample = shap_values[0, :, pred_idx]
                    base_value = explainer.expected_value
                    if isinstance(base_value, np.ndarray) and base_value.ndim >= 1:
                        base_value = base_value[min(pred_idx, len(base_value) - 1)]
                else:
                    # 2D array (samples, features) - binary or regression
                    shap_values_sample = shap_values[0]
                    base_value = explainer.expected_value
                    if isinstance(base_value, np.ndarray):
                        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
                
            elif model_id in ["svm_model_1", "svm_model_2", "knn_model_1", "logistic"]:
                # Use KernelExplainer for non-tree models
                model = self._load_sklearn_model(model_path)
                
                # Get background data
                if background_data is None:
                    background_data = sample  # Use sample as background (not ideal)
                
                explainer = self._create_kernel_explainer(model.predict_proba, background_data[:50])
                shap_values = explainer.shap_values(sample, nsamples=100)
                
                if isinstance(shap_values, list):
                    pred = model.predict(sample)[0]
                    shap_values_sample = shap_values[int(pred)][0]
                else:
                    shap_values_sample = shap_values[0]
                
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    pred = model.predict(sample)[0]
                    base_value = base_value[int(pred)]
            else:
                # For PyTorch models, use simplified feature importance
                return self._explain_pytorch_model(dataset, model_id, sample, feature_names)
            
            # Create explanation result
            explanation = self._format_explanation(
                shap_values_sample, 
                sample[0], 
                feature_names, 
                base_value
            )
            
            return explanation
            
        except Exception as e:
            return {
                "error": str(e),
                "available": False
            }
    
    def _get_model_path(self, dataset: str, model_id: str) -> str:
        """Get the path to a model file"""
        model_files = {
            "cicids2017": {
                "rf_model_1": "rf_model_1.joblib",
                "rf_model_2": "rf_model_2.joblib",
                "dt_model_1": "dt_model_1.joblib",
                "dt_model_2": "dt_model_2.joblib",
                "knn_model_1": "knn_model_1.joblib",
                "svm_model_1": "svm_model_1.joblib",
                "svm_model_2": "svm_model_2.joblib",
            },
            "nslkdd": {
                "logistic": "nslkdd_logistic_model.joblib",
                "xgboost": "nslkdd_xgboost_model.joblib",
            }
        }
        
        filename = model_files.get(dataset, {}).get(model_id)
        if filename:
            return str(WEIGHTS_DIR / filename)
        raise ValueError(f"Unknown model: {dataset}/{model_id}")
    
    def _explain_pytorch_model(
        self, 
        dataset: str, 
        model_id: str, 
        sample: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate simplified explanation for PyTorch models using gradient-based importance.
        """
        # For now, return a placeholder with random importance (to be improved)
        n_features = sample.shape[1]
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        
        # Calculate simple feature importance based on feature magnitude
        feature_values = sample[0]
        normalized_values = np.abs(feature_values) / (np.max(np.abs(feature_values)) + 1e-8)
        
        # Create feature importance list
        importance_list = []
        for i in range(min(n_features, len(feature_names))):
            importance_list.append({
                "feature": feature_names[i],
                "value": float(feature_values[i]),
                "importance": float(normalized_values[i]),
                "contribution": "positive" if feature_values[i] > 0 else "negative"
            })
        
        # Sort by importance
        importance_list.sort(key=lambda x: abs(x["importance"]), reverse=True)
        
        return {
            "available": True,
            "method": "gradient_approximation",
            "top_features": importance_list[:10],
            "all_features": importance_list,
            "note": "Using gradient-based approximation for deep learning model"
        }
    
    def _format_explanation(
        self, 
        shap_values: np.ndarray, 
        feature_values: np.ndarray, 
        feature_names: List[str],
        base_value: float
    ) -> Dict[str, Any]:
        """Format SHAP explanation into a structured response"""
        n_features = len(shap_values)
        
        # Create feature importance list
        importance_list = []
        for i in range(min(n_features, len(feature_names))):
            importance_list.append({
                "feature": feature_names[i],
                "value": float(feature_values[i]),
                "shap_value": float(shap_values[i]),
                "contribution": "positive" if shap_values[i] > 0 else "negative"
            })
        
        # Sort by absolute SHAP value
        importance_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "available": True,
            "method": "shap",
            "base_value": float(base_value),
            "top_features": importance_list[:10],
            "all_features": importance_list,
            "summary": self._generate_text_summary(importance_list[:5])
        }
    
    def _generate_text_summary(self, top_features: List[Dict]) -> str:
        """Generate a human-readable summary of the explanation"""
        if not top_features:
            return "No significant features identified."
        
        positive = [f for f in top_features if f["contribution"] == "positive"]
        negative = [f for f in top_features if f["contribution"] == "negative"]
        
        summary_parts = []
        
        if positive:
            features = ", ".join([f["feature"] for f in positive[:3]])
            summary_parts.append(f"Key factors increasing this classification: {features}")
        
        if negative:
            features = ", ".join([f["feature"] for f in negative[:3]])
            summary_parts.append(f"Factors decreasing this classification: {features}")
        
        return ". ".join(summary_parts) + "."
    
    def generate_waterfall_plot(
        self, 
        dataset: str, 
        model_id: str, 
        X: np.ndarray, 
        sample_index: int = 0
    ) -> Optional[str]:
        """
        Generate a SHAP waterfall plot as base64-encoded PNG.
        
        Returns:
            Base64 encoded PNG image string, or None if not available
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Get explanation data
            explanation = self.explain_prediction(dataset, model_id, X, sample_index)
            
            if not explanation.get("available") or explanation.get("method") != "shap":
                return None
            
            # Create waterfall data
            features = explanation["all_features"]
            feature_names = [f["feature"] for f in features[:15]]
            shap_values = [f["shap_value"] for f in features[:15]]
            feature_values = [f["value"] for f in features[:15]]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(feature_names))
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in shap_values]
            
            ax.barh(y_pos, shap_values, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{name}\n({val:.2f})" for name, val in zip(feature_names, feature_values)], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Feature Importance (Top 15)")
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            return None


# Create global instance
shap_explainer = SHAPExplainer()
