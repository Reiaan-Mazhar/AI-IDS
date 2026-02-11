"""
Real-Time Inference Engine
Connects packet capture to ML models via WebSocket streaming
"""

import time
import threading
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from queue import Queue, Empty
from collections import deque

from report_generator import report_generator

# Import the inference engine
try:
    from inference_engine import InferenceEngine
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: Inference engine not available")


class RealTimeInferenceEngine:
    """Real-time inference engine for network traffic analysis"""
    
    # Feature mapping from packet capture to model features
    CICIDS_FEATURES = [
        'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
        'total_fwd_bytes', 'total_bwd_bytes', 'fwd_packet_length_max',
        'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
        'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
        'bwd_packet_length_std', 'flow_bytes_s', 'flow_packets_s',
        'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
        'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std',
        'bwd_iat_max', 'bwd_iat_min', 'fin_flag_count', 'syn_flag_count',
        'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
        'down_up_ratio', 'avg_packet_size', 'fwd_avg_bytes_bulk',
        'bwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'bwd_avg_packets_bulk',
        'fwd_avg_bulk_rate', 'bwd_avg_bulk_rate', 'subflow_fwd_packets',
        'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes',
        'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'active_mean', 'active_std', 'active_max',
        'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'
    ]
    
    def __init__(self, dataset: str = "cicids2017", model_id: str = "rf_model_1"):
        """
        Initialize real-time inference engine
        
        Args:
            dataset: Dataset type (cicids2017 or nslkdd)
            model_id: Model identifier to use for inference
        """
        self.dataset = dataset
        self.model_id = model_id
        self.inference_engine = InferenceEngine() if INFERENCE_AVAILABLE else None
        
        # Queues and state
        self.flow_queue: Queue = Queue()
        self.results_queue: Queue = Queue()
        self.is_running = False
        self.inference_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'flows_analyzed': 0,
            'threats_detected': 0,
            'benign_count': 0,
            'attack_counts': {},
            'start_time': None,
            'last_prediction_time': None
        }
        
        # Recent predictions buffer
        self.recent_predictions: deque = deque(maxlen=100)
        
        # Callbacks for real-time updates
        self.callbacks: List = []
        
    def add_callback(self, callback):
        """Add callback for prediction results"""
        self.callbacks.append(callback)
        
    def _notify_callbacks(self, result: Dict):
        """Notify all callbacks of a new prediction"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Callback error: {e}")
                
    def _prepare_features(self, flow_data: Dict) -> Optional[pd.DataFrame]:
        """Convert flow data to model input features"""
        try:
            # Create feature vector
            features = {}
            for feature in self.CICIDS_FEATURES:
                if feature in flow_data:
                    features[feature] = flow_data[feature]
                else:
                    features[feature] = 0.0
                    
            # Create DataFrame
            df = pd.DataFrame([features])
            return df
            
        except Exception as e:
            print(f"Feature preparation error: {e}")
            return None
            
    def _run_inference(self, flow_data: Dict) -> Optional[Dict]:
        """Run inference on a single flow"""
        if not self.inference_engine:
            return None
            
        try:
            # Prepare features
            df = self._prepare_features(flow_data)
            if df is None:
                return None
                
            # Get model
            model_info = self.inference_engine.get_model(self.dataset, self.model_id)
            if not model_info:
                return None
                
            model = model_info['model']
            label_mapping = model_info.get('label_mapping', {})
            
            # Run prediction
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(df.values)
                pred_idx = np.argmax(probs[0])
                confidence = float(probs[0][pred_idx])
            else:
                pred_idx = int(model.predict(df.values)[0])
                confidence = 0.95  # Default for models without probability
                
            # Get label
            if isinstance(pred_idx, (int, np.integer)):
                # Reverse lookup
                inv_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {}
                label = inv_mapping.get(pred_idx, f"Class_{pred_idx}")
            else:
                label = str(pred_idx)
                
            # Generate result
            result = {
                'label': label,
                'confidence': confidence,
                'is_threat': label.lower() not in ['benign', 'normal'],
                'timestamp': time.time(),
                'features': df.to_dict(orient='records')[0],  # Include features for XAI
                'flow_info': {
                    'src_ip': flow_data.get('src_ip', 'unknown'),
                    'dst_ip': flow_data.get('dst_ip', 'unknown'),
                    'src_port': flow_data.get('src_port', 0),
                    'dst_port': flow_data.get('dst_port', 0),
                    'protocol': flow_data.get('protocol', 0)
                }
            }
            
            # Auto-generate report for threats
            if result['is_threat']:
                report_id = str(uuid.uuid4())
                result['report_id'] = report_id
                result['report_url'] = f"/static/reports/report_{report_id}.json"
                
                # Fire async generation
                threading.Thread(target=report_generator.generate_report, kwargs={
                    'flow_data': result['flow_info'],
                    'prediction': result,
                    'features': result['features'],
                    'dataset': self.dataset,
                    'model_id': self.model_id,
                    'report_id': report_id
                }).start()

            return result
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None
            
    def _inference_worker(self):
        """Background worker for processing flow queue"""
        while self.is_running:
            try:
                # Get flow from queue with timeout
                flow_data = self.flow_queue.get(timeout=0.5)
                
                # Run inference
                result = self._run_inference(flow_data)
                
                if result:
                    # Update statistics
                    self.stats['flows_analyzed'] += 1
                    self.stats['last_prediction_time'] = time.time()
                    
                    if result['is_threat']:
                        self.stats['threats_detected'] += 1
                        label = result['label']
                        self.stats['attack_counts'][label] = \
                            self.stats['attack_counts'].get(label, 0) + 1
                    else:
                        self.stats['benign_count'] += 1
                        
                    # Add to recent predictions
                    self.recent_predictions.append(result)
                    
                    # Add to results queue
                    self.results_queue.put(result)
                    
                    # Notify callbacks
                    self._notify_callbacks(result)
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                
    def process_flow(self, flow_data: Dict):
        """Add a flow to the processing queue"""
        if self.is_running:
            self.flow_queue.put(flow_data)
            
    def start(self):
        """Start the inference engine"""
        if self.is_running:
            return
            
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        self.inference_thread = threading.Thread(
            target=self._inference_worker, 
            daemon=True
        )
        self.inference_thread.start()
        
        print(f"Real-time inference started (model: {self.model_id})")
        
    def stop(self):
        """Stop the inference engine"""
        self.is_running = False
        
        if self.inference_thread:
            self.inference_thread.join(timeout=2)
            
        print(f"Real-time inference stopped. Analyzed {self.stats['flows_analyzed']} flows")
        
    def get_stats(self) -> Dict:
        """Get current statistics"""
        uptime = 0
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            
        return {
            **self.stats,
            'uptime': uptime,
            'flows_per_second': self.stats['flows_analyzed'] / max(1, uptime),
            'threat_rate': self.stats['threats_detected'] / max(1, self.stats['flows_analyzed']),
            'recent_count': len(self.recent_predictions)
        }
        
    def get_recent_predictions(self, count: int = 10) -> List[Dict]:
        """Get recent predictions"""
        return list(self.recent_predictions)[-count:]
        
    def set_model(self, model_id: str):
        """Change the active model"""
        self.model_id = model_id
        print(f"Switched to model: {model_id}")
        
    def get_result(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get a result from the queue (non-blocking)"""
        try:
            return self.results_queue.get(timeout=timeout)
        except Empty:
            return None


class MockRealTimeEngine:
    """Mock engine for testing when inference is not available"""
    
    def __init__(self, *args, **kwargs):
        self.is_running = False
        self.callbacks = []
        self.stats = {
            'flows_analyzed': 0,
            'threats_detected': 0,
            'benign_count': 0,
            'attack_counts': {},
            'start_time': None
        }
        
    def start(self):
        self.is_running = True
        self.stats['start_time'] = time.time()
        
    def stop(self):
        self.is_running = False
        
    def process_flow(self, flow_data: Dict):
        """Generate mock prediction"""
        import random
        
        labels = ['BENIGN', 'DDoS', 'PortScan', 'DoS Hulk', 'Bot']
        weights = [0.7, 0.1, 0.1, 0.05, 0.05]
        
        label = random.choices(labels, weights)[0]
        is_threat = label != 'BENIGN'
        
        result = {
            'label': label,
            'confidence': random.uniform(0.75, 0.99),
            'is_threat': is_threat,
            'timestamp': time.time(),
            'flow_info': {
                'src_ip': flow_data.get('src_ip', '192.168.1.1'),
                'dst_ip': flow_data.get('dst_ip', '10.0.0.1'),
                'src_port': flow_data.get('src_port', 12345),
                'dst_port': flow_data.get('dst_port', 80),
                'protocol': flow_data.get('protocol', 6)
            }
        }
        
        self.stats['flows_analyzed'] += 1
        if is_threat:
            self.stats['threats_detected'] += 1
            self.stats['attack_counts'][label] = \
                self.stats['attack_counts'].get(label, 0) + 1
        else:
            self.stats['benign_count'] += 1
            
        for callback in self.callbacks:
            try:
                callback(result)
            except:
                pass
                
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
    def get_stats(self):
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        return {**self.stats, 'uptime': uptime}
        
    def set_model(self, model_id):
        pass


# Factory function
def create_realtime_engine(dataset: str = "cicids2017", model_id: str = "rf_model_1"):
    """Create appropriate real-time engine"""
    if INFERENCE_AVAILABLE:
        return RealTimeInferenceEngine(dataset, model_id)
    else:
        print("Using mock inference engine")
        return MockRealTimeEngine(dataset, model_id)
