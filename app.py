"""
Flask Web Application for Intrusion Detection Model Inference
With SHAP Explainability, Security Advisor (Grok API), and Real-Time Detection
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import io
import traceback
import json
import threading
import os
import glob
from urllib.parse import urlparse

from inference_engine import engine
from shap_explainer import shap_explainer
from security_advisor import get_security_advice, generate_firewall_rules, generate_explanation_text, execute_response_action
from notification_service import send_webhook, get_notification_history, test_webhook, clear_notification_history, send_email_alert
from extensions import db, login_manager, mail
from auth_models import User
from flask_login import login_user, logout_user, login_required, current_user
from flask_mail import Message
from werkzeug.security import generate_password_hash, check_password_hash

# Real-time modules
try:
    from packet_capture import PacketCapture, get_available_interfaces
    from attack_simulator import AttackSimulator, AttackConfig, ATTACK_TYPES
    from realtime_engine import create_realtime_engine
    REALTIME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real-time modules not fully available: {e}")
    REALTIME_AVAILABLE = False

# Demo traffic generator (works without sudo)
try:
    from demo_traffic import DemoTrafficGenerator, DEMO_ATTACK_TYPES
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Demo traffic module not available: {e}")
    DEMO_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-please-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

CORS(app)

# Initialize Extensions
db.init_app(app)
login_manager.init_app(app)
mail.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store recent predictions for explanation requests
prediction_cache = {}

# Real-time components
packet_capture = None
attack_simulator = None
realtime_engine = None
demo_generator = None


@app.route("/")
@login_required
def index():
    """Serve the main page"""
    return render_template("index.html", user=current_user)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = True if request.form.get("remember") else False
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            
            # Handle next redirect safely
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('index')
            return redirect(next_page)
            
        flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")
        
        if password != confirm:
            flash("Passwords do not match", "error")
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return redirect(url_for('register'))
            
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


@app.route("/api/models")
def get_models():
    """Get available models grouped by dataset"""
    return jsonify({
        "cicids2017": engine.get_available_models("cicids2017"),
        "nslkdd": engine.get_available_models("nslkdd")
    })


@app.route("/api/labels/<dataset>")
def get_labels(dataset):
    """Get label mapping for a dataset"""
    return jsonify(engine.get_label_map(dataset))


# Global cache for XAI
last_inference_features = None


@app.route("/api/infer", methods=["POST"])
def infer():
    """Run inference on uploaded data"""
    global last_inference_features
    try:
        dataset = request.form.get("dataset")
        model_id = request.form.get("model")
        
        if not dataset or not model_id:
            return jsonify({"error": "Missing dataset or model parameter"}), 400
        
        # Handle file upload
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        
        # Drop label columns if present
        label_cols = ["label", "label_id", "y_test"]
        y_true = None
        
        for col in label_cols:
            if col in df.columns:
                y_true = df[col].values
                break
                
        df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")
        
        X = df.values.astype(np.float32)
        
        # Cache for XAI - need global declaration to update the global variable
        global last_inference_features
        last_inference_features = X
        
        # Run inference
        results = engine.predict(dataset, model_id, X, y_true)
        results["num_samples"] = len(X)
        results["feature_columns"] = df.columns.tolist()
        
        return jsonify(results)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare multiple models on the same data"""
    try:
        dataset = request.form.get("dataset")
        model_ids = request.form.getlist("models")
        
        if not dataset or not model_ids:
            return jsonify({"error": "Missing dataset or models parameter"}), 400
        
        # Handle file upload
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        
        # Drop label columns if present
        label_cols = ["label", "label_id", "y_test"]
        y_true = None
        
        for col in label_cols:
            if col in df.columns:
                y_true = df[col].values
                break
                
        df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")
        
        X = df.values.astype(np.float32)
        
        # Run comparison
        results = engine.compare_models(dataset, model_ids, X, y_true)
        
        return jsonify({
            "results": results,
            "num_samples": len(X)
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def explain_prediction():
    """Generate SHAP explanation for a prediction"""
    try:
        data = request.get_json()
        dataset = data.get("dataset")
        model_id = data.get("model")
        sample_index = data.get("sample_index", 0)
        label = data.get("label", "Unknown")
        confidence = data.get("confidence", 0.0)
        
        if not dataset or not model_id:
            return jsonify({"error": "Missing dataset or model parameter"}), 400
        
        # Get features from request or cache
        features = data.get("features")
        
        X = None
        if features is not None:
            # Handle dictionary of features (from real-time/demo mode)
            if isinstance(features, dict):
                feature_names = shap_explainer.get_feature_names(dataset)
                # Create ordered list, defaulting to 0 if missing
                ordered_features = [float(features.get(name, 0.0)) for name in feature_names]
                X = np.array([ordered_features], dtype=np.float32)
            else:
                # Assume it's already a list/array
                X = np.array(features, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
        elif last_inference_features is not None:
            # Try to use cached data
            if 0 <= sample_index < len(last_inference_features):
                X = last_inference_features[sample_index:sample_index+1]
            else:
                return jsonify({"error": f"Sample index {sample_index} out of range for cached data"}), 400
        else:
            return jsonify({"error": "No features provided"}), 400
        
        # Try to generate SHAP explanation
        explanation = {}
        try:
            explanation = shap_explainer.explain_prediction(
                dataset=dataset,
                model_id=model_id,
                X=X,
                sample_index=0  # Always 0 since X is single sample
            )
        except Exception as shap_error:
            print(f"SHAP explanation failed: {shap_error}")
            explanation = {"error": str(shap_error), "top_features": []}
        
        # Generate text explanation
        if explanation.get("top_features"):
            explanation["text_explanation"] = generate_explanation_text(
                attack_type=label,
                top_features=explanation["top_features"],
                confidence=confidence
            )
        else:
            # Fallback: Try to use LLM even without SHAP features
            is_threat = label.lower() not in ['benign', 'normal', 'unknown']
            if is_threat:
                try:
                    explanation["text_explanation"] = generate_explanation_text(
                        attack_type=label,
                        top_features=[], # Pass empty list, LLM agent now handles it
                        confidence=confidence
                    )
                except Exception as e:
                     explanation["text_explanation"] = f"The AI model detected a {label} attack with {confidence*100:.1f}% confidence. (AI Explanation unavailable: {str(e)})"
            else:
                explanation["text_explanation"] = f"The flow was classified as {label} with {confidence*100:.1f}% confidence. This appears to be normal network traffic."
        
        # Try to generate waterfall plot
        try:
            plot_base64 = shap_explainer.generate_waterfall_plot(
                dataset=dataset,
                model_id=model_id,
                X=X,
                sample_index=0
            )
            if plot_base64:
                explanation["waterfall_plot"] = plot_base64
        except Exception as plot_error:
            print(f"Waterfall plot failed: {plot_error}")
        
        # Add fallback security advice for threats
        is_threat = label.lower() not in ['benign', 'normal', 'unknown']
        if is_threat:
            explanation["security_advice"] = get_security_advice(
                attack_type=label,
                confidence=confidence,
                feature_importance=explanation.get("top_features")
            )
        
        return jsonify(explanation)
    
    except Exception as e:
        traceback.print_exc()
        print(f"Explain Error: {str(e)}")
        return jsonify({"error": str(e), "available": False}), 500


@app.route("/api/security-advice", methods=["POST"])
def get_security_advice_endpoint():
    """Get AI-powered security advice for a detected threat"""
    try:
        data = request.get_json()
        attack_type = data.get("attack_type")
        
        if not attack_type:
            return jsonify({"error": "Missing attack_type parameter"}), 400
        
        # Get optional parameters
        source_ip = data.get("source_ip")
        destination_ip = data.get("destination_ip")
        confidence = data.get("confidence", 0.0)
        feature_importance = data.get("feature_importance")
        additional_context = data.get("additional_context")
        
        # Get advice from Grok
        advice = get_security_advice(
            attack_type=attack_type,
            source_ip=source_ip,
            destination_ip=destination_ip,
            confidence=confidence,
            feature_importance=feature_importance,
            additional_context=additional_context
        )
        
        return jsonify(advice)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/firewall-rules", methods=["POST"])
def generate_firewall_rules_endpoint():
    """Generate firewall rules for blocking threats"""
    try:
        data = request.get_json()
        source_ip = data.get("source_ip")
        
        if not source_ip:
            return jsonify({"error": "Missing source_ip parameter"}), 400
        
        action = data.get("action", "DROP")
        protocol = data.get("protocol")
        port = data.get("port")
        
        rules = generate_firewall_rules(
            source_ip=source_ip,
            action=action,
            protocol=protocol,
            port=port
        )
        
        return jsonify({
            "success": True,
            "rules": rules,
            "source_ip": source_ip
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/response/execute", methods=["POST"])
def execute_response_endpoint():
    """Execute a security response action (threat mitigation)"""
    try:
        data = request.get_json()
        action = data.get("action")
        target = data.get("target")

        if not action or not target:
            return jsonify({"error": "Missing action or target parameter"}), 400

        result = execute_response_action(action, target)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


# ===================================
# Threat Reports API Endpoints
# ===================================

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'reports')

@app.route("/api/reports")
def list_reports():
    """List all threat reports with summary data"""
    try:
        reports = []
        report_files = glob.glob(os.path.join(REPORTS_DIR, 'report_*.json'))
        
        for filepath in report_files:
            try:
                with open(filepath, 'r') as f:
                    report = json.load(f)
                    # Return summary data only
                    reports.append({
                        'report_id': report.get('report_id'),
                        'timestamp': report.get('timestamp'),
                        'classification': report.get('classification', {}),
                        'flow_details': report.get('flow_details', {}),
                        'has_xai': bool(report.get('analysis', {}).get('top_features')),
                        'has_security_advice': bool(report.get('security_advice'))
                    })
            except (json.JSONDecodeError, IOError) as e:
                continue
        
        # Sort by timestamp descending (newest first)
        reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'reports': reports,
            'count': len(reports)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route("/api/reports/<report_id>")
def get_report(report_id):
    """Get full report details including XAI analysis"""
    try:
        filepath = os.path.join(REPORTS_DIR, f'report_{report_id}.json')
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Report not found', 'success': False}), 404
            
        with open(filepath, 'r') as f:
            report = json.load(f)
            
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route("/api/reports", methods=["DELETE"])
def clear_reports():
    """Clear all threat reports"""
    try:
        report_files = glob.glob(os.path.join(REPORTS_DIR, 'report_*.json'))
        deleted_count = 0
        
        for filepath in report_files:
            try:
                os.remove(filepath)
                deleted_count += 1
            except OSError as e:
                print(f"Failed to delete {filepath}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} reports',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ===================================
# Real-Time Detection API Endpoints
# ===================================

@app.route("/api/realtime/interfaces")
def get_interfaces():
    """Get available network interfaces"""
    if not REALTIME_AVAILABLE:
        return jsonify({"error": "Real-time modules not available"}), 503
        
    try:
        interfaces = get_available_interfaces()
        return jsonify({"interfaces": interfaces})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/realtime/start", methods=["POST"])
def start_realtime_capture():
    """Start real-time packet capture and inference"""
    global packet_capture, realtime_engine
    
    if not REALTIME_AVAILABLE:
        return jsonify({"error": "Real-time modules not available"}), 503
        
    try:
        data = request.get_json() or {}
        interface = data.get("interface", "lo")
        dataset = data.get("dataset", "cicids2017")
        model_id = data.get("model", "rf_model_1")
        
        # Stop existing capture if running
        if packet_capture and packet_capture.is_capturing:
            packet_capture.stop()
            
        # Create real-time engine
        realtime_engine = create_realtime_engine(dataset, model_id)
        
        # Capture current user email for alerts (if logged in)
        active_email = current_user.email if current_user.is_authenticated else None

        # Add WebSocket callback for streaming results
        def on_prediction(result):
            # Send email alert if threat detected and active user exists
            if active_email and result.get('is_threat') and result.get('confidence', 0) > 0.8:
                try:
                    with app.app_context():
                        send_email_alert(
                            attack_type=result['label'],
                            confidence=result['confidence'],
                            source_ip=result['flow_info'].get('src_ip', 'Unknown'),
                            user_email=active_email
                        )
                except Exception as e:
                    print(f"Failed to send email alert: {e}")

            # Automated Response: Block IP if confidence is very high
            response_result = None
            if result.get('is_threat') and result.get('confidence', 0) > 0.9:
                target_ip = result['flow_info'].get('src_ip')
                if target_ip and target_ip != '127.0.0.1':
                    try:
                        # Call the security agent to ACT
                        response_result = execute_response_action("BLOCK_IP", target_ip)
                        # Log success
                        print(f"AUTOMATED RESPONSE: {response_result.get('message')}")
                    except Exception as e:
                        print(f"Response failed: {e}")

            socketio.emit('prediction', {
                'label': result['label'],
                'confidence': result['confidence'],
                'is_threat': result['is_threat'],
                'timestamp': result['timestamp'],
                'flow_info': result['flow_info'],
                'response_action': response_result # Send action details to UI
            })
            
        realtime_engine.add_callback(on_prediction)
        realtime_engine.start()
        
        # Create and start packet capture
        packet_capture = PacketCapture(interface=interface, flow_timeout=5.0)
        packet_capture.add_callback(realtime_engine.process_flow)
        packet_capture.start()
        
        return jsonify({
            "success": True,
            "message": f"Started capture on {interface}",
            "interface": interface,
            "model": model_id
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/realtime/stop", methods=["POST"])
def stop_realtime_capture():
    """Stop real-time packet capture"""
    global packet_capture, realtime_engine
    
    try:
        if packet_capture:
            packet_capture.stop()
            packet_capture = None
            
        if realtime_engine:
            realtime_engine.stop()
            realtime_engine = None
            
        return jsonify({"success": True, "message": "Capture stopped"})
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/realtime/status")
def get_realtime_status():
    """Get current real-time capture status"""
    try:
        status = {
            "capture_active": packet_capture.is_capturing if packet_capture else False,
            "inference_active": realtime_engine.is_running if realtime_engine else False
        }
        
        if packet_capture:
            status["capture_stats"] = packet_capture.get_stats()
        if realtime_engine:
            status["inference_stats"] = realtime_engine.get_stats()
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/attack/types")
def get_attack_types():
    """Get available attack simulation types"""
    if not REALTIME_AVAILABLE:
        return jsonify({"error": "Attack simulator not available"}), 503
        
    return jsonify({"attacks": ATTACK_TYPES})


@app.route("/api/attack/start", methods=["POST"])
def start_attack():
    """Start an attack simulation"""
    global attack_simulator
    
    if not REALTIME_AVAILABLE:
        return jsonify({"error": "Attack simulator not available"}), 503
        
    try:
        data = request.get_json() or {}
        attack_type = data.get("attack_type", "port_scan")
        target_ip = data.get("target_ip", "127.0.0.1")
        target_port = data.get("target_port", 80)
        duration = data.get("duration", 10)
        intensity = data.get("intensity", "medium")
        
        # Validate attack type
        if attack_type not in ATTACK_TYPES:
            return jsonify({"error": f"Unknown attack type: {attack_type}"}), 400
            
        # Create callback for WebSocket updates
        def on_attack_log(message):
            socketio.emit('attack_log', {'message': message})
            
        attack_simulator = AttackSimulator(callback=on_attack_log)
        
        config = AttackConfig(
            target_ip=target_ip,
            target_port=target_port,
            duration=duration,
            intensity=intensity
        )
        
        success = attack_simulator.start_attack(attack_type, config)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Started {ATTACK_TYPES[attack_type]['name']} attack",
                "attack_type": attack_type,
                "config": {
                    "target_ip": target_ip,
                    "target_port": target_port,
                    "duration": duration,
                    "intensity": intensity
                }
            })
        else:
            return jsonify({"error": "Failed to start attack"}), 500
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/attack/stop", methods=["POST"])
def stop_attack():
    """Stop the current attack simulation"""
    global attack_simulator
    
    try:
        if attack_simulator:
            attack_simulator.stop_attack()
            attack_simulator = None
            
        return jsonify({"success": True, "message": "Attack stopped"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/attack/status")
def get_attack_status():
    """Get current attack simulation status"""
    try:
        if attack_simulator:
            return jsonify(attack_simulator.get_status())
        return jsonify({"is_running": False, "current_attack": None})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================
# Demo Mode API Endpoints
# ===================================

@app.route("/api/demo/start", methods=["POST"])
def start_demo():
    """Start demo traffic generation (no sudo required)"""
    global demo_generator
    
    if not DEMO_AVAILABLE:
        return jsonify({"error": "Demo module not available"}), 503
        
    try:
        data = request.get_json() or {}
        mode = data.get("mode", "mixed")  # 'benign', 'attack', 'mixed'
        attack_type = data.get("attack_type", None)
        speed = data.get("speed", "medium")  # 'slow', 'medium', 'fast'
        
        # Stop existing demo if running
        if demo_generator and demo_generator.is_running:
            demo_generator.stop()
            
        # Capture current user email for alerts
        active_email = current_user.email if current_user.is_authenticated else None
            
        # Create callback for WebSocket streaming
        def on_prediction(result):
            # Send email alerts for high-confidence threats to active user
            if active_email and result.get('is_threat') and result.get('confidence', 0) > 0.8:
                try:
                    with app.app_context():
                        send_email_alert(
                            attack_type=result['label'],
                            confidence=result['confidence'],
                            source_ip=result['flow_info'].get('src_ip', 'Unknown'),
                            user_email=active_email
                        )
                except Exception as e:
                    print(f"Failed to send email alerts: {e}")

            socketio.emit('prediction', {
                'label': result['label'],
                'confidence': result['confidence'],
                'is_threat': result['is_threat'],
                'timestamp': result['timestamp'],
                'flow_info': result['flow_info'],
                'description': result.get('description', '')
            })
            
            # Simulate automated response in demo
            if result.get('is_threat') and result.get('confidence', 0) > 0.9:
                target_ip = result['flow_info'].get('src_ip')
                if target_ip:
                    # In demo mode, we just log it via socketio as an 'action' event or just log
                    # Actually, execute_response_action handles simulation.
                    res = execute_response_action("BLOCK_IP", target_ip)
                    socketio.emit('action_log', {'message': f"🛡️ AGENT ACTED: {res['message']}", 'type': 'success'})
            
        demo_generator = DemoTrafficGenerator(callback=on_prediction)
        demo_generator.start(mode=mode, attack_type=attack_type, speed=speed)
        
        return jsonify({
            "success": True,
            "message": f"Demo started in {mode} mode at {speed} speed",
            "mode": mode,
            "speed": speed
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/demo/stop", methods=["POST"])
def stop_demo():
    """Stop demo traffic generation"""
    global demo_generator
    
    try:
        if demo_generator:
            stats = demo_generator.get_stats()
            demo_generator.stop()
            demo_generator = None
            return jsonify({
                "success": True,
                "message": "Demo stopped",
                "final_stats": stats
            })
            
        return jsonify({"success": True, "message": "No demo running"})
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/demo/status")
def get_demo_status():
    """Get demo generation status"""
    try:
        if demo_generator:
            return jsonify(demo_generator.get_stats())
        return jsonify({"is_running": False, "total_flows": 0})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo/attack-burst", methods=["POST"])
def trigger_attack_burst():
    """Trigger a burst of attack traffic"""
    global demo_generator
    
    if not DEMO_AVAILABLE:
        return jsonify({"error": "Demo module not available"}), 503
        
    try:
        data = request.get_json() or {}
        attack_type = data.get("attack_type", "portscan")
        count = data.get("count", 10)
        
        # Create demo generator if needed
        if not demo_generator:
            def on_prediction(result):
                socketio.emit('prediction', {
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'is_threat': result['is_threat'],
                    'timestamp': result['timestamp'],
                    'flow_info': result['flow_info'],
                    'description': result.get('description', '')
                })
            demo_generator = DemoTrafficGenerator(callback=on_prediction)
            
        demo_generator.trigger_attack_burst(attack_type, count)
        
        return jsonify({
            "success": True,
            "message": f"Triggered {count} {attack_type} attack packets"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo/attacks")
def get_demo_attacks():
    """Get available demo attack types"""
    if not DEMO_AVAILABLE:
        return jsonify({"error": "Demo module not available"}), 503
        
    return jsonify({"attacks": DEMO_ATTACK_TYPES})


# ===================================
# WebSocket Event Handlers
# ===================================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"Client connected")
    emit('connected', {'message': 'Connected to IDS server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"Client disconnected")


@socketio.on('ping_server')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'timestamp': pd.Timestamp.now().isoformat()})


# ===================================
# Notification API Endpoints
# ===================================
@app.route("/api/notifications/history", methods=["GET"])
def get_notifications():
    """Get notification history"""
    return jsonify({
        "success": True,
        "notifications": get_notification_history()
    })


@app.route("/api/notifications/test", methods=["POST"])
def test_notification():
    """Test webhook notification"""
    result = test_webhook()
    return jsonify(result)


@app.route("/api/notifications/clear", methods=["DELETE"])
def clear_notifications():
    """Clear notification history"""
    clear_notification_history()
    return jsonify({"success": True, "message": "Notification history cleared"})


@app.route("/api/notifications/send", methods=["POST"])
def send_notification():
    """Manually send a notification"""
    try:
        data = request.get_json()
        attack_type = data.get("attack_type", "Unknown")
        source_ip = data.get("source_ip")
        destination_ip = data.get("destination_ip")
        confidence = data.get("confidence", 0.0)
        
        # Generate firewall rules
        rules = []
        if source_ip:
            rules = generate_firewall_rules(source_ip)
        
        result = send_webhook(
            attack_type=attack_type,
            source_ip=source_ip,
            destination_ip=destination_ip,
            confidence=confidence,
            firewall_rules=rules
        )
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ===================================
# Firewall Rules API Endpoints
# ===================================
@app.route("/api/firewall-rules", methods=["POST"])
def get_firewall_rules():
    """Generate firewall rules for a threat"""
    try:
        data = request.get_json()
        source_ip = data.get("source_ip")
        attack_type = data.get("attack_type", "Unknown")
        protocol = data.get("protocol")
        port = data.get("port")
        
        if not source_ip:
            return jsonify({"error": "source_ip is required"}), 400
        
        # Generate basic rules
        rules = generate_firewall_rules(
            source_ip=source_ip,
            action="DROP",
            protocol=protocol,
            port=port
        )
        
        # Add attack-specific rules
        attack_rules = get_attack_specific_rules(attack_type, source_ip, port)
        
        return jsonify({
            "success": True,
            "source_ip": source_ip,
            "attack_type": attack_type,
            "rules": rules,
            "attack_specific": attack_rules,
            "instructions": [
                "1. Review the suggested rules below",
                "2. Copy the commands to your terminal",
                "3. Run with sudo to apply",
                "4. Monitor logs for effectiveness"
            ]
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def get_attack_specific_rules(attack_type: str, source_ip: str, port: int = None) -> list:
    """Get attack-specific firewall rules"""
    rules = []
    
    if attack_type in ["DDoS", "DoS"]:
        rules = [
            f"# Rate limiting for DDoS mitigation",
            f"iptables -A INPUT -s {source_ip} -m limit --limit 10/minute --limit-burst 20 -j ACCEPT",
            f"iptables -A INPUT -s {source_ip} -j DROP",
            f"# Enable SYN flood protection",
            f"echo 1 > /proc/sys/net/ipv4/tcp_syncookies"
        ]
    elif attack_type == "Port Scan":
        rules = [
            f"# Block port scanner",
            f"iptables -A INPUT -s {source_ip} -m recent --name portscan --set -j DROP",
            f"iptables -A INPUT -m recent --name portscan --rcheck --seconds 86400 -j DROP"
        ]
    elif attack_type == "Brute Force":
        rules = [
            f"# Block brute force attempts",
            f"iptables -A INPUT -s {source_ip} -p tcp --dport 22 -j DROP",
            f"iptables -A INPUT -s {source_ip} -p tcp --dport 21 -j DROP",
            f"# Install fail2ban for better protection",
            f"# apt-get install fail2ban"
        ]
    elif attack_type == "Bot":
        rules = [
            f"# Block botnet C&C communication",
            f"iptables -A OUTPUT -d {source_ip} -j DROP",
            f"iptables -A INPUT -s {source_ip} -j DROP",
            f"# Check for malware",
            f"# clamscan -r /home"
        ]
    elif attack_type == "Web Attack":
        rules = [
            f"# Block web attacker",
            f"iptables -A INPUT -s {source_ip} -p tcp --dport 80 -j DROP",
            f"iptables -A INPUT -s {source_ip} -p tcp --dport 443 -j DROP",
            f"# Enable mod_security if using Apache/Nginx"
        ]
    else:
        rules = [
            f"# Generic block rule",
            f"iptables -A INPUT -s {source_ip} -j DROP"
        ]
    
    return rules


if __name__ == "__main__":
    print("Starting Intrusion Detection Inference Server...")
    print("Features: Model Inference | SHAP Explainability | Security Advisor | Real-Time Detection")
    print("\nAvailable models:")
    for dataset in ["cicids2017", "nslkdd"]:
        print(f"\n  {dataset.upper()}:")
        for model in engine.get_available_models(dataset):
            print(f"    - {model['name']} ({model['id']})")
    print("\nAPI Endpoints:")
    print("  - POST /api/infer - Run model inference")
    print("  - POST /api/compare - Compare multiple models")
    print("  - POST /api/explain - Get SHAP explanation")
    print("  - POST /api/security-advice - Get AI security advice")
    print("  - POST /api/firewall-rules - Generate firewall rules")
    print("  - GET  /api/realtime/interfaces - List network interfaces")
    print("  - POST /api/realtime/start - Start real-time capture")
    print("  - POST /api/realtime/stop - Stop real-time capture")
    print("  - POST /api/attack/start - Start attack simulation")
    print("  - POST /api/attack/stop - Stop attack simulation")
    print("\nWebSocket: ws://localhost:5000")
    print("Server running at http://localhost:5000")
    
    # Use socketio.run for WebSocket support
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)

