
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

# Import analysis modules
import numpy as np

# Import analysis modules
from shap_explainer import shap_explainer
from security_advisor import get_security_advice, generate_explanation_text

class ReportGenerator:
    """
    Generates comprehensive threat analysis reports for detected attacks.
    """
    
    def __init__(self, output_dir: str = "static/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, 
                       flow_data: dict, 
                       prediction: dict, 
                       features: dict, 
                       dataset: str, 
                       model_id: str,
                       report_id: str = None) -> dict:
        """
        Generate a threat report, save it, and return metadata.
        """
        if report_id is None:
            report_id = str(uuid.uuid4())
            
        timestamp = datetime.now().isoformat()
        
        # 1. Prepare Features for XAI
        # Convert feature dict to list/array expected by SHAP
        # (Assuming shap_explainer can handle dict or we convert it here)
        # Using app.py logic logic for consistency
        feature_names = shap_explainer.get_feature_names(dataset)
        ordered_features = [float(features.get(name, 0.0)) for name in feature_names]
        X = np.array([ordered_features], dtype=np.float32)
        
        # 2. Generate SHAP Explanation
        try:
            explanation = shap_explainer.explain_prediction(
                dataset=dataset,
                model_id=model_id,
                X=X,
                sample_index=0
            )
        except Exception as e:
            explanation = {"error": str(e), "top_features": []}
            
        # 3. Generate Natural Language Explanation
        text_explanation = "Analysis unavailable."
        if explanation.get("top_features"):
            text_explanation = generate_explanation_text(
                attack_type=prediction.get('label', 'Unknown'),
                top_features=explanation['top_features'],
                confidence=prediction.get('confidence', 0.0)
            )
            explanation['text_explanation'] = text_explanation
            
        # 4. Get Security Advice
        security_advice = get_security_advice(
            attack_type=prediction.get('label', 'Unknown'),
            source_ip=flow_data.get('src_ip'),
            destination_ip=flow_data.get('dst_ip'),
            confidence=prediction.get('confidence', 0.0),
            feature_importance=explanation.get('top_features'),
            additional_context=text_explanation
        )
        
        # 5. Compile Report
        report = {
            "report_id": report_id,
            "timestamp": timestamp,
            "classification": {
                "label": prediction.get('label'),
                "confidence": prediction.get('confidence'),
                "is_threat": prediction.get('is_threat')
            },
            "flow_details": flow_data,
            "analysis": {
                "explanation_text": text_explanation,
                "top_features": explanation.get('top_features', []),
                "waterfall_plot": explanation.get('waterfall_plot') # Base64 image
            },
            "security_advice": security_advice
        }
        
        # 6. Save Report
        filename = f"report_{report_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        return {
            "report_id": report_id,
            "report_url": f"/static/reports/{filename}",
            "generated_at": timestamp
        }

# Global instance
report_generator = ReportGenerator()
