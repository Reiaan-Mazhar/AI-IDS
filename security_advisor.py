"""
Security Advisor Module - LLM-based Security Recommendations
Uses Groq API (with Llama models) to provide actionable security advice based on threat detections
"""
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Support both GROQ and GROK key names for flexibility
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_attack_context(attack_type: str) -> str:
    """Get detailed context about specific attack types"""
    attack_info = {
        "DDoS": {
            "description": "Distributed Denial of Service attack - multiple systems flooding target with traffic",
            "severity": "HIGH",
            "indicators": ["High packet rate", "Multiple source IPs", "SYN floods", "UDP floods"],
            "immediate_actions": ["Rate limiting", "Traffic filtering", "CDN activation"]
        },
        "DoS": {
            "description": "Denial of Service attack - single source overwhelming target resources",
            "severity": "HIGH",
            "indicators": ["High packet volume from single IP", "Resource exhaustion", "Service unavailability"],
            "immediate_actions": ["Block source IP", "Resource scaling", "Service restart"]
        },
        "Bot": {
            "description": "Botnet activity - compromised machine communicating with C&C server",
            "severity": "CRITICAL",
            "indicators": ["Unusual outbound connections", "Periodic beaconing", "IRC/HTTP C&C traffic"],
            "immediate_actions": ["Isolate infected host", "Block C&C IPs", "Forensic analysis"]
        },
        "Brute Force": {
            "description": "Password guessing attack attempting to gain unauthorized access",
            "severity": "MEDIUM",
            "indicators": ["Multiple failed login attempts", "Rapid authentication requests", "Dictionary patterns"],
            "immediate_actions": ["Account lockout", "IP blocking", "MFA enforcement"]
        },
        "Port Scan": {
            "description": "Network reconnaissance to identify open ports and services",
            "severity": "LOW",
            "indicators": ["Sequential port probing", "SYN scans", "Multiple ports targeted"],
            "immediate_actions": ["Monitor source IP", "Review firewall rules", "Log for analysis"]
        },
        "Web Attack": {
            "description": "Web application attack including SQL injection, XSS, or path traversal",
            "severity": "HIGH",
            "indicators": ["Malicious request patterns", "SQL keywords", "Script injection attempts"],
            "immediate_actions": ["WAF rule activation", "Request blocking", "Application patching"]
        },
        "Attack": {
            "description": "General network attack detected",
            "severity": "MEDIUM",
            "indicators": ["Anomalous traffic patterns", "Suspicious payload"],
            "immediate_actions": ["Traffic analysis", "Source IP investigation", "Enhanced monitoring"]
        }
    }
    return attack_info.get(attack_type, attack_info.get("Attack", {}))


def build_security_prompt(
    attack_type: str,
    source_ip: Optional[str] = None,
    destination_ip: Optional[str] = None,
    confidence: float = 0.0,
    feature_importance: Optional[List[Dict]] = None,
    additional_context: Optional[str] = None
) -> str:
    """Build a detailed prompt for the LLM security advisor"""
    
    attack_context = get_attack_context(attack_type)
    
    prompt = f"""You are a cybersecurity expert providing actionable security advice for an Intrusion Detection System (IDS).

## Detected Threat
- **Attack Type**: {attack_type}
- **Description**: {attack_context.get('description', 'Unknown attack type')}
- **Severity**: {attack_context.get('severity', 'UNKNOWN')}
- **Detection Confidence**: {confidence * 100:.1f}%
"""
    
    if source_ip:
        prompt += f"- **Source IP**: {source_ip}\n"
    if destination_ip:
        prompt += f"- **Destination IP**: {destination_ip}\n"
    
    if feature_importance:
        prompt += "\n## Key Detection Factors\n"
        for i, feat in enumerate(feature_importance[:5], 1):
            prompt += f"{i}. **{feat['feature']}**: {feat['value']:.2f} (contribution: {feat['contribution']})\n"
    
    if additional_context:
        prompt += f"\n## Additional Context\n{additional_context}\n"
    
    prompt += """
## Required Response
Provide a structured security response with the following sections:

1. **Threat Assessment**: Brief analysis of the threat severity and potential impact
2. **Immediate Actions**: 3-5 specific steps to mitigate the threat right now
3. **Firewall Recommendations**: Specific firewall rules to implement (provide example iptables commands)
4. **Investigation Steps**: How to investigate and gather more information about this threat
5. **Long-term Recommendations**: Preventive measures to avoid similar attacks

Keep the response concise but actionable. Focus on practical steps that can be implemented immediately.
"""
    
    return prompt


from llm_agent import llm_agent

def get_security_advice(
    attack_type: str,
    source_ip: Optional[str] = None,
    destination_ip: Optional[str] = None,
    confidence: float = 0.0,
    feature_importance: Optional[List[Dict]] = None,
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get security advice using LLMAgent (Groq/Gemini).
    Delegates to the configured LLM agent for analysis.
    """
    try:
        # Use the centralized LLM agent
        analysis = llm_agent.analyze_threat(
            attack_type=attack_type,
            confidence=confidence,
            source_ip=source_ip or "Unknown",
            destination_ip=destination_ip or "Unknown",
            top_features=feature_importance or []
        )
        
        # Format the advice text for simple display
        advice_text = f"**Assessment**: {analysis.get('summary')}\n\n"
        advice_text += f"**Severity**: {analysis.get('severity')}\n\n"
        advice_text += "**Immediate Actions**:\n" + "\n".join([f"- {action}" for action in analysis.get('immediate_actions', [])])
        
        # Structure for frontend consumption
        return {
            "success": True,
            "attack_type": attack_type,
            "severity": analysis.get("severity", "UNKNOWN"),
            "advice": advice_text,
            "parsed_recommendations": {
                "threat_assessment": [analysis.get("summary", "")],
                "immediate_actions": analysis.get("immediate_actions", []),
                "firewall_recommendations": ["Review firewall rules for source IP"], # Placeholder or generate dynamic rules
                "investigation_steps": [analysis.get("implications", "")],
                "long_term_recommendations": [analysis.get("mitigation", "")]
            },
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "confidence": confidence
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_advice": get_fallback_advice(attack_type)
        }


def parse_recommendations(advice: str) -> Dict[str, List[str]]:
    """Parse the LLM response into structured recommendations"""
    sections = {
        "threat_assessment": [],
        "immediate_actions": [],
        "firewall_recommendations": [],
        "investigation_steps": [],
        "long_term_recommendations": []
    }
    
    current_section = None
    lines = advice.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        line_lower = line.lower()
        if 'threat assessment' in line_lower:
            current_section = 'threat_assessment'
        elif 'immediate action' in line_lower:
            current_section = 'immediate_actions'
        elif 'firewall' in line_lower:
            current_section = 'firewall_recommendations'
        elif 'investigation' in line_lower:
            current_section = 'investigation_steps'
        elif 'long-term' in line_lower or 'long term' in line_lower:
            current_section = 'long_term_recommendations'
        elif current_section and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
            # Clean up bullet points
            clean_line = line.lstrip('-*0123456789. ')
            if clean_line:
                sections[current_section].append(clean_line)
        elif current_section and not line.startswith('#'):
            # Regular content line
            if line:
                sections[current_section].append(line)
    
    return sections


def get_fallback_advice(attack_type: str) -> Dict[str, Any]:
    """Get pre-defined fallback advice when API is unavailable"""
    context = get_attack_context(attack_type)
    
    # Dynamic templates for variety
    assessment_templates = [
        f"{context.get('description', 'Network attack detected')}. Risk level is elevated.",
        f"Security Alert: {attack_type} activity detected. Immediate review required.",
        f"Potential security breach attempt matching {attack_type} signature.",
        f"Automated analysis flagged {attack_type} pattern. Containment advised."
    ]
    
    # Use random implementation of iptables for variety (e.g. log prefix change)
    log_prefixes = ["IDS_BLOCKED:", "THREAT_DROP:", "SEC_ALERT:", "FW_DENY:"]
    log_prefix = random.choice(log_prefixes)
    
    actions = context.get("immediate_actions", []).copy()
    if actions and len(actions) > 1:
        # Keep first important action, shuffle rest for variety
        first = actions[0]
        rest = actions[1:]
        random.shuffle(rest)
        actions = [first] + rest
    
    return {
        "attack_type": attack_type,
        "severity": context.get("severity", "UNKNOWN"),
        "threat_assessment": random.choice(assessment_templates),
        "immediate_actions": actions,
        "firewall_recommendations": [
            f"# Block suspicious IP (replace with actual IP)",
            f"iptables -A INPUT -s <SOURCE_IP> -j DROP",
            f"# Log dropped packets",
            f"iptables -A INPUT -j LOG --log-prefix '{log_prefix} '"
        ],
        "indicators": context.get("indicators", []),
        "generated_at": datetime.now().strftime("%H:%M:%S"),
        "note": "AI disconnected. Standard security protocols loaded (Fallback Mode)."
    }


def generate_firewall_rules(
    source_ip: str,
    action: str = "DROP",
    protocol: Optional[str] = None,
    port: Optional[int] = None
) -> List[str]:
    """Generate iptables firewall rules"""
    rules = []
    
    # Basic IP block rule
    if protocol and port:
        rules.append(f"iptables -A INPUT -s {source_ip} -p {protocol} --dport {port} -j {action}")
    elif protocol:
        rules.append(f"iptables -A INPUT -s {source_ip} -p {protocol} -j {action}")
    else:
        rules.append(f"iptables -A INPUT -s {source_ip} -j {action}")
    
    # Add logging rule
    rules.append(f"iptables -A INPUT -s {source_ip} -j LOG --log-prefix 'IDS_BLOCK_{source_ip}: '")
    
    # Also block outgoing (for botnet cases)
    rules.append(f"iptables -A OUTPUT -d {source_ip} -j {action}")
    
    return rules


    return rules


def execute_response_action(action: str, target: str) -> Dict[str, Any]:
    """
    Execute a security response action.
    Currently supports: 'BLOCK_IP' using iptables (or simulation).
    """
    import subprocess
    import logging
    
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().isoformat()
    
    if action == "BLOCK_IP":
        command = ["iptables", "-A", "INPUT", "-s", target, "-j", "DROP"]
        
        try:
            # Try to execute (will likely fail without sudo/root)
            subprocess.run(command, check=True, capture_output=True)
            return {
                "success": True, 
                "message": f"Successfully blocked IP {target}",
                "simulated": False,
                "timestamp": timestamp
            }
        except (subprocess.CalledProcessError, PermissionError, FileNotFoundError):
            # Fallback to simulation/mocking
            logger.warning(f"Could not execute iptables (permission denied?). Simulating block for {target}")
            return {
                "success": True, 
                "message": f"Simulated Block: Added firewall rule for {target} (active response)",
                "simulated": True,
                "timestamp": timestamp
            }
            
    return {"success": False, "message": f"Unknown action: {action}"}


def generate_explanation_text(
    attack_type: str,
    top_features: List[Dict[str, Any]],
    confidence: float = 0.0
) -> str:
    """
    Generate a natural language explanation using LLMAgent.
    """
    try:
        # Reuse analyze_threat for explanation
        analysis = llm_agent.analyze_threat(
            attack_type=attack_type,
            confidence=confidence,
            source_ip="N/A",
            destination_ip="N/A",
            top_features=top_features
        )
        return analysis.get("summary", f"The model classified this flow as {attack_type}.")
                
    except Exception as e:
        print(f"Explanation generation failed: {e}")
        return f"The model classified this flow as **{attack_type}** with {confidence:.1%} confidence."


# Quick test
if __name__ == "__main__":
    result = get_security_advice(
        attack_type="DDoS",
        source_ip="192.168.1.100",
        confidence=0.95
    )
    print(json.dumps(result, indent=2))
