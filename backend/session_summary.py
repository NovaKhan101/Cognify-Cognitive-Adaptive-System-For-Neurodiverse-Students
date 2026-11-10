"""
session_summary.py
Cognify — LLM-Powered Session Summarizer
Generates a parent-friendly and teacher-friendly session summary.
Uses CraftX API (OpenAI-compatible) instead of Google Gemini SDK.
"""

import os
import json
import logging
import re
from datetime import datetime
from statistics import mean
from typing import Dict, Optional
import requests

from sqlalchemy.orm import Session
from models import TaskAttempt, SessionModel, Student

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONSTANTS & CRAFTX API SETUP
# ─────────────────────────────────────────────────────────
CRAFTX_API_KEY = os.getenv("CRAFTX_API_KEY")
CRAFTX_ENDPOINT = "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions"
GEMINI_MODEL = "Gemini 2.5 Flash"

def call_craftx(prompt: str) -> Optional[str]:
    """Helper function to call CraftX API."""
    if not CRAFTX_API_KEY:
        logger.warning("CRAFTX_API_KEY not found")
        return None
    try:
        response = requests.post(
            CRAFTX_ENDPOINT,
            headers={
                "Authorization": f"Bearer {CRAFTX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GEMINI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("choices"):
            logger.error("CraftX response missing choices: %s", data)
            return None
            
        return data["choices"][0].get("message", {}).get("content")
    except Exception as e:
        logger.error(f"CraftX API call failed: {e}")
        return None

TONE_BY_ACCURACY = {
    "high":   "celebratory and proud",    # accuracy >= 70%
    "medium": "encouraging and positive", # accuracy 40-70%
    "low":    "gentle and supportive"     # accuracy < 40%
}

PROFILE_DESCRIPTIONS = {
    "attention-variable":      "has difficulty sustaining attention",
    "working-memory-limited":  "has working memory challenges",
    "routine-seeking":         "prefers predictable routines",
    "sensory-hypersensitive":  "is sensitive to sensory stimulation",
    "unknown":                 "has ASD"
}

DOMAIN_FRIENDLY_NAMES = {
    "Attention":            "Focus & Attention",
    "Memory":               "Memory & Recall",
    "Language":             "Language & Communication",
    "Executive Function":   "Planning & Problem Solving",
    "Social Cognition":     "Social Understanding",
    "Sensory Processing":   "Sensory Awareness",
    "Motor Skills":         "Motor Skills",
    "Emotional Regulation": "Emotional Regulation"
}

# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────
def build_summary_prompt(session_data: Dict) -> str:
    acc = session_data.get("accuracy", 0)
    if acc >= 70: tone = TONE_BY_ACCURACY["high"]
    elif acc >= 40: tone = TONE_BY_ACCURACY["medium"]
    else: tone = TONE_BY_ACCURACY["low"]
    
    profile_desc = PROFILE_DESCRIPTIONS.get(session_data.get("cognitive_profile", "unknown"), PROFILE_DESCRIPTIONS["unknown"])
    
    # FIX #2: Sanitize student name to prevent prompt injection or weird formatting
    raw_name = str(session_data.get('student_name', 'The student'))
    safe_name = " ".join(raw_name.split())[:50]  # Remove newlines and cap length
    
    prompt = f"""
[ROLE]
You are a compassionate educational assistant helping parents and teachers
understand how a child with ASD performed in a learning session.
Write in simple, warm, non-clinical language.
A parent with no educational background should understand every word.
Never use jargon. Never be negative. Always be constructive.

[SESSION DATA]
Student Name: {safe_name}
Student Profile: {profile_desc}
Session Date: {session_data.get('session_date', 'Today')}
Total Tasks Completed: {session_data.get('total_attempts', 0)}
Overall Accuracy: {acc}%
Session Duration: {session_data.get('duration_mins', 0)} minutes
Domains Practiced: {', '.join(session_data.get('domains_practiced', ['N/A']))}
Frustration Incidents: {session_data.get('frustration_count', 0)}
Attention Drops: {session_data.get('attention_drops', 0)}
Average Fatigue Level: {session_data.get('fatigue_label', 'LOW')}
Recovery Domain Used: {'YES' if session_data.get('recovery_used') else 'NO'}
Best Performing Domain: {session_data.get('best_domain', 'N/A')} ({session_data.get('best_accuracy', 0)}%)
Weakest Domain: {session_data.get('weak_domain', 'N/A')} ({session_data.get('weak_accuracy', 0)}%)

[TONE]
Write in a {tone} tone.

[INSTRUCTIONS]
Generate exactly 3 bullet points:
1. Overall performance summary (1-2 sentences, mention accuracy naturally)
2. What went well / strongest moment (mention best domain)
3. Recommendation for parent/teacher (concrete, actionable, gentle)

Rules:
- Each bullet max 2 sentences
- Use the child's name
- Never say "failed" or "struggled" — say "found challenging" or "is working on"
- If frustration was high, mention the system helped them take a calming break
- If accuracy was high, celebrate it specifically
- End with one encouragement sentence

[OUTPUT FORMAT]
Return ONLY a JSON object. No markdown. No explanation.
{{
  "bullet_1": "...",
  "bullet_2": "...",
  "bullet_3": "...",
  "encouragement": "...",
  "teacher_note": "one brief observation for the teacher/therapist, still in plain language, no jargon"
}}
"""
    return prompt

# ─────────────────────────────────────────────────────────
# PARSING & FALLBACK
# ─────────────────────────────────────────────────────────
def parse_summary_response(raw_text: str) -> Optional[Dict]:
    try:
        cleaned = re.sub(r'```json|```', '', raw_text).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # NOTE: Intentionally using greedy \{.*\} to capture the full JSON object
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON Regex parse failed: {e}")
        logger.error(f"Failed to parse response: {raw_text}")
        return None

def fallback_summary(session_data: Dict) -> Dict:
    name = session_data.get("student_name", "The student")
    accuracy = session_data.get("accuracy", 0)
    best = session_data.get("best_domain", "N/A")
    
    bullet_1 = f"{name} completed {session_data.get('total_attempts', 0)} tasks today with {accuracy}% accuracy."
    bullet_2 = f"{name} showed strong performance in {best}." if best != "N/A" else f"{name} worked hard across multiple areas today."
    bullet_3 = "Continue practicing at the same difficulty level and celebrate small wins."
    encouragement = f"Great effort today, {name}! Every session is progress."
    teacher_note = f"Session completed with {accuracy}% accuracy. Review domain performance stats for details."

    return {
        "session_id": session_data.get("session_id"),
        "student_name": name,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "bullet_1": bullet_1,
            "bullet_2": bullet_2,
            "bullet_3": bullet_3,
            "encouragement": encouragement,
        },
        "teacher_note": teacher_note,
        "stats": {
            "accuracy": accuracy,
            "total_tasks": session_data.get("total_attempts", 0),
            "duration_mins": session_data.get("duration_mins", 0),
            "best_domain": best,
            "weak_domain": session_data.get("weak_domain", "N/A"),
            "fatigue_level": session_data.get("fatigue_label", "LOW"),
        },
        "source": "fallback"
    }

# ─────────────────────────────────────────────────────────
# DATABASE QUERY & ORCHESTRATION
# ─────────────────────────────────────────────────────────
def build_session_data(session_id: str, db: Session) -> Dict:
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not session:
        return {"error": "Session not found"}
        
    student = db.query(Student).filter(Student.student_id == session.student_id).first()
    attempts = db.query(TaskAttempt).filter(TaskAttempt.session_id == session_id).all()
    
    if not attempts:
        return {"error": "No attempts found for this session"}
        
    total_attempts = len(attempts)
    correct_count = sum(1 for a in attempts if a.correct)
    accuracy = round(correct_count / total_attempts * 100, 1) if total_attempts > 0 else 0
    frustration_count = sum(1 for a in attempts if a.frustration_flag)
    attention_drops = sum(1 for a in attempts if a.attention_drop_flag)
    
    fatigue_vals = [a.session_fatigue_index for a in attempts if a.session_fatigue_index is not None]
    avg_fatigue = mean(fatigue_vals) if fatigue_vals else 0.0
    
    if session.session_end and session.session_start:
        duration_mins = round((session.session_end - session.session_start).total_seconds() / 60, 1)
    else:
        duration_mins = 0.0
        
    domain_stats = {}
    for a in attempts:
        d = (a.domain_id or "unknown").strip()
        if d not in domain_stats:
            domain_stats[d] = {"attempts": 0, "correct": 0}
        domain_stats[d]["attempts"] += 1
        if a.correct:
            domain_stats[d]["correct"] += 1
            
    # FIX #3: Removed the >= 2 threshold so short demo sessions still calculate domains
    domain_accuracies = {
        d: s["correct"] / s["attempts"] * 100
        for d, s in domain_stats.items()
        if s["attempts"] > 0
    }
    
    # Prevent Strongest and Weakest from being the same if only 1 domain was played
    if len(domain_accuracies) > 1:
        best_domain = max(domain_accuracies, key=domain_accuracies.get)
        weak_domain = min(domain_accuracies, key=domain_accuracies.get)
    elif len(domain_accuracies) == 1:
        best_domain = list(domain_accuracies.keys())[0]
        weak_domain = "N/A (Single domain session)"
    else:
        best_domain = "N/A"
        weak_domain = "N/A"
        
    recovery_domains = ["Sensory Processing", "Emotional Regulation"]
    recovery_used = any((a.domain_id or "").strip() in recovery_domains for a in attempts)
    
    fatigue_label = "HIGH" if avg_fatigue > 0.65 else "MODERATE" if avg_fatigue > 0.35 else "LOW"
    
    # FIX #1: Safe fallback for session_start to prevent None.strftime() crash
    session_date_str = session.session_start.strftime("%B %d, %Y") if session.session_start else "Today"
    
    return {
        "session_id": session_id,
        "student_name": student.name if student and student.name else session.student_id,
        "cognitive_profile": student.cognitive_profile if student else "unknown",
        "asd_severity": student.asd_severity if student else "moderate",
        "session_date": session_date_str,
        "total_attempts": total_attempts,
        "accuracy": accuracy,
        "duration_mins": duration_mins,
        "frustration_count": frustration_count,
        "attention_drops": attention_drops,
        "avg_fatigue": avg_fatigue,
        "fatigue_label": fatigue_label,
        "recovery_used": recovery_used,
        "domains_practiced": list(domain_stats.keys()),
        "best_domain": DOMAIN_FRIENDLY_NAMES.get(best_domain, best_domain),
        "best_accuracy": round(domain_accuracies.get(best_domain, 0), 1),
        "weak_domain": DOMAIN_FRIENDLY_NAMES.get(weak_domain, weak_domain),
        "weak_accuracy": round(domain_accuracies.get(weak_domain, 0), 1),
        "domain_stats": domain_stats
    }

def generate_session_summary(session_id: str, db: Session) -> Dict:
    """Main entry point — orchestrates everything."""
    session_data = build_session_data(session_id, db)
    
    if "error" in session_data:
        return session_data
        
    if not CRAFTX_API_KEY:
        return fallback_summary(session_data)
        
    try:
        prompt = build_summary_prompt(session_data)
        raw = call_craftx(prompt)
        if not raw:
            return fallback_summary(session_data)
            
        summary = parse_summary_response(raw)
        
        if not summary:
            return fallback_summary(session_data)
            
        return {
            "session_id": session_id,
            "student_name": session_data["student_name"],
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "bullet_1": summary.get("bullet_1", ""),
                "bullet_2": summary.get("bullet_2", ""),
                "bullet_3": summary.get("bullet_3", ""),
                "encouragement": summary.get("encouragement", ""),
            },
            "teacher_note": summary.get("teacher_note", ""),
            "stats": {
                "accuracy": session_data["accuracy"],
                "total_tasks": session_data["total_attempts"],
                "duration_mins": session_data["duration_mins"],
                "best_domain": session_data["best_domain"],
                "weak_domain": session_data["weak_domain"],
                "fatigue_level": session_data["fatigue_label"],
            },
            "source": "gemini"
        }
        
    except Exception as e:
        logger.error(f"Error generating session summary: {e}")
        return fallback_summary(session_data)