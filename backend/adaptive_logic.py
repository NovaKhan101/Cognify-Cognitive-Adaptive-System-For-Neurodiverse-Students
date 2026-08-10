"""
adaptive_logic.py
Cognify — Pure Business Logic for Adaptive Learning
No DB calls, no ML calls, no API calls. Just deterministic rules.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
DOMAINS = [
    "Attention", "Memory", "Language", "Executive Function",
    "Social Cognition", "Sensory Processing", "Motor Skills", "Emotional Regulation"
]

DOMAIN_TASK_MAPPING = {
    "Attention":            list(range(0, 9)),
    "Memory":               list(range(9, 18)),
    "Language":             list(range(18, 27)),
    "Executive Function":   list(range(27, 36)),
    "Social Cognition":     list(range(36, 45)),
    "Sensory Processing":   list(range(45, 54)),
    "Motor Skills":         list(range(54, 63)),
    "Emotional Regulation": list(range(63, 71)),  # FIX 1: 8 tasks (63-70)
}

RECOVERY_DOMAINS = ["Sensory Processing", "Emotional Regulation"]
WORK_DOMAINS = [d for d in DOMAINS if d not in RECOVERY_DOMAINS]

# FIX 2: Sensory loads matched exactly to generate_asd_data.py
SENSORY_LOAD = {
    "Attention": 2,
    "Memory": 1,
    "Language": 1,
    "Executive Function": 2,
    "Social Cognition": 2,
    "Sensory Processing": 3,
    "Motor Skills": 1,
    "Emotional Regulation": 3
}

STEPS_PER_SESSION = 20
DIAGNOSTIC_TOTAL_TASKS = 16
DIAGNOSTIC_SEQUENCE = DOMAINS * 2  # 8 domains visited twice = 16 tasks


# ─────────────────────────────────────────────────────────
# SYSTEM STATE & DIAGNOSTICS
# ─────────────────────────────────────────────────────────
def get_system_mode(attempts_count: int) -> str:
    if attempts_count == 0:
        return "cold_start"
    elif 1 <= attempts_count <= 31:
        return "diagnostic"
    else:
        return "adaptive"

def get_diagnostic_domain(attempt_number: int) -> str:
    return DIAGNOSTIC_SEQUENCE[attempt_number % DIAGNOSTIC_TOTAL_TASKS]

def get_default_mastery(asd_severity: str) -> float:
    severity = (asd_severity or "").lower()
    if severity == "mild":
        return 0.60
    elif severity == "moderate":
        return 0.50
    elif severity == "severe":
        return 0.35
    return 0.50


# ─────────────────────────────────────────────────────────
# BEHAVIORAL TRIGGERS & GATING
# ─────────────────────────────────────────────────────────
def check_frustration_trigger(attempts_last_5: List[Dict]) -> bool:
    if len(attempts_last_5) < 3:
        return False
    
    frust_count = sum(1 for a in attempts_last_5 if a.get("frustration_flag"))
    wrong_count = sum(1 for a in attempts_last_5 if not a.get("correct"))
    
    return frust_count >= 3 or wrong_count >= 4

def check_frustration_trigger_recovery(attempts_last_5: List[Dict]) -> Tuple[bool, Optional[str]]:
    if not check_frustration_trigger(attempts_last_5):
        return False, None
    
    # Look at recent history to alternate recovery domains
    recent_domains = [a.get("domain_id") for a in attempts_last_5 if a.get("domain_id") in RECOVERY_DOMAINS]
    
    if recent_domains and recent_domains[-1] == "Sensory Processing":
        return True, "Emotional Regulation"
    elif recent_domains and recent_domains[-1] == "Emotional Regulation":
        return True, "Sensory Processing"
    else:
        return True, "Sensory Processing"

def should_advance_task(attempts_in_current_domain: List[Dict]) -> bool:
    if len(attempts_in_current_domain) < 2:
        return False
    return all(a.get("correct") for a in attempts_in_current_domain[-2:])

def should_end_session(attempts: List[Dict], asd_severity: str) -> Tuple[bool, Optional[str]]:
    if len(attempts) < 5:
        return False, None
    
    if len(attempts) >= STEPS_PER_SESSION:
        return True, "Session complete"
    
    severity = (asd_severity or "").lower()
    fatigue_threshold = 0.65 if severity == "severe" else 0.75
    
    last_3_fatigue = [float(a.get("session_fatigue_index", 0.0)) for a in attempts[-3:]]
    avg_fatigue = np.mean(last_3_fatigue) if last_3_fatigue else 0.0
    
    last_5_correct = [1 if a.get("correct") else 0 for a in attempts[-5:]]
    recent_accuracy = np.mean(last_5_correct) if last_5_correct else 0.0
    
    if avg_fatigue > fatigue_threshold and recent_accuracy < 0.4:
        return True, "Fatigue detected"
        
    return False, None

# FIX 3: Added consecutive_correct as an explicit parameter
def get_scaffolding_flag(last_attempt: Dict, consecutive_correct: int) -> Tuple[bool, int, Optional[str]]:
    if not last_attempt.get("correct", True):
        return True, -1, "Let's try something a little easier"
    elif last_attempt.get("correct") and consecutive_correct >= 2:
        return False, 1, "Ready to advance"
    return False, 0, None


# ─────────────────────────────────────────────────────────
# TASK SELECTION
# ─────────────────────────────────────────────────────────
def select_next_task_id(domain: str, difficulty: int, student_id: str, attempts_history: List[Dict]) -> int:
    domain_tasks = DOMAIN_TASK_MAPPING.get(domain, [0])
    
    # Filter by difficulty range (assuming 8-9 tasks per domain)
    if difficulty in [1, 2]:
        pool = domain_tasks[:3]
    elif difficulty == 3:
        pool = domain_tasks[3:6]
    elif difficulty in [4, 5]:
        pool = domain_tasks[6:9]
    else:
        pool = domain_tasks
        
    if not pool:
        return domain_tasks[0] if domain_tasks else 0
    
    # Remove recently failed tasks (last 10 attempts)
    recent_failed = {
        a.get("task_id") for a in attempts_history[-10:] 
        if a.get("domain_id") == domain and not a.get("correct")
    }
    available_pool = [t for t in pool if t not in recent_failed]
    
    if not available_pool:
        available_pool = pool  # Fallback if all were failed recently
        
    # Prefer unseen tasks
    seen_tasks = {
        a.get("task_id") for a in attempts_history 
        if a.get("domain_id") == domain
    }
    unseen_pool = [t for t in available_pool if t not in seen_tasks]
    
    final_pool = unseen_pool if unseen_pool else available_pool
    
    return random.choice(final_pool)


# ─────────────────────────────────────────────────────────
# UI ADAPTATION & ALERTS
# ─────────────────────────────────────────────────────────
def get_ui_adaptation_flags(student_context: Dict) -> Dict:
    avg_fatigue = float(student_context.get("avg_fatigue", 0.0))
    sensory_load = float(student_context.get("avg_sensory_load", 2.0))
    frustration_rate = float(student_context.get("frustration_rate", 0.0))
    cognitive_profile = student_context.get("cognitive_profile", "unknown")
    
    return {
        "reduce_visual_stimuli": avg_fatigue > 0.7 or sensory_load > 2,
        "hide_timer": avg_fatigue > 0.6,
        "encouragement_frequency": "high" if frustration_rate > 0.3 else "normal",
        "color_mode": "calm" if (avg_fatigue > 0.65 or sensory_load > 2) else "normal",
        "simplify_layout": cognitive_profile == "working-memory-limited" and avg_fatigue > 0.5
    }

def check_meltdown_alert(attempts_last_5: List[Dict]) -> Tuple[bool, Optional[str], Optional[str]]:
    if len(attempts_last_5) < 4:
        return False, None, None
        
    wrong_count = sum(1 for a in attempts_last_5 if not a.get("correct"))
    frust_count = sum(1 for a in attempts_last_5 if a.get("frustration_flag"))
    
    rts = [float(a.get("response_time_ms", 0.0)) for a in attempts_last_5]
    last_rt = rts[-1]
    avg_prev_rt = np.mean(rts[:-1]) if rts[:-1] else 0.0
    
    rt_increasing = avg_prev_rt > 0 and last_rt > (avg_prev_rt * 1.3)
    
    if wrong_count >= 3 and frust_count >= 2:
        return True, "HIGH", "Student showing severe frustration and repeated errors. Consider 1:1 intervention."
    elif wrong_count >= 2 and rt_increasing:
        return True, "MEDIUM", "Student is struggling and slowing down. Consider offering a hint or a break."
        
    return False, None, None


# ─────────────────────────────────────────────────────────
# ANALYTICS & PROGRESS
# ─────────────────────────────────────────────────────────
def get_progress_delta(current_session_stats: Dict[str, float], previous_session_stats: Dict[str, float]) -> Dict:
    deltas = {}
    all_domains = set(current_session_stats.keys()).union(set(previous_session_stats.keys()))
    
    for d in all_domains:
        curr = float(current_session_stats.get(d, 0.0))
        prev = float(previous_session_stats.get(d, 0.0))
        delta = curr - prev
        
        if delta > 5:
            trend = "improving"
        elif delta < -5:
            trend = "declining"
        else:
            trend = "stable"
            
        deltas[d] = {
            "current": curr,
            "previous": prev,
            "delta": round(delta, 2),
            "trend": trend
        }
        
    return deltas