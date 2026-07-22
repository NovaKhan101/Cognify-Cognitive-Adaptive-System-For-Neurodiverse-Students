# """
# generate_asd_data_v3.py
# Cognify — ASD Adaptive Learning Platform
# North South University Bangladesh — Capstone 2026

# Behavioral data generator with clinically-grounded cognitive profiles:
#   Profile: attention-variable      → mild severity, attention dysregulation
#   Profile: working-memory-limited  → moderate severity, WM deficits
#   Profile: routine-seeking         → severe severity, insistence on sameness

# Rules:
#   R1 — Difficulty progression tied to correct + frustration_flag
#   R2 — Task mastery gating (2x consecutive correct to advance)
#   R3 — Temporary domain switch on rolling frustration > 0.65, then return
#   R4 — Attention drop flag (independent from frustration, tracks focus loss)
#   R5 — Session fatigue index (increases per step, faster for severe profiles)
#   R6 — Routine-seeking penalty (+0.25 frustration prob for 3 steps after switch)
#   R7 — Perceived difficulty (profile-adjusted, capped at 5)
#   R8 — Forced domain rotation every 8 steps (therapist intervention model)

# Output: cognify_synthetic_data_v3.csv
# """

# import random
# import numpy as np
# import pandas as pd
# from collections import deque

# # ── Constants ──────────────────────────────────────────────────────────
# DOMAINS = [
#     "Attention", "Memory", "Language", "Executive Function",
#     "Social Cognition", "Sensory Processing", "Motor Skills", "Emotional Regulation"
# ]

# DOMAIN_TASK_MAPPING = {
#     "Attention":            list(range(0, 9)),
#     "Memory":               list(range(9, 18)),
#     "Language":             list(range(18, 27)),
#     "Executive Function":   list(range(27, 36)),
#     "Social Cognition":     list(range(36, 45)),
#     "Sensory Processing":   list(range(45, 54)),
#     "Motor Skills":         list(range(54, 63)),
#     "Emotional Regulation": list(range(63, 71)),
# }

# RECOVERY_DOMAINS = ["Sensory Processing", "Emotional Regulation"]
# WORK_DOMAINS     = [d for d in DOMAINS if d not in RECOVERY_DOMAINS]

# SENSORY_LOAD = {
#     "Sensory Processing":    3,
#     "Emotional Regulation":  3,
#     "Social Cognition":      2,
#     "Attention":             2,
#     "Executive Function":    2,
#     "Language":              1,
#     "Memory":                1,
#     "Motor Skills":          1,
# }

# # 3 clinically-grounded cognitive profiles
# PROFILES = {
#     "attention-variable": {
#         "severity": "mild",
#         "base_diff": 3,
#         "frustration_rate": 0.10,
#         "engagement_base": 0.82,
#         "attention_drop_boost": 0.18,
#         "fatigue_rate": 1.0,          # baseline
#         "wm_penalty": False,
#         "routine_penalty": False,
#     },
#     "working-memory-limited": {
#         "severity": "moderate",
#         "base_diff": 2,
#         "frustration_rate": 0.20,
#         "engagement_base": 0.65,
#         "attention_drop_boost": 0.08,
#         "fatigue_rate": 1.3,
#         "wm_penalty": True,           # perceived diff +1, slower RT
#         "routine_penalty": False,
#     },
#     "routine-seeking": {
#         "severity": "severe",
#         "base_diff": 1,
#         "frustration_rate": 0.28,
#         "engagement_base": 0.52,
#         "attention_drop_boost": 0.05,
#         "fatigue_rate": 2.0,          # 2x faster fatigue
#         "wm_penalty": False,
#         "routine_penalty": True,      # frustration on domain switch
#     },
# }

# NUM_STUDENTS         = 300
# SESSIONS_PER_STUDENT = 10
# STEPS_PER_SESSION    = 20
# RANDOM_SEED          = 42

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# # ── Helpers ────────────────────────────────────────────────────────────

# def compute_perceived_difficulty(designed_diff, profile, attention_drop_flag, domain_switch_count):
#     """R7: Perceived difficulty differs from designed based on profile.

#     WM-limited: always +1 (capped at 5).
#     WM-limited + attention_drop: +1 additional (total +2, still capped at 5).
#     Routine-seeking + domain_switch_count > 1: +1.
#     """
#     pdiff = designed_diff
#     p = PROFILES[profile]

#     if p["wm_penalty"]:
#         pdiff += 1
#     # Double penalty: WM-limited with attention drop gets +2 total
#     if p["wm_penalty"] and attention_drop_flag == 1:
#         pdiff += 1
#     if p["routine_penalty"] and domain_switch_count > 1:
#         pdiff += 1

#     return min(5, pdiff)

# def simulate_step(student_id, session_id, sequence_step,
#                   domain, task_id, difficulty, profile,
#                   consecutive_correct, session_num,
#                   fatigue_index, domain_switch_count,
#                   steps_since_switch):
#     p = PROFILES[profile]

#     # R5: Cross-session fatigue multiplier (separate from within-session fatigue_index)
#     cross_session_fatigue = 1.0 + (session_num / SESSIONS_PER_STUDENT) * 0.20
#     fatigue_index = min(1.0, fatigue_index)

#     # Base correct probability adjusted by difficulty and fatigue
#     base_correct_prob = max(0.15, min(0.92, 0.5 + (3 - difficulty) * 0.12))
#     # Fatigue reduces correct probability
#     fatigue_penalty = fatigue_index * 0.15
#     correct_prob = max(0.05, base_correct_prob - fatigue_penalty)
#     correct = int(random.random() < correct_prob)

#     # Response time: WM-limited is slower; fatigue increases RT
#     if correct:
#         resp_ms = np.random.normal(4000 - difficulty * 400, 800)
#     else:
#         resp_ms = np.random.normal(8000 + difficulty * 500, 1500)

#     # WM penalty: slower by 1500ms on average
#     if p["wm_penalty"]:
#         resp_ms += 1500
#     # Fatigue increases RT
#     resp_ms += fatigue_index * 2000
#     resp_ms = max(500.0, min(60000.0, resp_ms))

#     hints_used = 0 if correct else int(np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25]))
#     # WM-limited uses more hints
#     if p["wm_penalty"] and not correct:
#         hints_used = min(3, hints_used + 1)

#     # Frustration calculation
#     base_frust = p["frustration_rate"] * cross_session_fatigue
#     if not correct:       base_frust += 0.18
#     if hints_used >= 2:   base_frust += 0.12

#     # R6: Routine-seeking penalty on domain switch (first 3 steps after switch)
#     if p["routine_penalty"] and steps_since_switch <= 3 and domain_switch_count > 0:
#         base_frust += 0.25

#     frustration_flag = int(random.random() < min(base_frust, 0.85))

#     # R4: Attention drop flag
#     base_attn_drop = 0.08
#     if not correct:           base_attn_drop += 0.15
#     if hints_used >= 2:       base_attn_drop += 0.10
#     if frustration_flag == 1: base_attn_drop += 0.12
#     if domain == "Attention": base_attn_drop += 0.05
#     base_attn_drop += p["attention_drop_boost"]
#     attention_drop_flag = int(random.random() < min(base_attn_drop, 0.70))

#     # Engagement score
#     engagement = (
#         p["engagement_base"]
#         - frustration_flag * 0.12
#         - attention_drop_flag * 0.08
#         + consecutive_correct * 0.03
#     )
#     engagement = max(0.1, min(1.0, engagement + np.random.normal(0, 0.04)))

#     # R7: Perceived difficulty
#     perceived_difficulty = compute_perceived_difficulty(
#         difficulty, profile, attention_drop_flag, domain_switch_count
#     )

#     # 1-indexed: values run 0.05 → 1.0, never 0.0
#     session_position_normalized = sequence_step / STEPS_PER_SESSION

#     return {
#         "student_id":                  student_id,
#         "session_id":                  session_id,
#         "sequence_step":               sequence_step,
#         "domain_id":                   domain,
#         "task_id":                     task_id,
#         "difficulty":                  difficulty,
#         "correct":                     correct,
#         "response_time_ms":            round(resp_ms, 1),
#         "hints_used":                  hints_used,
#         "frustration_flag":            frustration_flag,
#         "attention_drop_flag":         attention_drop_flag,
#         "engagement_score":            round(engagement, 3),
#         # ── NEW v3 columns ───────────────────────────────────────────
#         "cognitive_profile":           profile,
#         "session_fatigue_index":       round(fatigue_index, 3),
#         "perceived_difficulty":        perceived_difficulty,
#         "sensory_load":                SENSORY_LOAD[domain],
#         "domain_switch_count":         domain_switch_count,
#         "session_position_normalized": round(session_position_normalized, 3),
#     }

# def generate_session(student_id, session_num, profile, start_domain):
#     session_id = f"{student_id}_S{session_num:02d}"
#     rows = []
#     p = PROFILES[profile]

#     domain              = start_domain
#     difficulty          = p["base_diff"]
#     task_id             = random.choice(DOMAIN_TASK_MAPPING[domain])
#     consecutive_correct = 0
#     frustration_window  = deque(maxlen=3)

#     # Recovery state tracking
#     in_recovery         = False
#     recovery_steps_left = 0
#     pre_recovery_domain = None
#     pre_recovery_task   = None
#     recovery_switches   = 0

#     # Forced rotation counter
#     steps_since_domain_switch = 0
#     MAX_STEPS_PER_DOMAIN = 8

#     # v3: fatigue and domain-switch tracking
#     fatigue_index = 0.0
#     domain_switch_count = 0
#     steps_since_switch = 0  # guard (domain_switch_count > 0) prevents penalty at start

#     for step in range(STEPS_PER_SESSION):
#         sequence_step = step + 1

#         # R5: fatigue increases each step, rate depends on profile severity
#         fatigue_increment = (1.0 / STEPS_PER_SESSION) * p["fatigue_rate"]
#         fatigue_index = min(1.0, fatigue_index + fatigue_increment)

#         row = simulate_step(
#             student_id, session_id, sequence_step,
#             domain, task_id, difficulty, profile,
#             consecutive_correct, session_num,
#             fatigue_index, domain_switch_count,
#             steps_since_switch
#         )
#         rows.append(row)

#         correct      = row["correct"]
#         frustration  = row["frustration_flag"]
#         hints        = row["hints_used"]

#         frustration_window.append(frustration)
#         steps_since_domain_switch += 1
#         steps_since_switch += 1

#         # ── Return from recovery after cooldown ──────────────────────
#         if in_recovery:
#             recovery_steps_left -= 1
#             if recovery_steps_left <= 0:
#                 in_recovery = False
#                 domain   = pre_recovery_domain
#                 task_id  = pre_recovery_task
#                 difficulty = max(1, difficulty)
#                 frustration_window.clear()
#                 steps_since_domain_switch = 0
#                 steps_since_switch = 0
#                 domain_switch_count += 1
#             if correct and not frustration:
#                 consecutive_correct += 1
#             else:
#                 consecutive_correct = 0
#             continue

#         # ── R3: Trigger recovery on rolling frustration ─────────────
#         rolling_frust = np.mean(frustration_window) if len(frustration_window) == 3 else 0
#         if rolling_frust > 0.65 and recovery_switches < 2:
#             pre_recovery_domain = domain
#             pre_recovery_task   = task_id
#             domain              = random.choice(RECOVERY_DOMAINS)
#             task_id             = random.choice(DOMAIN_TASK_MAPPING[domain])
#             difficulty          = max(1, difficulty - 1)
#             consecutive_correct = 0
#             in_recovery         = True
#             recovery_steps_left = 4
#             recovery_switches  += 1
#             frustration_window.clear()
#             steps_since_domain_switch = 0
#             steps_since_switch = 0
#             domain_switch_count += 1
#             continue

#         # ── R8: Forced domain rotation (therapist intervention) ──────
#         if steps_since_domain_switch >= MAX_STEPS_PER_DOMAIN:
#             other_domains = [d for d in WORK_DOMAINS if d != domain]
#             if other_domains:
#                 domain = random.choice(other_domains)
#                 task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
#                 consecutive_correct = 0
#                 steps_since_domain_switch = 0
#                 steps_since_switch = 0
#                 domain_switch_count += 1
#                 continue

#         # ── R2: Mastery gating — advance task after 2x correct ──────
#         if correct and not frustration:
#             consecutive_correct += 1
#         else:
#             consecutive_correct = 0

#         if consecutive_correct >= 2:
#             current_tasks = DOMAIN_TASK_MAPPING[domain]
#             pos = current_tasks.index(task_id) if task_id in current_tasks else 0
#             if pos + 1 < len(current_tasks):
#                 task_id = current_tasks[pos + 1]
#             else:
#                 # Domain mastered — move to next work domain
#                 domain  = random.choice([d for d in WORK_DOMAINS if d != domain])
#                 task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
#                 steps_since_domain_switch = 0
#                 steps_since_switch = 0
#                 domain_switch_count += 1
#             consecutive_correct = 0
#             continue

#         # ── R1: Difficulty tied to performance ─────────────────────
#         if correct and not frustration and hints == 0:
#             difficulty = min(5, difficulty + 1)
#         elif not correct or hints >= 2:
#             difficulty = max(1, difficulty - 1)
#             current_tasks = DOMAIN_TASK_MAPPING[domain]
#             pos = current_tasks.index(task_id) if task_id in current_tasks else 0
#             if pos > 0 and not correct:
#                 task_id = current_tasks[pos - 1]

#     return rows

# # ── Main ───────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=" * 60)
#     print("COGNIFY — Behavioral Data Generator V3")
#     print("Cognitive Profiles: attention-variable | working-memory-limited | routine-seeking")
#     print("=" * 60)

#     all_rows = []
#     # 40% attention-variable, 40% WM-limited, 20% routine-seeking
#     profile_cycle = (
#         ["attention-variable"] * 40
#         + ["working-memory-limited"] * 40
#         + ["routine-seeking"] * 20
#     )

#     for s in range(NUM_STUDENTS):
#         student_id = f"STU_{s:04d}"
#         profile    = profile_cycle[s % len(profile_cycle)]
#         for sess in range(SESSIONS_PER_STUDENT):
#             session_domain = WORK_DOMAINS[(s + sess) % len(WORK_DOMAINS)]
#             rows = generate_session(student_id, sess + 1, profile, session_domain)
#             all_rows.extend(rows)

#     df = pd.DataFrame(all_rows)

#     # ── Verification Stats ─────────────────────────────────────────
#     print(f"\nTotal rows              : {len(df):,}")
#     print(f"Students                : {df['student_id'].nunique()}")
#     print(f"Sessions                : {df['session_id'].nunique()}")
#     print(f"\nDomain distribution     :\n{df['domain_id'].value_counts().to_string()}")
#     print(f"\nDifficulty distribution :\n{df['difficulty'].value_counts().sort_index().to_string()}")
#     print(f"\nSensory load distribution :\n{df['sensory_load'].value_counts().sort_index().to_string()}")
#     print(f"\nCognitive profile dist  :\n{df['cognitive_profile'].value_counts().to_string()}")

#     print(f"\n--- Per-Profile Averages ---")
#     for prof in PROFILES.keys():
#         sub = df[df['cognitive_profile'] == prof]
#         print(f"\n{prof}:")
#         print(f"  Avg correct           : {sub['correct'].mean():.3f}")
#         print(f"  Avg frustration       : {sub['frustration_flag'].mean():.3f}")
#         print(f"  Avg attention drop    : {sub['attention_drop_flag'].mean():.3f}")
#         print(f"  Avg fatigue index     : {sub['session_fatigue_index'].mean():.3f}")
#         print(f"  Avg perceived diff    : {sub['perceived_difficulty'].mean():.2f}")
#         print(f"  Avg designed diff     : {sub['difficulty'].mean():.2f}")
#         print(f"  Avg response time (ms): {sub['response_time_ms'].mean():.0f}")
#         print(f"  Avg hints used        : {sub['hints_used'].mean():.2f}")

#     print(f"\n--- Perceived vs Designed Difficulty Correlation ---")
#     corr = df['perceived_difficulty'].corr(df['difficulty'])
#     print(f"  Pearson r = {corr:.3f}")
#     print(f"  (Lower r = more profile-driven divergence)")

#     out = "cognify_synthetic_data_v3.csv"
#     df.to_csv(out, index=False)
#     print(f"\n✅ Saved → {out} ({len(df):,} rows, {df.shape[1]} columns)")

"""
generate_asd_data_v4.py
Cognify — ASD Adaptive Learning Platform
North South University Bangladesh — Capstone 2026

V4 CHANGES:
  1. SESSIONS_PER_STUDENT = 15
  2. Per-student Gaussian noise on profile parameters
  3. 4th profile: sensory-hypersensitive
  4. Structured domain transitions via curriculum matrix
  5. Sensory load now actively affects correctness
  6. [FIX] Added domain mastery state & recent history for DKT learnability

Rules R1–R8 preserved exactly.
Output: cognify_synthetic_data.csv
"""

import random
import numpy as np
import pandas as pd
from collections import deque

# ── Constants ──────────────────────────────────────────────────────────
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
    "Emotional Regulation": list(range(63, 71)),
}

RECOVERY_DOMAINS = ["Sensory Processing", "Emotional Regulation"]
WORK_DOMAINS = [d for d in DOMAINS if d not in RECOVERY_DOMAINS]

SENSORY_LOAD = {
    "Sensory Processing":    3,
    "Emotional Regulation":  3,
    "Social Cognition":      2,
    "Attention":             2,
    "Executive Function":    2,
    "Language":              1,
    "Memory":                1,
    "Motor Skills":          1,
}

# ── Curriculum transition matrix for WORK domains ────────────────────
WORK_TRANSITION_MATRIX = np.array([
    # Attention  Memory  Language  Exec     Social   Motor
    [0.35,      0.15,   0.15,     0.15,    0.10,    0.10],
    [0.15,      0.35,   0.15,     0.10,    0.10,    0.15],
    [0.15,      0.15,   0.35,     0.15,    0.10,    0.10],
    [0.15,      0.10,   0.15,     0.35,    0.15,    0.10],
    [0.10,      0.10,   0.10,     0.15,    0.40,    0.15],
    [0.10,      0.15,   0.10,     0.10,    0.15,    0.40],
])

# ── Profile templates with noise std ─────────────────────────────────
PROFILE_TEMPLATES = {
    "attention-variable": {
        "severity": "mild",
        "base_diff": 3,
        "frustration_rate": 0.10,
        "engagement_base": 0.82,
        "attention_drop_boost": 0.18,
        "fatigue_rate": 1.0,
        "wm_penalty": False,
        "routine_penalty": False,
        "sensory_sensitivity": 0.05,
        "noise_std": 0.04,
    },
    "working-memory-limited": {
        "severity": "moderate",
        "base_diff": 2,
        "frustration_rate": 0.20,
        "engagement_base": 0.65,
        "attention_drop_boost": 0.08,
        "fatigue_rate": 1.3,
        "wm_penalty": True,
        "routine_penalty": False,
        "sensory_sensitivity": 0.08,
        "noise_std": 0.05,
    },
    "routine-seeking": {
        "severity": "severe",
        "base_diff": 1,
        "frustration_rate": 0.28,
        "engagement_base": 0.52,
        "attention_drop_boost": 0.05,
        "fatigue_rate": 2.0,
        "wm_penalty": False,
        "routine_penalty": True,
        "sensory_sensitivity": 0.10,
        "noise_std": 0.06,
    },
    "sensory-hypersensitive": {
        "severity": "moderate",
        "base_diff": 2,
        "frustration_rate": 0.12,
        "engagement_base": 0.60,
        "attention_drop_boost": 0.10,
        "fatigue_rate": 1.4,
        "wm_penalty": False,
        "routine_penalty": False,
        "sensory_sensitivity": 0.50,
        "noise_std": 0.05,
    },
}

NUM_STUDENTS = 500
SESSIONS_PER_STUDENT = 15
STEPS_PER_SESSION = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Helpers ────────────────────────────────────────────────────────────

def noisify(value, noise_std, min_val, max_val):
    return float(np.clip(value + np.random.normal(0, noise_std), min_val, max_val))


def generate_student_params(profile_name):
    template = PROFILE_TEMPLATES[profile_name]
    noise = template["noise_std"]
    return {
        "profile_name": profile_name,
        "severity": template["severity"],
        "base_diff": int(round(noisify(template["base_diff"], noise, 1, 5))),
        "frustration_rate": noisify(template["frustration_rate"], noise, 0.02, 0.40),
        "engagement_base": noisify(template["engagement_base"], noise, 0.30, 0.95),
        "attention_drop_boost": noisify(template["attention_drop_boost"], noise * 0.6, 0.01, 0.25),
        "fatigue_rate": noisify(template["fatigue_rate"], noise * 0.5, 0.5, 3.0),
        "wm_penalty": template["wm_penalty"],
        "routine_penalty": template["routine_penalty"],
        "sensory_sensitivity": noisify(template["sensory_sensitivity"], noise, 0.0, 0.70),
    }


def pick_next_domain(current_domain, domain_list, transition_matrix):
    if current_domain not in domain_list:
        return random.choice(domain_list)
    idx = domain_list.index(current_domain)
    probs = transition_matrix[idx].copy()
    probs[idx] = 0
    if probs.sum() == 0:
        return random.choice([d for d in domain_list if d != current_domain])
    probs = probs / probs.sum()
    return np.random.choice(domain_list, p=probs)


def compute_perceived_difficulty(designed_diff, p, attention_drop_flag, domain_switch_count):
    pdiff = designed_diff
    if p["wm_penalty"]:
        pdiff += 1
    if p["wm_penalty"] and attention_drop_flag == 1:
        pdiff += 1
    if p["routine_penalty"] and domain_switch_count > 1:
        pdiff += 1
    return min(5, pdiff)


def simulate_step(student_id, session_id, sequence_step,
                  domain, task_id, difficulty, p,
                  consecutive_correct, session_num,
                  fatigue_index, domain_switch_count,
                  steps_since_switch,
                  domain_mastery, recent_history):  # <-- NEW: State tracking dicts
    cross_session_fatigue = 1.0 + (session_num / SESSIONS_PER_STUDENT) * 0.20
    fatigue_index = min(1.0, fatigue_index)

    base_correct_prob = max(0.15, min(0.92, 0.5 + (3 - difficulty) * 0.12))
    fatigue_penalty = fatigue_index * 0.08    # Halved to reduce single-step noise

    sensory_penalty = (SENSORY_LOAD[domain] / 3.0) * p["sensory_sensitivity"] * 0.15 * (1 + fatigue_index * 0.5)  # Halved

    # ================= FIX START =================
    # 1. Long-term learning (mastery grows as student practices domain)
    mastery_offset = (domain_mastery[domain] - 0.5) * 0.60
    
    # 2. Short-term autocorrelation (success breeds immediate success)
    if len(recent_history[domain]) > 0:
        recent_avg = np.mean(list(recent_history[domain]))
        autocorr_offset = (recent_avg - 0.5) * 0.15
    else:
        autocorr_offset = 0.0
        
    # 3. Global momentum (positive lag-1 across ALL domains)
    if len(recent_history["_global"]) > 0:
        global_avg = np.mean(list(recent_history["_global"]))
        global_momentum = (global_avg - 0.5) * 0.05
    else:
        global_momentum = 0.0
    # ================= FIX END =================

    correct_prob = max(0.05, base_correct_prob - fatigue_penalty - sensory_penalty 
                       + mastery_offset + autocorr_offset + global_momentum) # <-- NEW: Add offsets
    correct = int(random.random() < correct_prob)

    # ================= FIX START =================
    # Update state AFTER generating outcome
    learning_rate = 0.015
    if correct:
        domain_mastery[domain] += learning_rate
    else:
        domain_mastery[domain] -= learning_rate * 0.3  # Failure decreases mastery less than success increases it
        
    domain_mastery[domain] = np.clip(domain_mastery[domain], 0.2, 0.9)
    recent_history[domain].append(correct)
    recent_history["_global"].append(correct)
    # ================= FIX END =================

    if correct:
        resp_ms = np.random.normal(4000 - difficulty * 400, 800)
    else:
        resp_ms = np.random.normal(8000 + difficulty * 500, 1500)

    if p["wm_penalty"]:
        resp_ms += 1500
    resp_ms += fatigue_index * 2000
    resp_ms = max(500.0, min(60000.0, resp_ms))

    hints_used = 0 if correct else int(np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25]))
    if p["wm_penalty"] and not correct:
        hints_used = min(3, hints_used + 1)

    base_frust = p["frustration_rate"] * cross_session_fatigue
    if not correct:
        base_frust += 0.18
    if hints_used >= 2:
        base_frust += 0.12

    if p["routine_penalty"] and steps_since_switch <= 3 and domain_switch_count > 0:
        base_frust += 0.25

    frustration_flag = int(random.random() < min(base_frust, 0.85))

    base_attn_drop = 0.08
    if not correct:
        base_attn_drop += 0.15
    if hints_used >= 2:
        base_attn_drop += 0.10
    if frustration_flag == 1:
        base_attn_drop += 0.12
    if domain == "Attention":
        base_attn_drop += 0.05
    base_attn_drop += p["attention_drop_boost"]
    attention_drop_flag = int(random.random() < min(base_attn_drop, 0.70))

    engagement = (
        p["engagement_base"]
        - frustration_flag * 0.12
        - attention_drop_flag * 0.08
        + consecutive_correct * 0.03
    )
    engagement = max(0.1, min(1.0, engagement + np.random.normal(0, 0.04)))

    perceived_difficulty = compute_perceived_difficulty(
        difficulty, p, attention_drop_flag, domain_switch_count
    )

    session_position_normalized = sequence_step / STEPS_PER_SESSION

    return {
        "student_id": student_id,
        "session_id": session_id,
        "sequence_step": sequence_step,
        "domain_id": domain,
        "task_id": task_id,
        "difficulty": difficulty,
        "correct": correct,
        "response_time_ms": round(resp_ms, 1),
        "hints_used": hints_used,
        "frustration_flag": frustration_flag,
        "attention_drop_flag": attention_drop_flag,
        "engagement_score": round(engagement, 3),
        "cognitive_profile": p["profile_name"],
        "session_fatigue_index": round(fatigue_index, 3),
        "perceived_difficulty": perceived_difficulty,
        "sensory_load": SENSORY_LOAD[domain],
        "domain_switch_count": domain_switch_count,
        "session_position_normalized": round(session_position_normalized, 3),
    }


def generate_session(student_id, session_num, p, start_domain, domain_mastery, recent_history):  # <-- NEW ARGS
    session_id = f"{student_id}_S{session_num:02d}"
    rows = []

    domain = start_domain
    difficulty = p["base_diff"]
    task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
    consecutive_correct = 0
    frustration_window = deque(maxlen=3)

    in_recovery = False
    recovery_steps_left = 0
    pre_recovery_domain = None
    pre_recovery_task = None
    recovery_switches = 0

    steps_since_domain_switch = 0
    MAX_STEPS_PER_DOMAIN = 8

    fatigue_index = 0.0
    domain_switch_count = 0
    steps_since_switch = 0

    for step in range(STEPS_PER_SESSION):
        sequence_step = step + 1

        fatigue_increment = (1.0 / STEPS_PER_SESSION) * p["fatigue_rate"]
        fatigue_index = min(1.0, fatigue_index + fatigue_increment)

        row = simulate_step(
            student_id, session_id, sequence_step,
            domain, task_id, difficulty, p,
            consecutive_correct, session_num,
            fatigue_index, domain_switch_count,
            steps_since_switch,
            domain_mastery, recent_history  # <-- NEW: Pass state dicts
        )
        rows.append(row)

        correct = row["correct"]
        frustration = row["frustration_flag"]
        hints = row["hints_used"]

        frustration_window.append(frustration)
        steps_since_domain_switch += 1
        steps_since_switch += 1

        if in_recovery:
            recovery_steps_left -= 1
            if recovery_steps_left <= 0:
                in_recovery = False
                domain = pre_recovery_domain
                task_id = pre_recovery_task
                difficulty = max(1, difficulty)
                frustration_window.clear()
                steps_since_domain_switch = 0
                steps_since_switch = 0
                domain_switch_count += 1
            if correct and not frustration:
                consecutive_correct += 1
            else:
                consecutive_correct = 0
            continue

        rolling_frust = np.mean(frustration_window) if len(frustration_window) == 3 else 0
        if rolling_frust > 0.65 and recovery_switches < 2:
            pre_recovery_domain = domain
            pre_recovery_task = task_id
            domain = random.choice(RECOVERY_DOMAINS)
            task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
            difficulty = max(1, difficulty - 1)
            consecutive_correct = 0
            in_recovery = True
            recovery_steps_left = 4
            recovery_switches += 1
            frustration_window.clear()
            steps_since_domain_switch = 0
            steps_since_switch = 0
            domain_switch_count += 1
            continue

        if steps_since_domain_switch >= MAX_STEPS_PER_DOMAIN:
            domain = pick_next_domain(domain, WORK_DOMAINS, WORK_TRANSITION_MATRIX)
            task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
            consecutive_correct = 0
            steps_since_domain_switch = 0
            steps_since_switch = 0
            domain_switch_count += 1
            continue

        if correct and not frustration:
            consecutive_correct += 1
        else:
            consecutive_correct = 0

        if consecutive_correct >= 2:
            current_tasks = DOMAIN_TASK_MAPPING[domain]
            pos = current_tasks.index(task_id) if task_id in current_tasks else 0
            if pos + 1 < len(current_tasks):
                task_id = current_tasks[pos + 1]
            else:
                domain = pick_next_domain(domain, WORK_DOMAINS, WORK_TRANSITION_MATRIX)
                task_id = random.choice(DOMAIN_TASK_MAPPING[domain])
                steps_since_domain_switch = 0
                steps_since_switch = 0
                domain_switch_count += 1
            consecutive_correct = 0
            continue

        if correct and not frustration and hints == 0:
            difficulty = min(5, difficulty + 1)
        elif not correct or hints >= 2:
            difficulty = max(1, difficulty - 1)
            current_tasks = DOMAIN_TASK_MAPPING[domain]
            pos = current_tasks.index(task_id) if task_id in current_tasks else 0
            if pos > 0 and not correct:
                task_id = current_tasks[pos - 1]

    return rows, domain


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("COGNIFY — Behavioral Data Generator V4")
    print("Cognitive Profiles: attention-variable | working-memory-limited")
    print("                   routine-seeking | sensory-hypersensitive")
    print("=" * 60)

    all_rows = []
    profile_cycle = (
        ["attention-variable"] * 125
        + ["working-memory-limited"] * 125
        + ["routine-seeking"] * 125
        + ["sensory-hypersensitive"] * 125
    )
    random.shuffle(profile_cycle)

    for s in range(NUM_STUDENTS):
        student_id = f"STU_{s:04d}"
        profile_name = profile_cycle[s]
        p = generate_student_params(profile_name)

        # ================= FIX START =================
        # Initialize mastery & history states for the student
        severity_offset = {"mild": 0.0, "moderate": -0.05, "severe": -0.10}
        domain_mastery = {d: 0.45 + severity_offset.get(p["severity"], 0.0) for d in DOMAINS}
        recent_history = {d: deque(maxlen=5) for d in DOMAINS}
        recent_history["_global"] = deque(maxlen=5)
        # ================= FIX END =================

        last_domain = None
        for sess in range(SESSIONS_PER_STUDENT):
            if sess == 0:
                session_domain = WORK_DOMAINS[s % len(WORK_DOMAINS)]
            else:
                session_domain = pick_next_domain(last_domain, WORK_DOMAINS, WORK_TRANSITION_MATRIX)

            rows, last_domain = generate_session(
                student_id, sess + 1, p, session_domain, 
                domain_mastery, recent_history  # <-- NEW: Pass states
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    print(f"\nTotal rows              : {len(df):,}")
    print(f"Students                : {df['student_id'].nunique()}")
    print(f"Sessions                : {df['session_id'].nunique()}")
    print(f"\nDomain distribution     :\n{df['domain_id'].value_counts().to_string()}")
    print(f"\nDifficulty distribution :\n{df['difficulty'].value_counts().sort_index().to_string()}")
    print(f"\nSensory load distribution :\n{df['sensory_load'].value_counts().sort_index().to_string()}")
    print(f"\nCognitive profile dist  :\n{df['cognitive_profile'].value_counts().to_string()}")

    print(f"\n--- Per-Profile Averages ---")
    for prof in PROFILE_TEMPLATES.keys():
        sub = df[df['cognitive_profile'] == prof]
        print(f"\n{prof}:")
        print(f"  Avg correct           : {sub['correct'].mean():.3f}")
        print(f"  Avg frustration       : {sub['frustration_flag'].mean():.3f}")
        print(f"  Avg attention drop    : {sub['attention_drop_flag'].mean():.3f}")
        print(f"  Avg fatigue index     : {sub['session_fatigue_index'].mean():.3f}")
        print(f"  Avg perceived diff    : {sub['perceived_difficulty'].mean():.2f}")
        print(f"  Avg designed diff     : {sub['difficulty'].mean():.2f}")
        print(f"  Avg response time (ms): {sub['response_time_ms'].mean():.0f}")
        print(f"  Avg hints used        : {sub['hints_used'].mean():.2f}")

    print(f"\n--- Perceived vs Designed Difficulty Correlation ---")
    corr = df['perceived_difficulty'].corr(df['difficulty'])
    print(f"  Pearson r = {corr:.3f}")
    print(f"  (Lower r = more profile-driven divergence)")

    # ================= FIX START =================
    # Quick validation block so you can see it works immediately
    print(f"\n--- DKT Learnability Check ---")
    autocorrs = []
    for sid in df['student_id'].unique():
        student = df[df['student_id'] == sid].sort_values(['session_id', 'sequence_step'])
        c = student['correct'].values
        if len(c) > 2:
            corr_val = np.corrcoef(c[:-1], c[1:])[0, 1]
            if not np.isnan(corr_val):
                autocorrs.append(corr_val)
    mean_ac = np.mean(autocorrs)
    print(f"  Lag-1 Autocorrelation: {mean_ac:.4f} (Needs to be > 0.05)")
    
    first_50 = df.groupby('student_id').head(50).groupby('student_id')['correct'].mean().mean()
    last_50 = df.groupby('student_id').tail(50).groupby('student_id')['correct'].mean().mean()
    print(f"  Learning Curve (First 50 vs Last 50): {first_50:.3f} -> {last_50:.3f} (Needs positive diff)")
    # ================= FIX END =================

    out = "cognify_synthetic_data.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Saved → {out} ({len(df):,} rows, {df.shape[1]} columns)")