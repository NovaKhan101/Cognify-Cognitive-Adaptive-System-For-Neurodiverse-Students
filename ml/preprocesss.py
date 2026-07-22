# """
# preprocess.py
# Cognify — ASD Adaptive Learning Platform
# Optimized — vectorized, no pandas in loops, no leakage
# """

# import numpy as np
# import pandas as pd
# import pickle
# import os
# import warnings
# warnings.filterwarnings('ignore')

# print("=" * 60)
# print("COGNIFY — Preprocessing Dataset (Optimized)")
# print("=" * 60)

# # ==================== LOAD ====================
# print("\n📂 Loading cognify_synthetic_data_v3.csv...")
# df = pd.read_csv("cognify_synthetic_data_v3.csv")
# df = df.sort_values(["student_id", "session_id", "sequence_step"]).reset_index(drop=True)

# print(f"   Records:  {len(df):,}")
# print(f"   Students: {df['student_id'].nunique()}")

# # ==================== PRE-COMPUTE ALL FEATURES (VECTORIZED) ====================
# print("\n🔧 Pre-computing features (vectorized)...")

# DOMAINS = sorted(df["domain_id"].unique().tolist())
# domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
# n_domains = len(DOMAINS)

# # Global stats
# global_mean_rt = df["response_time_ms"].mean()
# global_std_rt = df["response_time_ms"].std()
# if global_std_rt == 0 or np.isnan(global_std_rt):
#     global_std_rt = 1.0

# # Convert to numpy for speed
# student_ids = df["student_id"].values
# session_ids = df["session_id"].values
# domain_ids = df["domain_id"].values
# task_ids = df["task_id"].values
# difficulties = df["difficulty"].values.astype(np.float32)
# corrects = df["correct"].values.astype(np.float32)
# response_times = df["response_time_ms"].values.astype(np.float32)
# hints = df["hints_used"].values.astype(np.float32)
# frustrations = df["frustration_flag"].values.astype(np.float32)
# attention_drops = df["attention_drop_flag"].values.astype(np.float32)
# engagements = df["engagement_score"].values.astype(np.float32)
# # ── NEW v3 columns ──
# fatigue_index = df["session_fatigue_index"].values.astype(np.float32)
# perceived_diff = df["perceived_difficulty"].values.astype(np.float32)

# # Encode domains as indices
# domain_indices = np.array([domain_to_idx[d] for d in domain_ids], dtype=np.int32)

# # Pre-compute per-student domain accuracy (cumulative, NO LEAKAGE)
# print("   Computing per-student domain histories...")
# student_domain_correct = {}
# student_domain_count = {}

# for i in range(len(df)):
#     sid = student_ids[i]
#     d_idx = domain_indices[i]
#     corr = corrects[i]

#     if sid not in student_domain_correct:
#         student_domain_correct[sid] = {d: [] for d in range(n_domains)}
#         student_domain_count[sid] = {d: 0 for d in range(n_domains)}

#     student_domain_correct[sid][d_idx].append(corr)
#     student_domain_count[sid][d_idx] += 1

# # Pre-compute rolling stats per student (vectorized within each student)
# print("   Computing rolling statistics...")
# unique_students = np.unique(student_ids)
# rolling_acc_5 = np.zeros(len(df), dtype=np.float32)
# rolling_rt_5 = np.zeros(len(df), dtype=np.float32)
# rolling_eng_5 = np.zeros(len(df), dtype=np.float32)

# for sid in unique_students:
#     mask = student_ids == sid
#     idx = np.where(mask)[0]
#     n = len(idx)

#     for i in range(n):
#         pos = idx[i]
#         start = max(0, i - 4)
#         sub_idx = idx[start:i+1]
#         rolling_acc_5[pos] = corrects[sub_idx].mean()
#         rolling_rt_5[pos] = (response_times[sub_idx].mean() - global_mean_rt) / (global_std_rt * 3) + 0.5
#         rolling_eng_5[pos] = engagements[sub_idx].mean()

# # Clip rolling RT
# rolling_rt_5 = np.clip(rolling_rt_5, 0.0, 1.0)

# # ==================== BUILD SEQUENCES ====================
# print("\n🔁 Building sequences...")

# SEQ_LEN = 10
# N_FEATURES = 34  # Updated: 32 + fatigue_index + perceived_difficulty

# # Group by student then session
# student_session_groups = df.groupby(["student_id", "session_id"]).indices

# X_perf = []
# y_perf = []
# X_diff = []
# y_diff = []
# out_student_ids = []

# for (sid, sess_id), group_idx in student_session_groups.items():
#     group_idx = np.sort(group_idx)
#     n = len(group_idx)

#     if n < SEQ_LEN + 1:
#         continue

#     # Get student's domain history BEFORE this session
#     student_all_idx = np.where(student_ids == sid)[0]
#     session_start_idx = group_idx[0]
#     history_idx = student_all_idx[student_all_idx < session_start_idx]

#     # Pre-compute domain accuracy from history
#     hist_domain_acc = np.zeros(n_domains, dtype=np.float32)
#     for d in range(n_domains):
#         hist_count = sum(1 for h_idx in history_idx if domain_indices[h_idx] == d)
#         if hist_count > 0:
#             hist_sum = sum(corrects[h_idx] for h_idx in history_idx if domain_indices[h_idx] == d)
#             hist_domain_acc[d] = hist_sum / hist_count
#         else:
#             hist_domain_acc[d] = 0.5

#     for i in range(n - SEQ_LEN):
#         window_idx = group_idx[i:i+SEQ_LEN]
#         next_idx = group_idx[i+SEQ_LEN]

#         seq_feats = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

#         for t, w_idx in enumerate(window_idx):
#             # 1. Performance outcomes (2)
#             seq_feats[t, 0] = corrects[w_idx]
#             seq_feats[t, 1] = 1.0 - corrects[w_idx]

#             # 2. Response Time (1)
#             seq_feats[t, 2] = np.clip((response_times[w_idx] - global_mean_rt) / (global_std_rt * 3) + 0.5, 0.0, 1.0)

#             # 3. Difficulty (1)
#             seq_feats[t, 3] = difficulties[w_idx] / 5.0

#             # 4. Telemetry (4)
#             seq_feats[t, 4] = engagements[w_idx]
#             seq_feats[t, 5] = hints[w_idx] / 3.0
#             seq_feats[t, 6] = frustrations[w_idx]
#             seq_feats[t, 7] = attention_drops[w_idx]

#             # 5. Domain one-hot (8)
#             seq_feats[t, 8 + domain_indices[w_idx]] = 1.0

#             # 6. Difficulty one-hot (5)
#             diff_oh = max(0, min(4, int(difficulties[w_idx]) - 1))
#             seq_feats[t, 16 + diff_oh] = 1.0

#             # 7. Historic domain performance (8) — from history + window so far
#             for d in range(n_domains):
#                 count = 0
#                 acc = hist_domain_acc[d]
#                 for j in range(t+1):
#                     prev_idx = window_idx[j]
#                     if domain_indices[prev_idx] == d:
#                         count += 1
#                         if count == 1:
#                             acc = corrects[prev_idx]
#                         else:
#                             acc = (acc * (count - 1) + corrects[prev_idx]) / count
#                 seq_feats[t, 21 + d] = acc if count > 0 else hist_domain_acc[d]

#             # 8. Rolling stats (3)
#             seq_feats[t, 29] = rolling_acc_5[w_idx]
#             seq_feats[t, 30] = rolling_rt_5[w_idx]
#             seq_feats[t, 31] = rolling_eng_5[w_idx]

#             # ── NEW v3 features (2) ──
#             seq_feats[t, 32] = fatigue_index[w_idx]           # session_fatigue_index
#             seq_feats[t, 33] = perceived_diff[w_idx] / 5.0    # perceived_difficulty normalized

#         X_perf.append(seq_feats)
#         y_perf.append(1 if corrects[next_idx] > 0.5 else 0)
#         X_diff.append(seq_feats.copy())
#         y_diff.append(int(difficulties[next_idx]) - 1)
#         out_student_ids.append(sid)

# X_perf = np.array(X_perf, dtype=np.float32)
# y_perf = np.array(y_perf, dtype=np.int64)
# X_diff = np.array(X_diff, dtype=np.float32)
# y_diff = np.array(y_diff, dtype=np.int64)

# print(f"   Performance sequences: {X_perf.shape}")
# print(f"   Performance targets:   {y_perf.shape}")

# # ==================== TRAIN/VAL/TEST SPLIT ====================
# print("\n📊 Splitting data...")
# unique_out_students = list(set(out_student_ids))
# np.random.seed(42)
# np.random.shuffle(unique_out_students)

# n_train = int(len(unique_out_students) * 0.7)
# n_val = int(len(unique_out_students) * 0.15)

# train_students = set(unique_out_students[:n_train])
# val_students = set(unique_out_students[n_train:n_train + n_val])
# test_students = set(unique_out_students[n_train + n_val:])

# train_mask = np.array([sid in train_students for sid in out_student_ids])
# val_mask = np.array([sid in val_students for sid in out_student_ids])
# test_mask = np.array([sid in test_students for sid in out_student_ids])

# X_train = X_perf[train_mask]
# X_val = X_perf[val_mask]
# X_test = X_perf[test_mask]
# y_train = y_perf[train_mask]
# y_val = y_perf[val_mask]
# y_test = y_perf[test_mask]

# X_diff_train = X_diff[train_mask]
# X_diff_val = X_diff[val_mask]
# X_diff_test = X_diff[test_mask]
# y_diff_train = y_diff[train_mask]
# y_diff_val = y_diff[val_mask]
# y_diff_test = y_diff[test_mask]

# print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
# print(f"   Class balance (perf): correct={y_train.mean():.3f}")
# print(f"   Difficulty dist (train): {np.bincount(y_diff_train)}")

# # ==================== SAVE ====================
# print("\n💾 Saving...")
# os.makedirs("model_data", exist_ok=True)

# np.save("model_data/X_train.npy", X_train)
# np.save("model_data/X_val.npy", X_val)
# np.save("model_data/X_test.npy", X_test)
# np.save("model_data/y_train.npy", y_train)
# np.save("model_data/y_val.npy", y_val)
# np.save("model_data/y_test.npy", y_test)

# np.save("model_data/X_diff_train.npy", X_diff_train)
# np.save("model_data/X_diff_val.npy", X_diff_val)
# np.save("model_data/X_diff_test.npy", X_diff_test)
# np.save("model_data/y_diff_train.npy", y_diff_train)
# np.save("model_data/y_diff_val.npy", y_diff_val)
# np.save("model_data/y_diff_test.npy", y_diff_test)

# feature_info = {
#     "n_features": N_FEATURES,
#     "seq_len": SEQ_LEN,
#     "domains": DOMAINS,
#     "domain_to_idx": domain_to_idx,
#     "n_domains": n_domains,
#     "n_difficulty_levels": 5,
# }

# with open("model_data/feature_info.pkl", "wb") as f:
#     pickle.dump(feature_info, f)

# print(f"\n{'='*60}")
# print("✅ PREPROCESSING COMPLETE!")
# print(f"{'='*60}")
# print(f"\n📁 Saved in model_data/")
# print(f"\n▶️  Next: Run train_model_improved.py")

"""
preprocess.py
Cognify — ASD Adaptive Learning Platform
Optimized — vectorized, no pandas in loops, no leakage
V4.1 UPDATE: Added DKT sequence generation + Global Context Features
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COGNIFY — Preprocessing Dataset (Optimized)")
print("=" * 60)

# ==================== LOAD ====================
print("\n📂 Loading cognify_synthetic_data.csv...")
df = pd.read_csv("cognify_synthetic_data.csv")
df = df.sort_values(["student_id", "session_id", "sequence_step"]).reset_index(drop=True)

print(f"   Records:  {len(df):,}")
print(f"   Students: {df['student_id'].nunique()}")

# ==================== PRE-COMPUTE ALL FEATURES (VECTORIZED) ====================
print("\n🔧 Pre-computing features (vectorized)...")

DOMAINS = sorted(df["domain_id"].unique().tolist())
domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
n_domains = len(DOMAINS)

# Global stats
global_mean_rt = df["response_time_ms"].mean()
global_std_rt = df["response_time_ms"].std()
if global_std_rt == 0 or np.isnan(global_std_rt):
    global_std_rt = 1.0

# Convert to numpy for speed
student_ids = df["student_id"].values
session_ids = df["session_id"].values
domain_ids = df["domain_id"].values
task_ids = df["task_id"].values
difficulties = df["difficulty"].values.astype(np.float32)
corrects = df["correct"].values.astype(np.float32)
response_times = df["response_time_ms"].values.astype(np.float32)
hints = df["hints_used"].values.astype(np.float32)
frustrations = df["frustration_flag"].values.astype(np.float32)
attention_drops = df["attention_drop_flag"].values.astype(np.float32)
engagements = df["engagement_score"].values.astype(np.float32)
fatigue_index = df["session_fatigue_index"].values.astype(np.float32)
perceived_diff = df["perceived_difficulty"].values.astype(np.float32)

# Encode domains as indices
domain_indices = np.array([domain_to_idx[d] for d in domain_ids], dtype=np.int32)

# Pre-compute per-student domain accuracy (cumulative, NO LEAKAGE)
print("   Computing per-student domain histories...")
student_domain_correct = {}
student_domain_count = {}

for i in range(len(df)):
    sid = student_ids[i]
    d_idx = domain_indices[i]
    corr = corrects[i]

    if sid not in student_domain_correct:
        student_domain_correct[sid] = {d: [] for d in range(n_domains)}
        student_domain_count[sid] = {d: 0 for d in range(n_domains)}

    student_domain_correct[sid][d_idx].append(corr)
    student_domain_count[sid][d_idx] += 1

# Pre-compute rolling stats per student (vectorized within each student)
print("   Computing rolling statistics...")
unique_students = np.unique(student_ids)
rolling_acc_5 = np.zeros(len(df), dtype=np.float32)
rolling_rt_5 = np.zeros(len(df), dtype=np.float32)
rolling_eng_5 = np.zeros(len(df), dtype=np.float32)

for sid in unique_students:
    mask = student_ids == sid
    idx = np.where(mask)[0]
    n = len(idx)

    for i in range(n):
        pos = idx[i]
        start = max(0, i - 4)
        sub_idx = idx[start:i+1]
        rolling_acc_5[pos] = corrects[sub_idx].mean()
        rolling_rt_5[pos] = (response_times[sub_idx].mean() - global_mean_rt) / (global_std_rt * 3) + 0.5
        rolling_eng_5[pos] = engagements[sub_idx].mean()

rolling_rt_5 = np.clip(rolling_rt_5, 0.0, 1.0)

# ==================== BUILD SEQUENCES (ORIGINAL MODELS) ====================
print("\n🔁 Building sequences for LSTM/CNN models...")

SEQ_LEN = 10
N_FEATURES = 36  # Updated: 34 + 2 global context features

student_session_groups = df.groupby(["student_id", "session_id"]).indices

X_perf = []
y_perf = []
X_diff = []
y_diff = []
out_student_ids = []

for (sid, sess_id), group_idx in student_session_groups.items():
    group_idx = np.sort(group_idx)
    n = len(group_idx)

    if n < SEQ_LEN + 1:
        continue

    student_all_idx = np.where(student_ids == sid)[0]
    session_start_idx = group_idx[0]
    history_idx = student_all_idx[student_all_idx < session_start_idx]

    hist_domain_acc = np.zeros(n_domains, dtype=np.float32)
    for d in range(n_domains):
        hist_count = sum(1 for h_idx in history_idx if domain_indices[h_idx] == d)
        if hist_count > 0:
            hist_sum = sum(corrects[h_idx] for h_idx in history_idx if domain_indices[h_idx] == d)
            hist_domain_acc[d] = hist_sum / hist_count
        else:
            hist_domain_acc[d] = 0.5

    for i in range(n - SEQ_LEN):
        window_idx = group_idx[i:i+SEQ_LEN]
        next_idx = group_idx[i+SEQ_LEN]

        seq_feats = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

        for t, w_idx in enumerate(window_idx):
            seq_feats[t, 0] = corrects[w_idx]
            seq_feats[t, 1] = 1.0 - corrects[w_idx]
            seq_feats[t, 2] = np.clip((response_times[w_idx] - global_mean_rt) / (global_std_rt * 3) + 0.5, 0.0, 1.0)
            seq_feats[t, 3] = difficulties[w_idx] / 5.0
            seq_feats[t, 4] = engagements[w_idx]
            seq_feats[t, 5] = hints[w_idx] / 3.0
            seq_feats[t, 6] = frustrations[w_idx]
            seq_feats[t, 7] = attention_drops[w_idx]
            seq_feats[t, 8 + domain_indices[w_idx]] = 1.0
            diff_oh = max(0, min(4, int(difficulties[w_idx]) - 1))
            seq_feats[t, 16 + diff_oh] = 1.0
            
            for d in range(n_domains):
                count = 0
                acc = hist_domain_acc[d]
                for j in range(t+1):
                    prev_idx = window_idx[j]
                    if domain_indices[prev_idx] == d:
                        count += 1
                        if count == 1:
                            acc = corrects[prev_idx]
                        else:
                            acc = (acc * (count - 1) + corrects[prev_idx]) / count
                seq_feats[t, 21 + d] = acc if count > 0 else hist_domain_acc[d]

            seq_feats[t, 29] = rolling_acc_5[w_idx]
            seq_feats[t, 30] = rolling_rt_5[w_idx]
            seq_feats[t, 31] = rolling_eng_5[w_idx]
            seq_feats[t, 32] = fatigue_index[w_idx]           
            seq_feats[t, 33] = perceived_diff[w_idx] / 5.0   

            # ── GLOBAL CONTEXT FEATURES (Leakage-Free) ──
            seq_feats[t, 34] = corrects[history_idx].mean() if len(history_idx) > 0 else 0.5  # Historical baseline
            try:
                sess_num = int(str(sess_id).split("S")[-1])
            except (ValueError, IndexError):
                sess_num = 1
            seq_feats[t, 35] = sess_num / 15.0                   # Session progress

        X_perf.append(seq_feats)
        y_perf.append(1 if corrects[next_idx] > 0.5 else 0)
        X_diff.append(seq_feats.copy())
        y_diff.append(int(difficulties[next_idx]) - 1)
        out_student_ids.append(sid)

X_perf = np.array(X_perf, dtype=np.float32)
y_perf = np.array(y_perf, dtype=np.int64)
X_diff = np.array(X_diff, dtype=np.float32)
y_diff = np.array(y_diff, dtype=np.int64)

print(f"   Performance sequences: {X_perf.shape}")

# ==================== SPLIT ORIGINAL MODELS ====================
print("\n📊 Splitting data for LSTM/CNN...")
unique_out_students = list(set(out_student_ids))
np.random.seed(42)
np.random.shuffle(unique_out_students)

n_train = int(len(unique_out_students) * 0.7)
n_val = int(len(unique_out_students) * 0.15)

train_students = set(unique_out_students[:n_train])
val_students = set(unique_out_students[n_train:n_train + n_val])
test_students = set(unique_out_students[n_train + n_val:])

train_mask = np.array([sid in train_students for sid in out_student_ids])
val_mask = np.array([sid in val_students for sid in out_student_ids])
test_mask = np.array([sid in test_students for sid in out_student_ids])

# ==================== SAVE ORIGINAL MODEL DATA ====================
print("\n💾 Saving LSTM/CNN data...")
os.makedirs("model_data", exist_ok=True)

np.save("model_data/X_train.npy", X_perf[train_mask])
np.save("model_data/X_val.npy", X_perf[val_mask])
np.save("model_data/X_test.npy", X_perf[test_mask])
np.save("model_data/y_train.npy", y_perf[train_mask])
np.save("model_data/y_val.npy", y_perf[val_mask])
np.save("model_data/y_test.npy", y_perf[test_mask])

np.save("model_data/X_diff_train.npy", X_diff[train_mask])
np.save("model_data/X_diff_val.npy", X_diff[val_mask])
np.save("model_data/X_diff_test.npy", X_diff[test_mask])
np.save("model_data/y_diff_train.npy", y_diff[train_mask])
np.save("model_data/y_diff_val.npy", y_diff[val_mask])
np.save("model_data/y_diff_test.npy", y_diff[test_mask])


# ===================================================================================
# ── DKT SEQUENCES (ADDED FOR DEEP KNOWLEDGE TRACING) ──
# ===================================================================================
print("\n🔁 Building DKT-specific sequences (20-dim)...")

n_skills = n_domains * 2
n_behavioral = 2
dkt_input_size = n_skills + n_behavioral + 2  # 16 + 2 behavioral + 2 temporal = 20
DKT_SEQ_LEN = 20

X_dkt = []
y_dkt = []
dkt_student_ids = []

# Use exact same student splits to prevent leakage
dkt_train_students = train_students
dkt_val_students = val_students
dkt_test_students = test_students

# Extract the exact temporal columns you already generated
session_pos_normalized = df["session_position_normalized"].values.astype(np.float32)

for sid in unique_students:
    student_mask = student_ids == sid
    student_idx = np.where(student_mask)[0]
    
    if len(student_idx) < 7: # Need at least 5 history + 1 target + 1 buffer
        continue

    interactions = []
    targets = []

    for step_num, idx in enumerate(student_idx):
        d_idx = domain_indices[idx]
        c = int(corrects[idx])
        f = float(fatigue_index[idx])
        fr = float(frustrations[idx])
        
        # Exact temporal context (no modulo approximation)
        current_session_pos = session_pos_normalized[idx]
        overall_progress = step_num / len(student_idx)

        vec = np.zeros(dkt_input_size, dtype=np.float32)
        if c:
            vec[d_idx] = 1.0
        else:
            vec[d_idx + n_domains] = 1.0
        vec[n_skills] = f
        vec[n_skills + 1] = fr
        
        # Append exact temporal features
        vec[n_skills + 2] = current_session_pos
        vec[n_skills + 3] = overall_progress
        
        interactions.append(vec)
        targets.append((d_idx, c))

    # Build sliding windows
    for i in range(5, len(interactions) - 1):
        seq_x = np.array(interactions[max(0, i - DKT_SEQ_LEN):i])
        next_domain, next_correct = targets[i]

        # DKT Target: -1 for untested, 0/1 for tested
        target = np.full(n_domains, -1.0, dtype=np.float32)
        target[next_domain] = float(next_correct)

        # Pad to 20
        if len(seq_x) < DKT_SEQ_LEN:
            pad = np.zeros((DKT_SEQ_LEN - len(seq_x), dkt_input_size), dtype=np.float32)
            seq_x = np.vstack([pad, seq_x])

        X_dkt.append(seq_x)
        y_dkt.append(target)
        dkt_student_ids.append(sid)

X_dkt = np.array(X_dkt, dtype=np.float32)
y_dkt = np.array(y_dkt, dtype=np.float32)

# Split DKT Data using the EXACT SAME students
dkt_train_mask = np.array([sid in dkt_train_students for sid in dkt_student_ids])
dkt_val_mask = np.array([sid in dkt_val_students for sid in dkt_student_ids])
dkt_test_mask = np.array([sid in dkt_test_students for sid in dkt_student_ids])

print(f"   DKT Sequences shape: {X_dkt.shape}")
print(f"   DKT Targets shape:   {y_dkt.shape}")
print(f"   DKT Train: {dkt_train_mask.sum()} | Val: {dkt_val_mask.sum()} | Test: {dkt_test_mask.sum()}")

# Save DKT arrays
np.save("model_data/dkt_X_train.npy", X_dkt[dkt_train_mask])
np.save("model_data/dkt_X_val.npy", X_dkt[dkt_val_mask])
np.save("model_data/dkt_X_test.npy", X_dkt[dkt_test_mask])
np.save("model_data/dkt_y_train.npy", y_dkt[dkt_train_mask])
np.save("model_data/dkt_y_val.npy", y_dkt[dkt_val_mask])
np.save("model_data/dkt_y_test.npy", y_dkt[dkt_test_mask])


# ==================== SAVE METADATA ====================
feature_info = {
    "n_features": N_FEATURES, # Now 36
    "seq_len": SEQ_LEN,
    "domains": DOMAINS,
    "domain_to_idx": domain_to_idx,
    "n_domains": n_domains,
    "n_difficulty_levels": 5,
    # DKT specific info (Unchanged)
    "dkt_input_size": dkt_input_size,
    "dkt_seq_len": DKT_SEQ_LEN,
}

with open("model_data/feature_info.pkl", "wb") as f:
    pickle.dump(feature_info, f)

print(f"\n{'='*60}")
print("✅ PREPROCESSING COMPLETE (LSTM/CNN + DKT)!")
print(f"{'='*60}")
print(f"\n📁 Saved in model_data/")
print(f"\n▶️  Next: Run train_model_improved.py")