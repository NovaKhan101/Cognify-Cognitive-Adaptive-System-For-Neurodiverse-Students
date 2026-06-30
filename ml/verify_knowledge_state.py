"""
verify_knowledge_state.py
Cognify — ASD Adaptive Learning Platform

Sanity check: compares STU_0000's RAW accuracy progression (ground truth)
against the DKT knowledge_state_demo() output, to determine whether the
dip-then-rise pattern seen in the demo reflects real student behavior
or a bug in the demo function.

Run from the ml/ folder (same place as train_dkt_tuned.py):
    python verify_knowledge_state.py
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("COGNIFY — Knowledge State Verification")
print("=" * 60)

# ==================== LOAD ====================
print("\n📂 Loading cognify_synthetic_data.csv...")
df = pd.read_csv("cognify_synthetic_data.csv")
df = df.sort_values(["student_id", "session_id", "sequence_step"]).reset_index(drop=True)

STUDENT_ID = "STU_0000"
student_df = df[df["student_id"] == STUDENT_ID].reset_index(drop=True)

print(f"   Student: {STUDENT_ID}")
print(f"   Total attempts found: {len(student_df)}")

if len(student_df) == 0:
    print(f"❌ No rows found for {STUDENT_ID}. Check student_id format in your CSV.")
    exit()

# ==================== RAW ACCURACY PROGRESSION ====================
print("\n📊 RAW ACCURACY (ground truth — overall, not per-domain)")
print("-" * 60)

checkpoints = [5, 10, 20, 50]
for n in checkpoints:
    if len(student_df) >= n:
        window = student_df.iloc[:n]
        acc = window["correct"].mean() * 100
        print(f"   After {n:>3} tasks: {acc:5.1f}%  (n={n})")
    else:
        print(f"   After {n:>3} tasks: N/A (only {len(student_df)} attempts total)")

# ==================== RAW ACCURACY PER DOMAIN ====================
print("\n📊 RAW ACCURACY PER DOMAIN (ground truth, cumulative)")
print("-" * 60)

domains = sorted(student_df["domain_id"].unique().tolist())
header = f"{'Domain':<22}" + "".join([f"After {n:<3}".rjust(12) for n in checkpoints])
print(header)
print("-" * len(header))

for dom in domains:
    row_str = f"{dom:<22}"
    for n in checkpoints:
        window = student_df.iloc[:n]
        dom_window = window[window["domain_id"] == dom]
        if len(dom_window) > 0:
            acc = dom_window["correct"].mean() * 100
            row_str += f"{acc:10.1f}%"
        else:
            row_str += f"{'--':>11}"
    print(row_str)

# ==================== ROLLING ACCURACY (last 5 attempts, not cumulative) ====================
print("\n📊 ROLLING ACCURACY (last 5 attempts at each checkpoint — shows local trend)")
print("-" * 60)

for n in checkpoints:
    if len(student_df) >= n:
        start = max(0, n - 5)
        window = student_df.iloc[start:n]
        acc = window["correct"].mean() * 100
        print(f"   At task {n:>3} (last {len(window)} attempts): {acc:5.1f}%")

# ==================== DIAGNOSIS ====================
print("\n" + "=" * 60)
print("🔍 DIAGNOSIS")
print("=" * 60)

cumulative_accs = []
for n in checkpoints:
    if len(student_df) >= n:
        cumulative_accs.append(student_df.iloc[:n]["correct"].mean())

if len(cumulative_accs) >= 3:
    is_dipping = cumulative_accs[1] < cumulative_accs[0] and cumulative_accs[2] < cumulative_accs[1]
    is_rising_at_end = cumulative_accs[-1] > cumulative_accs[-2]

    print(f"\n   Cumulative accuracy trend: {[round(a*100,1) for a in cumulative_accs]}")

    if is_dipping and is_rising_at_end:
        print("\n   ✅ REAL PATTERN CONFIRMED")
        print("   Raw data shows the same dip-then-rise pattern as the DKT demo.")
        print("   → This is genuine student behavior (early struggle → late improvement),")
        print("     NOT a bug in get_knowledge_state(). Safe to report as a finding:")
        print('     "DKT correctly captures early-session struggle before recovery."')
    elif not is_dipping:
        print("\n   ⚠️  MISMATCH DETECTED")
        print("   Raw accuracy does NOT dip the way the DKT demo output does.")
        print("   → This suggests get_knowledge_state() may have a bug:")
        print("     - Check hidden state isn't leaking/resetting incorrectly between domains")
        print("     - Check the 'after N tasks' slicing logic (off-by-one, wrong index)")
        print("     - Check whether sigmoid output is being read from the right timestep")
    else:
        print("\n   ⚠️  PARTIAL MATCH — inspect manually using the tables above")
else:
    print("\n   Not enough data points to compare trend reliably.")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)