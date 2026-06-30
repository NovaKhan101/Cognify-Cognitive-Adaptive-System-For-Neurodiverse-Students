"""
COGNIFY - Deep Knowledge Tracing (DKT) V3 TUNED
=================================================
Tracks student knowledge state per domain over time.
Standard DKT formulation from Piech et al. (2015):
  Input:  one-hot encoding of (skill, outcome) + behavioral context
  Output: probability of correct on each skill next time — size n_skills

TUNING CHANGES:
  - hidden_size: 128 → 256
  - num_layers: 2 → 3
  - Input: + session_fatigue_index + frustration_flag (18-dim)
  - epochs: 80 → 150, lr: 0.001 → 0.0005, patience: 15 → 25

Run after: generate_asd_data_v3.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import pickle
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COGNIFY - Deep Knowledge Tracing (DKT) V3 TUNED")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ==================== CONFIG (TUNED) ====================
CONFIG = {
    'hidden_size': 256,       # TUNED: 128 → 256
    'num_layers': 3,          # TUNED: 2 → 3
    'dropout': 0.3,
    'batch_size': 32,
    'learning_rate': 0.0005,  # TUNED: 0.001 → 0.0005
    'epochs': 150,            # TUNED: 80 → 150
    'patience': 25,           # TUNED: 15 → 25
    'weight_decay': 1e-4,
}

# ==================== LOAD DATA ====================
print("\n📂 Loading data...")
os.makedirs('model_data', exist_ok=True)
df = pd.read_csv('cognify_synthetic_data_v3.csv')
df = df.sort_values(['student_id', 'session_id', 'sequence_step']).reset_index(drop=True)

domains = sorted(df['domain_id'].unique().tolist())
domain_to_idx = {d: i for i, d in enumerate(domains)}
n_domains = len(domains)

print(f"   Records:  {len(df)}")
print(f"   Students: {df['student_id'].nunique()}")
print(f"   Domains:  {n_domains} → {domains}")

# ==================== DKT INPUT ENCODING (TUNED: + behavioral) ====================
n_skills = n_domains * 2
n_behavioral = 2  # fatigue_index + frustration_flag
input_size = n_skills + n_behavioral
print(f"\n🔧 DKT encoding: {input_size} dim ({n_skills} domain-outcome + {n_behavioral} behavioral)")

def encode_interaction(domain_idx, correct, fatigue, frustration):
    """
    Enhanced DKT encoding with behavioral context.
    vec[0:n_domains] = correct one-hot
    vec[n_domains:2*n_domains] = incorrect one-hot
    vec[2*n_domains] = session_fatigue_index
    vec[2*n_domains+1] = frustration_flag
    """
    vec = np.zeros(input_size)
    if correct:
        vec[domain_idx] = 1.0
    else:
        vec[domain_idx + n_domains] = 1.0
    vec[n_skills] = fatigue
    vec[n_skills + 1] = frustration
    return vec

def build_dkt_sequences(df, min_seq_len=5):
    """
    Standard DKT sequence building with behavioral context.
    Target format: -1 for untested domains, 0/1 for tested domain.
    """
    X_sequences = []
    y_sequences = []
    student_ids = []

    for student_id in df['student_id'].unique():
        student_data = df[df["student_id"] == student_id].reset_index(drop=True)

        if len(student_data) < min_seq_len + 2:
            continue

        interactions = []
        targets = []

        for _, row in student_data.iterrows():
            domain_idx = domain_to_idx[row["domain_id"]]
            correct = int(row["correct"])
            fatigue = float(row["session_fatigue_index"])
            frustration = float(row["frustration_flag"])

            interaction = encode_interaction(domain_idx, correct, fatigue, frustration)
            interactions.append(interaction)
            targets.append((domain_idx, correct))

        # Build sequences: predict step i from history [0:i]
        for i in range(min_seq_len, len(interactions) - 1):
            seq_x = np.array(interactions[max(0, i-20):i])

            # Target: the NEXT interaction's (domain, correctness)
            next_domain, next_correct = targets[i]

            # -1 for untested, 0/1 for tested
            target = np.full(n_domains, -1.0)
            target[next_domain] = float(next_correct)

            # Pad if shorter than 20
            if len(seq_x) < 20:
                pad = np.zeros((20 - len(seq_x), input_size))
                seq_x = np.vstack([pad, seq_x])

            X_sequences.append(seq_x)
            y_sequences.append(target)
            student_ids.append(student_id)

    return np.array(X_sequences), np.array(y_sequences), student_ids

print("\n🔁 Building DKT sequences...")
X_dkt, y_dkt, student_ids_dkt = build_dkt_sequences(df)
print(f"   Sequences shape: {X_dkt.shape}")
print(f"   Targets shape:   {y_dkt.shape}")
print(f"   Target values:   {-1:>3} (untested), {0:>3}/{1:>3} (tested)")

# Train/val/test split (by student to avoid leakage)
unique_students = list(set(student_ids_dkt))
np.random.seed(42)
np.random.shuffle(unique_students)

n_train = int(len(unique_students) * 0.7)
n_val   = int(len(unique_students) * 0.15)

train_students = set(unique_students[:n_train])
val_students   = set(unique_students[n_train:n_train + n_val])
test_students  = set(unique_students[n_train + n_val:])

train_mask = [sid in train_students for sid in student_ids_dkt]
val_mask   = [sid in val_students   for sid in student_ids_dkt]
test_mask  = [sid in test_students  for sid in student_ids_dkt]

X_train = X_dkt[train_mask]
X_val   = X_dkt[val_mask]
X_test  = X_dkt[test_mask]
y_train = y_dkt[train_mask]
y_val   = y_dkt[val_mask]
y_test  = y_dkt[test_mask]

print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==================== MASKED LOSS ====================
class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Masked BCE loss: only compute loss on positions where target != -1.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        mask = (targets >= 0).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=preds.device)
        loss = self.bce(preds, targets)
        loss = (loss * mask).sum() / mask.sum()
        return loss

# ==================== DKT MODEL (TUNED) ====================
class DKTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_domains, dropout=0.3):
        super(DKTModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_domains = n_domains

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, n_domains)

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        predictions = self.output_layer(last_hidden)
        return predictions

    def get_knowledge_state(self, x):
        """Extract knowledge state (sigmoid of predictions)"""
        with torch.no_grad():
            preds = self.forward(x)
            return torch.sigmoid(preds)

# ==================== TRAINING ====================
def train_dkt(model, X_train, X_val, y_train, y_val):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'])

    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_auc = 0
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    print(f"\n🏋️  Training DKT Model (TUNED)...")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Epochs: {CONFIG['epochs']} | Patience: {CONFIG['patience']}")
    print(f"   LR: {CONFIG['learning_rate']} | Hidden: {CONFIG['hidden_size']} | Layers: {CONFIG['num_layers']}")
    print(f"   Input: {input_size} dim ({n_skills} domain + {n_behavioral} behavioral)")
    print(f"   Loss: MaskedBCEWithLogitsLoss (only tested domain)")

    for epoch in range(CONFIG['epochs']):
        start = time.time()

        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        val_preds_all, val_targets_all = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_preds_all.append(torch.sigmoid(preds).cpu().numpy())
                val_targets_all.append(y_batch.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        val_preds_arr = np.vstack(val_preds_all)
        val_targets_arr = np.vstack(val_targets_all)

        # Compute AUC per domain (only on tested samples)
        aucs = []
        for d in range(n_domains):
            mask = val_targets_arr[:, d] >= 0
            y_true_d = val_targets_arr[:, d][mask]
            y_pred_d = val_preds_arr[:, d][mask]

            if len(y_true_d) > 0 and len(np.unique(y_true_d)) > 1:
                try:
                    auc = roc_auc_score(y_true_d, y_pred_d)
                    aucs.append(auc)
                except:
                    pass

        val_auc = np.mean(aucs) if aucs else 0.5

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Time: {time.time()-start:.1f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save(model.state_dict(), 'model_data/best_dkt_model.pth')
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG['patience']:
                print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                break

    print(f"   ✅ Best Val AUC: {best_val_auc:.4f}")
    return history, best_val_auc

# ==================== EVALUATE DKT ====================
def evaluate_dkt(model, X_test, y_test):
    model.eval()
    ds = DataLoader(TensorDataset(torch.FloatTensor(X_test)), batch_size=64)
    all_preds = []
    with torch.no_grad():
        for (xb,) in ds:
            preds = torch.sigmoid(model(xb.to(device)))
            all_preds.append(preds.cpu().numpy())
    all_preds = np.vstack(all_preds)

    print(f"\n📊 DKT Evaluation Results:")
    print(f"   {'Domain':<20} {'AUC':>8} {'Accuracy':>10} {'N Tested':>10}")
    print(f"   {'-'*52}")

    domain_aucs = []
    domain_accs = []

    for d, domain in enumerate(domains):
        mask = y_test[:, d] >= 0
        y_true = y_test[:, d][mask]
        y_pred_prob = all_preds[:, d][mask]
        y_pred_bin = (y_pred_prob > 0.5).astype(int)

        n_tested = len(y_true)
        acc = accuracy_score(y_true, y_pred_bin) if n_tested > 0 else 0.0

        if n_tested > 0 and len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_pred_prob)
            except:
                auc = 0.5
        else:
            auc = 0.5

        domain_aucs.append(auc)
        domain_accs.append(acc)
        print(f"   {domain:<20} {auc:>8.4f} {acc:>10.4f} {n_tested:>10}")

    mean_auc = np.mean(domain_aucs)
    mean_acc = np.mean(domain_accs)
    print(f"   {'-'*52}")
    print(f"   {'MEAN':<20} {mean_auc:>8.4f} {mean_acc:>10.4f}")

    return mean_auc, mean_acc, all_preds

# ==================== KNOWLEDGE STATE DEMO ====================
def demo_knowledge_state(model, df, student_id=None):
    if student_id is None:
        student_id = df['student_id'].unique()[0]

    student_data = df[df["student_id"] == student_id].reset_index(drop=True)
    print(f"\n🧠 Knowledge State Demo — Student: {student_id}")
    print(f"   Total attempts: {len(student_data)}")

    interactions = []
    for _, row in student_data.iterrows():
        domain_idx = domain_to_idx[row['domain_id']]
        correct = int(row['correct'])
        fatigue = float(row['session_fatigue_index'])
        frustration = float(row['frustration_flag'])
        interaction = encode_interaction(domain_idx, correct, fatigue, frustration)
        interactions.append(interaction)

    checkpoints = [5, 10, 20, min(50, len(interactions) - 1)]
    checkpoints = [c for c in checkpoints if c < len(interactions)]

    print(f"\n   {'Domain':<22}", end='')
    for cp in checkpoints:
        print(f"  After {cp:2d} tasks", end='')
    print()
    print(f"   {'-'*70}")

    for d, domain in enumerate(domains):
        print(f"   {domain:<22}", end='')
        for cp in checkpoints:
            seq = np.array(interactions[:cp])
            if len(seq) < 20:
                pad = np.zeros((20 - len(seq), input_size))
                seq = np.vstack([pad, seq])
            else:
                seq = seq[-20:]

            x = torch.FloatTensor(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                ks = model.get_knowledge_state(x)
            mastery = ks[0, d].item() * 100
            print(f"  {mastery:5.1f}%", end='')
        print()

# ==================== RUN TRAINING ====================
dkt_model = DKTModel(
    input_size=input_size,
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    n_domains=n_domains,
    dropout=CONFIG['dropout']
).to(device)

history, best_val_auc = train_dkt(dkt_model, X_train, X_val, y_train, y_val)

# Load best model
dkt_model.load_state_dict(torch.load('model_data/best_dkt_model.pth'))

# Evaluate
mean_auc, mean_acc, test_preds = evaluate_dkt(dkt_model, X_test, y_test)

# Knowledge state demo
demo_knowledge_state(dkt_model, df)

# ==================== SAVE ====================
print("\n💾 Saving DKT model...")

torch.save(dkt_model.state_dict(), 'model_data/dkt_model_final_tuned.pth')

dkt_meta = {
    'config': CONFIG,
    'domains': domains,
    'domain_to_idx': domain_to_idx,
    'n_domains': n_domains,
    'n_skills': n_skills,
    'n_behavioral': n_behavioral,
    'input_size': input_size,
    'results': {
        'mean_auc': float(mean_auc),
        'mean_acc': float(mean_acc),
        'best_val_auc': float(best_val_auc),
    },
    'history': {k: [float(v) for v in vals] for k, vals in history.items()}
}

with open('model_data/dkt_meta_tuned.json', 'w') as f:
    json.dump(dkt_meta, f, indent=2)

with open('model_data/dkt_meta_tuned.pkl', 'wb') as f:
    pickle.dump({
        **dkt_meta,
        'encode_interaction_fn': encode_interaction,
    }, f)

np.save('model_data/dkt_X_test_tuned.npy', X_test)
np.save('model_data/dkt_y_test_tuned.npy', y_test)

print(f"\n{'='*60}")
print("✅ DKT TRAINING COMPLETE (TUNED)!")
print(f"{'='*60}")
print(f"   Mean AUC:     {mean_auc:.4f}")
print(f"   Mean Acc:     {mean_acc:.4f}")
print(f"   Best Val AUC: {best_val_auc:.4f}")
print(f"   Domains:      {n_domains}")
print(f"   Input dim:    {input_size} ({n_skills} + {n_behavioral} behavioral)")
print(f"\n   What DKT gives you:")
print(f"   → Per-domain mastery score (0-100%) per student")
print(f"   → Knowledge state evolution over time")
print(f"   → Behavioral-aware predictions (fatigue + frustration)")
print(f"\n📁 Saved: model_data/dkt_model_final_tuned.pth")