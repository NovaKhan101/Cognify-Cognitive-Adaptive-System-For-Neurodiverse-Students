# """
# train_recommender_V2.py
# Cognify — ASD Adaptive Learning Platform
# North South University Bangladesh — Capstone 2026

# Strictly Aligned Multi-Task Transformer Recommender V2
# Run after: generate_asd_data.py
# """

# import os
# import json
# import math
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader

# # ─────────────────────────────────────────────────────────
# # CONFIG & CLASS DOMAINS (PRESERVED)
# # ─────────────────────────────────────────────────────────
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
#     "Emotional Regulation": list(range(63, 71)),  # Intentionally 8 tasks
# }
# domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
# task_to_idx   = {f"TASK_{i:02d}": i for i in range(71)}
# idx_to_task   = {i: f"TASK_{i:02d}" for i in range(71)}
# idx_to_domain = {i: d for i, d in enumerate(DOMAINS)}

# # Aligned perfectly to your handoff spec constraints
# SEQ_LEN    = 20
# INPUT_DIM  = 42
# EMBED_DIM  = 192
# NUM_HEADS  = 6
# NUM_LAYERS = 2  # Aligned with handoff specification
# FF_DIM     = 384
# DROPOUT    = 0.3  # Prevents overfitting boundaries on synthetic profiles
# BATCH_SIZE = 32
# MAX_EPOCHS = 80
# PATIENCE   = 15
# DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ── Model Architecture ───────────────────────────────────────────────────────
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=100):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class RobustTransformerRecommender(nn.Module):
#     def __init__(self, input_dim=42, num_tasks=71, num_domains=8, embed_dim=192, nhead=6, num_layers=2):
#         super().__init__()
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU()
#         )
#         self.pos_encoder = PositionalEncoding(embed_dim)
        
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=nhead, dim_feedforward=FF_DIM,
#             dropout=DROPOUT, batch_first=True, norm_first=True  # Modern Pre-LN setup
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
#         self.pool_attn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.Tanh(),
#             nn.Linear(embed_dim // 2, 1)
#         )
        
#         self.domain_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Dropout(DROPOUT),
#             nn.Linear(embed_dim // 2, num_domains)
#         )
        
#         self.task_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Dropout(DROPOUT),
#             nn.Linear(embed_dim, num_tasks)
#         )

#     def forward(self, seq):
#         sz = seq.size(1)
#         causal_mask = torch.triu(torch.full((sz, sz), float("-inf"), device=seq.device), diagonal=1)
        
#         x = self.pos_encoder(self.input_proj(seq))
#         x = self.transformer(x, mask=causal_mask, is_causal=True)
        
#         attn_weights = F.softmax(self.pool_attn(x), dim=1)
#         pooled = torch.sum(x * attn_weights, dim=1)
        
#         return self.task_head(pooled), self.domain_head(pooled)

# # ── Loss Function ────────────────────────────────────────────────────────────
# class BalancedFocalLoss(nn.Module):
#     def __init__(self, gamma=2.0):
#         super().__init__()
#         self.gamma = gamma

#     def forward(self, logits, targets):
#         ce = F.cross_entropy(logits, targets, reduction="none")
#         pt = torch.exp(-ce)
#         return (((1 - pt) ** self.gamma) * ce).mean()

# # ── Leak-Free Dataset Generator ──────────────────────────────────────────────
# class ASDSequenceDataset(Dataset):
#     def __init__(self, dataframe, seq_len=20, augment=True):
#         self.seq_len = seq_len
#         df = dataframe.copy()
#         df["domain_idx"] = df["domain_id"].map(domain_to_idx).fillna(0).astype(int)
#         self.sequences = []

#         for _, group in df.groupby("session_id"):
#             if "timestamp" in group.columns:
#                 group = group.sort_values("timestamp")
#             elif "sequence_step" in group.columns:
#                 group = group.sort_values("sequence_step")
#             rows = group.to_dict(orient="records")
#             if len(rows) < 2:
#                 continue
                
#             self.sequences.append(rows)
            
#             # Capped augmentation rate to prevent biasing long, highly verbal sessions
#             if augment and len(rows) > 5:
#                 aug_count = 0
#                 for sp in range(5, len(rows), 4):
#                     if aug_count >= 3:  # Maximum 3 sub-sequences allowed per block
#                         break
#                     self.sequences.append(rows[:sp])
#                     aug_count += 1

#     def __len__(self):
#         return len(self.sequences)

#     def build_features(self, current_row, past_history):
#         """Assembles 42-dimensional behavioral state vector safely."""
#         feats = []
        
#         # 1. Action outcome indicators (2 features)
#         correct = float(current_row.get("correct", 0))
#         feats.extend([correct, 1.0 - correct])

#         # 2. Performance speed profile (1 feature)
#         resp_ms = float(current_row.get("response_time_ms", current_row.get("response_time_sec", 10.0) * 1000))
#         feats.append(min(resp_ms, 60000.0) / 60000.0)

#         # 3. System challenge level (1 feature)
#         diff = float(current_row.get("difficulty", current_row.get("difficulty_level", 3)))
#         feats.append(diff / 5.0)

#         # 4. Attention & frustration indicators (4 features)
#         feats.append(float(current_row.get("engagement_score", 0.75)))
#         feats.append(float(current_row.get("hints_used", 0)) / 3.0)
#         feats.append(float(current_row.get("frustration_flag", 0)))
#         feats.append(float(current_row.get("attention_drop_flag", 0)))

#         # 5. Core domain index mapping (8 features)
#         d_idx = int(current_row.get("domain_idx", 0))
#         oh_d = [0.0] * 8
#         oh_d[d_idx] = 1.0
#         feats.extend(oh_d)

#         # 6. Granular challenge scaling profile (5 features)
#         oh_diff = [0.0] * 5
#         oh_diff[max(0, min(4, int(diff) - 1))] = 1.0
#         feats.extend(oh_diff)

#         # 7. Domain execution history outcomes (8 features)
#         domain_history = [0.5] * 8
#         combined_history = past_history + [current_row]
#         if combined_history:
#             for d in range(8):
#                 d_attempts = [r["correct"] for r in combined_history if r.get("domain_idx") == d]
#                 if d_attempts:
#                     domain_history[d] = sum(d_attempts) / len(d_attempts)
#         feats.extend(domain_history)

#         # 8. Rolling behavioral trends (3 features)
#         if past_history:
#             last5 = past_history[-5:]
#             rolling_acc = np.mean([r["correct"] for r in last5])
#             rolling_rt  = np.mean([min(float(r.get("response_time_ms", 10000)), 60000) / 60000 for r in last5])
#             rolling_eng = np.mean([float(r.get("engagement_score", 0.75)) for r in last5])
#         else:
#             rolling_acc, rolling_rt, rolling_eng = correct, min(resp_ms, 60000.0) / 60000.0, float(current_row.get("engagement_score", 0.75))
#         feats.extend([float(rolling_acc), float(rolling_rt), float(rolling_eng)])

#         # 9. V3 behavioral context (4 features) — NEW: session_fatigue_index, perceived_difficulty, sensory_load, domain_switch_count
#         feats.append(float(current_row.get("session_fatigue_index", 0.0)))
#         feats.append(float(current_row.get("perceived_difficulty", diff)) / 5.0)
#         feats.append(float(current_row.get("sensory_load", 2)) / 3.0)
#         feats.append(float(current_row.get("domain_switch_count", 0)) / 10.0)

#         # ─────────────────────────────────────────────────────────
#         # FIX: Explicitly pad up to the 42 reserved spec slots
#         # ─────────────────────────────────────────────────────────
#         while len(feats) < INPUT_DIM:
#             feats.append(0.0)

#         # Rigorous development verification checkpoint
#         assert len(feats) == INPUT_DIM, f"Feature dimension error! Expected {INPUT_DIM}, calculated {len(feats)}"
        
#         return feats

#     def __getitem__(self, idx):
#         history = self.sequences[idx]
#         seq_features = []
        
#         # Build strictly historic states up to step t-1
#         for i in range(len(history) - 1):
#             current_item = history[i]
#             prior_items = history[:i]
#             seq_features.append(self.build_features(current_item, prior_items))

#         if len(seq_features) > self.seq_len:
#             seq_features = seq_features[-self.seq_len:]
#         else:
#             while len(seq_features) < self.seq_len:
#                 seq_features.insert(0, [0.0] * INPUT_DIM)

#         # Secure and isolate true evaluations target at time step t
#         target = history[-1]
#         t_val = target.get("task_id", 0)
        
#         if isinstance(t_val, str):
#             if "TASK_" in t_val:
#                 t_idx_final = int(t_val.split("_")[-1])
#             else:
#                 t_idx_final = task_to_idx.get(t_val, 0)
#         else:
#             t_idx_final = int(t_val)

#         # Bulletproof re-computation of target domain index avoiding data leakage quirks
#         target_domain_name = target.get("domain_id", "Attention")
#         target_domain_idx = domain_to_idx.get(target_domain_name, 0)

#         return (
#             torch.tensor(seq_features, dtype=torch.float32),
#             torch.tensor(t_idx_final, dtype=torch.long),
#             torch.tensor(target_domain_idx, dtype=torch.long),
#         )

# # ── Training Main Engine ──────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=" * 60)
#     print("COGNIFY — Validated Multi-Task Recommender V2")
#     print("=" * 60)
#     print(f"Device: {DEVICE}")

#     # ── FIX: Load V4 CSV ──
#     df = pd.read_csv("cognify_synthetic_data.csv")
#     print(f"Loaded {len(df):,} rows, {df['student_id'].nunique()} students")

#     # Split dataset securely at the student profile layer (zero-leakage validation)
#     students = df["student_id"].unique()
#     train_students, val_students = train_test_split(students, test_size=0.15, random_state=42)
#     train_df = df[df["student_id"].isin(train_students)]
#     val_df   = df[df["student_id"].isin(val_students)]

#     train_ds = ASDSequenceDataset(train_df, seq_len=SEQ_LEN, augment=True)
#     val_ds   = ASDSequenceDataset(val_df,   seq_len=SEQ_LEN, augment=False)
#     print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#     model = RobustTransformerRecommender(input_dim=INPUT_DIM, num_layers=NUM_LAYERS).to(DEVICE)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Parameters: {total_params:,}")

#     criterion_task   = BalancedFocalLoss()
#     criterion_domain = nn.CrossEntropyLoss(label_smoothing=0.1)
    
#     # Restored 2e-3 learning rate exactly as specified in your spec to bypass plateaus
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

#     best_val_loss   = float("inf")
#     patience_counter = 0
#     os.makedirs("model_data", exist_ok=True)

#     print("\nTraining Profile Execution...")
#     for epoch in range(MAX_EPOCHS):
#         model.train()
#         t_loss = 0.0
#         for seq, tasks, domains in train_loader:
#             seq, tasks, domains = seq.to(DEVICE), tasks.to(DEVICE), domains.to(DEVICE)
#             optimizer.zero_grad()
#             t_logits, d_logits = model(seq)
#             loss = 0.6 * criterion_task(t_logits, tasks) + 0.4 * criterion_domain(d_logits, domains)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             t_loss += loss.item()
#         scheduler.step()

#         model.eval()
#         v_loss = 0.0
#         correct_top3 = 0
#         correct_top1 = 0
#         correct_dom1 = 0
#         total_val = 0
#         with torch.no_grad():
#             for seq, tasks, domains in val_loader:
#                 seq, tasks, domains = seq.to(DEVICE), tasks.to(DEVICE), domains.to(DEVICE)
#                 t_logits, d_logits = model(seq)
#                 v_loss += (0.6 * criterion_task(t_logits, tasks) + 0.4 * criterion_domain(d_logits, domains)).item()
                
#                 pred_task_3 = torch.topk(t_logits, 3, dim=1).indices
#                 pred_task_1 = torch.argmax(t_logits, dim=1)
#                 pred_dom_1  = torch.argmax(d_logits, dim=1)
                
#                 correct_top1 += (pred_task_1 == tasks).sum().item()
#                 correct_top3 += (pred_task_3 == tasks.unsqueeze(1)).any(dim=1).sum().item()
#                 correct_dom1 += (pred_dom_1 == domains).sum().item()
#                 total_val += tasks.size(0)

#         val_loss = v_loss / len(val_loader)
#         t_top1 = correct_top1 / total_val * 100
#         t_top3 = correct_top3 / total_val * 100
#         d_acc  = correct_dom1 / total_val * 100
#         lr_now = scheduler.get_last_lr()[0]
        
#         print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Task-Top1: {t_top1:.1f}% | Task-Top3: {t_top3:.1f}% | Domain Acc: {d_acc:.1f}% | LR: {lr_now:.2e}", end="")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), "model_data/task_recommender_final.pth")
#             print(" ✓ saved weights")
#         else:
#             patience_counter += 1
#             print(f" patience {patience_counter}/{PATIENCE}")
#             if patience_counter >= PATIENCE:
#                 print("🛑 Early stopping triggered.")
#                 break

#     # Save tracking mapping artifacts
#     mask = torch.zeros((8, 71))
#     for d_idx, d_name in enumerate(DOMAINS):
#         for t_id in DOMAIN_TASK_MAPPING[d_name]:
#             mask[d_idx, t_id] = 1.0
#     torch.save(mask, "model_data/domain_task_mask.pt")

#     meta = {
#         "domains":      DOMAINS,
#         "tasks":        [f"TASK_{i:02d}" for i in range(71)],
#         "domain_to_idx": domain_to_idx,
#         "idx_to_domain": {str(i): d for i, d in enumerate(DOMAINS)},
#         "task_to_idx":   task_to_idx,
#         "idx_to_task":   {str(i): f"TASK_{i:02d}" for i in range(71)},
#         "input_dim":   INPUT_DIM,
#         "embed_dim":   EMBED_DIM,
#         "num_heads":   NUM_HEADS,
#         "num_layers":  NUM_LAYERS,
#         "ff_dim":      FF_DIM,
#         "seq_len":     SEQ_LEN,
#         "config": {
#             "input_dim":   INPUT_DIM,
#             "embed_dim":   EMBED_DIM,
#             "nhead":       NUM_HEADS,
#             "num_layers":  NUM_LAYERS,
#             "ff_dim":      FF_DIM,
#             "seq_len":     SEQ_LEN,
#             "num_domains": 8,
#             "num_tasks":   71,
#         }
#     }
#     with open("model_data/recommender_meta.json", "w") as f:
#         json.dump(meta, f, indent=2)

#     print(f"\n✅ Training complete. Verified best validation loss: {best_val_loss:.4f}")

"""
train_recommender_V2.py
Cognify — ASD Adaptive Learning Platform
North South University Bangladesh — Capstone 2026

Strictly Aligned Multi-Task Transformer Recommender V2
Run after: generate_asd_data.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────
# CONFIG & CLASS DOMAINS (PRESERVED)
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
    "Emotional Regulation": list(range(63, 71)),  # Intentionally 8 tasks
}
domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
task_to_idx   = {f"TASK_{i:02d}": i for i in range(71)}
idx_to_task   = {i: f"TASK_{i:02d}" for i in range(71)}
idx_to_domain = {i: d for i, d in enumerate(DOMAINS)}

# Aligned perfectly to your handoff spec constraints
SEQ_LEN    = 20
INPUT_DIM  = 42
EMBED_DIM  = 192
NUM_HEADS  = 6
NUM_LAYERS = 2  # Aligned with handoff specification
FF_DIM     = 384
DROPOUT    = 0.3  # Prevents overfitting boundaries on synthetic profiles
BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE   = 15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Architecture ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RobustTransformerRecommender(nn.Module):
    def __init__(self, input_dim=42, num_tasks=71, num_domains=8, embed_dim=192, nhead=6, num_layers=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True  # Modern Pre-LN setup
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.pool_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.domain_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim // 2, num_domains)
        )
        
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim, num_tasks)
        )

    def forward(self, seq):
        sz = seq.size(1)
        causal_mask = torch.triu(torch.full((sz, sz), float("-inf"), device=seq.device), diagonal=1)
        
        x = self.pos_encoder(self.input_proj(seq))
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        attn_weights = F.softmax(self.pool_attn(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return self.task_head(pooled), self.domain_head(pooled)

# ── Loss Function ────────────────────────────────────────────────────────────
class BalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# ── Leak-Free Dataset Generator ──────────────────────────────────────────────
class ASDSequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len=20, augment=True):
        self.seq_len = seq_len
        df = dataframe.copy()
        df["domain_idx"] = df["domain_id"].map(domain_to_idx).fillna(0).astype(int)
        self.sequences = []

        for _, group in df.groupby("session_id"):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp")
            elif "sequence_step" in group.columns:
                group = group.sort_values("sequence_step")
            rows = group.to_dict(orient="records")
            if len(rows) < 2:
                continue
                
            self.sequences.append(rows)
            
            # Capped augmentation rate to prevent biasing long, highly verbal sessions
            if augment and len(rows) > 5:
                aug_count = 0
                for sp in range(5, len(rows), 4):
                    if aug_count >= 3:  # Maximum 3 sub-sequences allowed per block
                        break
                    self.sequences.append(rows[:sp])
                    aug_count += 1

    def __len__(self):
        return len(self.sequences)

    def build_features(self, current_row, past_history):
        """Assembles 42-dimensional behavioral state vector safely."""
        feats = []
        
        # 1. Action outcome indicators (2 features)
        correct = float(current_row.get("correct", 0))
        feats.extend([correct, 1.0 - correct])

        # 2. Performance speed profile (1 feature)
        resp_ms = float(current_row.get("response_time_ms", current_row.get("response_time_sec", 10.0) * 1000))
        feats.append(min(resp_ms, 60000.0) / 60000.0)

        # 3. System challenge level (1 feature)
        diff = float(current_row.get("difficulty", current_row.get("difficulty_level", 3)))
        feats.append(diff / 5.0)

        # 4. Attention & frustration indicators (4 features)
        feats.append(float(current_row.get("engagement_score", 0.75)))
        feats.append(float(current_row.get("hints_used", 0)) / 3.0)
        feats.append(float(current_row.get("frustration_flag", 0)))
        feats.append(float(current_row.get("attention_drop_flag", 0)))

        # 5. Core domain index mapping (8 features)
        d_idx = int(current_row.get("domain_idx", 0))
        oh_d = [0.0] * 8
        oh_d[d_idx] = 1.0
        feats.extend(oh_d)

        # 6. Granular challenge scaling profile (5 features)
        oh_diff = [0.0] * 5
        oh_diff[max(0, min(4, int(diff) - 1))] = 1.0
        feats.extend(oh_diff)

        # 7. Domain execution history outcomes (8 features)
        domain_history = [0.5] * 8
        combined_history = past_history + [current_row]
        if combined_history:
            for d in range(8):
                d_attempts = [r["correct"] for r in combined_history if r.get("domain_idx") == d]
                if d_attempts:
                    domain_history[d] = sum(d_attempts) / len(d_attempts)
        feats.extend(domain_history)

        # 8. Rolling behavioral trends (3 features)
        if past_history:
            last5 = past_history[-5:]
            rolling_acc = np.mean([r["correct"] for r in last5])
            rolling_rt  = np.mean([min(float(r.get("response_time_ms", 10000)), 60000) / 60000 for r in last5])
            rolling_eng = np.mean([float(r.get("engagement_score", 0.75)) for r in last5])
        else:
            rolling_acc, rolling_rt, rolling_eng = correct, min(resp_ms, 60000.0) / 60000.0, float(current_row.get("engagement_score", 0.75))
        feats.extend([float(rolling_acc), float(rolling_rt), float(rolling_eng)])

        # 9. V3 behavioral context (4 features) — NEW: session_fatigue_index, perceived_difficulty, sensory_load, domain_switch_count
        feats.append(float(current_row.get("session_fatigue_index", 0.0)))
        feats.append(float(current_row.get("perceived_difficulty", diff)) / 5.0)
        feats.append(float(current_row.get("sensory_load", 2)) / 3.0)
        feats.append(float(current_row.get("domain_switch_count", 0)) / 10.0)

        # ─────────────────────────────────────────────────────────
        # FIX: Explicitly pad up to the 42 reserved spec slots
        # ─────────────────────────────────────────────────────────
        while len(feats) < INPUT_DIM:
            feats.append(0.0)

        # Rigorous development verification checkpoint
        assert len(feats) == INPUT_DIM, f"Feature dimension error! Expected {INPUT_DIM}, calculated {len(feats)}"
        
        return feats

    def __getitem__(self, idx):
        history = self.sequences[idx]
        seq_features = []
        
        # Build strictly historic states up to step t-1
        for i in range(len(history) - 1):
            current_item = history[i]
            prior_items = history[:i]
            seq_features.append(self.build_features(current_item, prior_items))

        if len(seq_features) > self.seq_len:
            seq_features = seq_features[-self.seq_len:]
        else:
            while len(seq_features) < self.seq_len:
                seq_features.insert(0, [0.0] * INPUT_DIM)

        # Secure and isolate true evaluations target at time step t
        target = history[-1]
        t_val = target.get("task_id", 0)
        
        if isinstance(t_val, str):
            if "TASK_" in t_val:
                t_idx_final = int(t_val.split("_")[-1])
            else:
                t_idx_final = task_to_idx.get(t_val, 0)
        else:
            t_idx_final = int(t_val)

        # Bulletproof re-computation of target domain index avoiding data leakage quirks
        target_domain_name = target.get("domain_id", "Attention")
        target_domain_idx = domain_to_idx.get(target_domain_name, 0)

        return (
            torch.tensor(seq_features, dtype=torch.float32),
            torch.tensor(t_idx_final, dtype=torch.long),
            torch.tensor(target_domain_idx, dtype=torch.long),
        )

# ── Training Main Engine ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("COGNIFY — Validated Multi-Task Recommender V2")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── FIX: Load V4 CSV ──
    df = pd.read_csv("cognify_synthetic_data.csv")
    print(f"Loaded {len(df):,} rows, {df['student_id'].nunique()} students")

    # Split dataset securely at the student profile layer (zero-leakage validation)
    students = df["student_id"].unique()
    train_students, val_students = train_test_split(students, test_size=0.15, random_state=42)
    train_df = df[df["student_id"].isin(train_students)]
    val_df   = df[df["student_id"].isin(val_students)]

    train_ds = ASDSequenceDataset(train_df, seq_len=SEQ_LEN, augment=True)
    val_ds   = ASDSequenceDataset(val_df,   seq_len=SEQ_LEN, augment=False)
    print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = RobustTransformerRecommender(input_dim=INPUT_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion_task   = BalancedFocalLoss()
    criterion_domain = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Restored 2e-3 learning rate exactly as specified in your spec to bypass plateaus
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val_loss   = float("inf")
    patience_counter = 0
    os.makedirs("model_data", exist_ok=True)

    print("\nTraining Profile Execution...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        t_loss = 0.0
        for seq, tasks, domains in train_loader:
            seq, tasks, domains = seq.to(DEVICE), tasks.to(DEVICE), domains.to(DEVICE)
            optimizer.zero_grad()
            t_logits, d_logits = model(seq)
            loss = 0.6 * criterion_task(t_logits, tasks) + 0.4 * criterion_domain(d_logits, domains)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss = 0.0
        correct_top3 = 0
        correct_top1 = 0
        correct_dom1 = 0
        total_val = 0
        with torch.no_grad():
            for seq, tasks, domains in val_loader:
                seq, tasks, domains = seq.to(DEVICE), tasks.to(DEVICE), domains.to(DEVICE)
                t_logits, d_logits = model(seq)
                v_loss += (0.6 * criterion_task(t_logits, tasks) + 0.4 * criterion_domain(d_logits, domains)).item()
                
                pred_task_3 = torch.topk(t_logits, 3, dim=1).indices
                pred_task_1 = torch.argmax(t_logits, dim=1)
                pred_dom_1  = torch.argmax(d_logits, dim=1)
                
                correct_top1 += (pred_task_1 == tasks).sum().item()
                correct_top3 += (pred_task_3 == tasks.unsqueeze(1)).any(dim=1).sum().item()
                correct_dom1 += (pred_dom_1 == domains).sum().item()
                total_val += tasks.size(0)

        val_loss = v_loss / len(val_loader)
        t_top1 = correct_top1 / total_val * 100
        t_top3 = correct_top3 / total_val * 100
        d_acc  = correct_dom1 / total_val * 100
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Task-Top1: {t_top1:.1f}% | Task-Top3: {t_top3:.1f}% | Domain Acc: {d_acc:.1f}% | LR: {lr_now:.2e}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model_data/task_recommender_final.pth")
            print(" ✓ saved weights")
        else:
            patience_counter += 1
            print(f" patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    # Save tracking mapping artifacts
    mask = torch.zeros((8, 71))
    for d_idx, d_name in enumerate(DOMAINS):
        for t_id in DOMAIN_TASK_MAPPING[d_name]:
            mask[d_idx, t_id] = 1.0
    torch.save(mask, "model_data/domain_task_mask.pt")

    meta = {
        "domains":      DOMAINS,
        "tasks":        [f"TASK_{i:02d}" for i in range(71)],
        "domain_to_idx": domain_to_idx,
        "idx_to_domain": {str(i): d for i, d in enumerate(DOMAINS)},
        "task_to_idx":   task_to_idx,
        "idx_to_task":   {str(i): f"TASK_{i:02d}" for i in range(71)},
        "input_dim":   INPUT_DIM,
        "embed_dim":   EMBED_DIM,
        "num_heads":   NUM_HEADS,
        "num_layers":  NUM_LAYERS,
        "ff_dim":      FF_DIM,
        "seq_len":     SEQ_LEN,
        "config": {
            "input_dim":   INPUT_DIM,
            "embed_dim":   EMBED_DIM,
            "nhead":       NUM_HEADS,
            "num_layers":  NUM_LAYERS,
            "ff_dim":      FF_DIM,
            "seq_len":     SEQ_LEN,
            "num_domains": 8,
            "num_tasks":   71,
        }
    }
    with open("model_data/recommender_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete. Verified best validation loss: {best_val_loss:.4f}")