"""
ml_inference_V2.py
Cognify — ASD Adaptive Learning Platform
North South University Bangladesh — Capstone 2026

Production Inference Wrapper (Fully Unlocked 8-Domain Simulation & Anti-Repetition Fallback)
Run after: train_recommender_V2.py + train_model_improved.py
"""

import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────
# REGULATED STRUCTURAL MAPPINGS
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
    "Emotional Regulation": list(range(63, 71)),
}
idx_to_domain = {i: d for i, d in enumerate(DOMAINS)}

# ── Architecture Alignment ───────────────────────────────────────────────────
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
            d_model=embed_dim, nhead=nhead, dim_feedforward=384,
            dropout=0.3, batch_first=True, norm_first=True
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
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_domains)
        )
        
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
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

# ── Difficulty Recommender (CognifyTransformer — mirrors train_model_improved.py) ──
class CognifyTransformerDiff(nn.Module):
    """Difficulty Recommender — mirrors train_model_improved.py CognifyTransformer"""
    def __init__(self, input_size=38, nhead=4, dim_feedforward=128, 
                 num_layers=2, output_size=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward, max_len=10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, 
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x))
        x = self.transformer(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(x)

# ── Core Engine ──────────────────────────────────────────────────────────────
class CognifyInferenceEngine:
    def __init__(self):
        _model_dir = os.path.normpath(os.getenv(
            "MODEL_DATA_PATH",
            os.path.join(os.path.dirname(__file__), "..", "ml", "model_data")
        ).strip())

        self.meta_path    = os.path.join(_model_dir, "recommender_meta.json")
        self.weights_path = os.path.join(_model_dir, "task_recommender_final.pth")
        self.mask_path    = os.path.join(_model_dir, "domain_task_mask.pt")
        self.diff_weights_path = os.path.join(_model_dir, "best_improved_diff_transformer.pth")
        self.fallback_mode = False

        # Check if production ML files exist. If missing, fall back gracefully.
        if not os.path.exists(self.meta_path) or not os.path.exists(self.weights_path):
            print("⚠️ ML Weights or Meta configuration files missing inside 'model_data/'")
            print("💡 Activating Adaptive Simulation Fallback Mode with all 8 Domains unlocked.")
            self.fallback_mode = True
            self.input_dim = 42
            return

        self._load_meta()
        cfg = self.meta["config"]

        self.model = RobustTransformerRecommender(
            input_dim   = cfg.get("input_dim", 42),
            num_tasks   = cfg.get("num_tasks", 71),
            num_domains = cfg.get("num_domains", 8),
            embed_dim   = cfg.get("embed_dim", 192),
            nhead       = cfg.get("nhead", 6),
            num_layers  = cfg.get("num_layers", 2)
        )
        self._load_weights()
        self._load_mask()
        
        # ── Load Difficulty Recommender ──
        self._load_difficulty_model()
        
        print(f"✅ CognifyInferenceEngine loaded: Task Recommender + Difficulty Recommender")

    def _load_meta(self):
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        self.input_dim = self.meta["config"]["input_dim"]

    def _load_weights(self):
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def _load_mask(self):
        if os.path.exists(self.mask_path):
            self.task_domain_mask = torch.load(self.mask_path, map_location=torch.device("cpu"))
        else:
            self.task_domain_mask = torch.zeros((8, 71))
            for d_idx, domain in enumerate(DOMAINS):
                for t_id in DOMAIN_TASK_MAPPING[domain]:
                    self.task_domain_mask[d_idx, t_id] = 1.0

    def _load_difficulty_model(self):
        """Load the trained Difficulty Recommender (CognifyTransformer, 38-dim input)."""
        self.difficulty_model = CognifyTransformerDiff(
            input_size=38,
            nhead=4,
            dim_feedforward=128,
            num_layers=2,
            output_size=5,
            dropout=0.3
        )
        if os.path.exists(self.diff_weights_path):
            self.difficulty_model.load_state_dict(
                torch.load(self.diff_weights_path, map_location=torch.device("cpu"))
            )
            self.difficulty_model.eval()
            print("   ✅ Difficulty Recommender loaded")
        else:
            print("   ⚠️ Difficulty weights not found — using rule-based fallback")
            self.difficulty_model = None

    def _get(self, obj, key, fallback):
        if hasattr(obj, key):
            v = getattr(obj, key)
            return v if v is not None else fallback
        if isinstance(obj, dict):
            return obj.get(key, fallback)
        return fallback

    def extract_features(self, row, recent_history):
        """Assembles the 42-dimensional feature vector exactly mirroring the trainer."""
        g = lambda k, fb: self._get(row, k, fb)
        feats = []

        # 1. Performance outcomes (2)
        correct = 1.0 if g("correct", False) or g("correct", 0) == 1 else 0.0
        feats.extend([correct, 1.0 - correct])

        # 2. Response Time (1)
        resp_ms = float(g("response_time_ms", g("response_time_sec", 10.0) * 1000.0))
        feats.append(min(resp_ms, 60000.0) / 60000.0)

        # 3. Task Difficulty Scalar (1)
        diff = float(g("difficulty", g("difficulty_level", 3)))
        feats.append(diff / 5.0)

        # 4. Telemetry metrics (4)
        feats.append(float(g("engagement_score", 0.75)))
        feats.append(float(g("hints_used", 0)) / 3.0)
        feats.append(1.0 if g("frustration_flag", False) or g("frustration_flag", 0) == 1 else 0.0)
        feats.append(1.0 if g("attention_drop_flag", False) or g("attention_drop_flag", 0) == 1 else 0.0)

        # 5. Active Domain One-Hot (8)
        domain_name = g("domain_id", "Attention")
        d_idx = DOMAINS.index(domain_name) if domain_name in DOMAINS else 0
        oh_d = [0.0] * 8
        oh_d[d_idx] = 1.0
        feats.extend(oh_d)

        # 6. Active Difficulty One-Hot (5)
        oh_diff = [0.0] * 5
        oh_diff[max(0, min(4, int(diff) - 1))] = 1.0
        feats.extend(oh_diff)

        # 7. Historic Domain Performance Averages (8)
        domain_history = [0.5] * 8
        combined_history = list(recent_history) + [row]
        for d in range(8):
            d_attempts = []
            for r in combined_history:
                r_domain = self._get(r, "domain_id", None)
                if r_domain == DOMAINS[d]:
                    d_attempts.append(1 if self._get(r, "correct", 0) or self._get(r, "correct", False) else 0)
            if d_attempts:
                domain_history[d] = sum(d_attempts) / len(d_attempts)
        feats.extend(domain_history)

        # 8. Shared Rolling Window Statistics (3)
        if recent_history:
            last5 = recent_history[-5:]
            rolling_acc = sum(1 if self._get(r, "correct", 0) or self._get(r, "correct", False) else 0 for r in last5) / len(last5)
            rolling_rt  = np.mean([min(float(self._get(r, "response_time_ms", 10000)), 60000.0) / 60000.0 for r in last5])
            rolling_eng = np.mean([float(self._get(r, "engagement_score", 0.75)) for r in last5])
        else:
            rolling_acc, rolling_rt, rolling_eng = correct, min(resp_ms, 60000.0) / 60000.0, float(g("engagement_score", 0.75))
        feats.extend([float(rolling_acc), float(rolling_rt), float(rolling_eng)])

        # 9. V3 Behavioral Context (4) — matches trainer build_features()
        feats.append(float(g("session_fatigue_index", 0.0)))
        feats.append(float(g("perceived_difficulty", diff)) / 5.0)
        feats.append(float(g("sensory_load", 2)) / 3.0)
        feats.append(float(g("domain_switch_count", 0)) / 10.0)

        # 10. Global context features (2) — matches preprocess.py indices 34-35
        # Historical baseline accuracy (all attempts before current session)
        hist_acc = float(np.mean([
            1 if (self._get(r, "correct", False) or self._get(r, "correct", 0) == 1) else 0
            for r in recent_history
        ])) if recent_history else 0.5
        feats.append(hist_acc)

        # Session progress (session number / 15)
        sess_id = g("session_id", "S01")
        try:
            sess_num = int(str(sess_id).split("S")[-1])
        except (ValueError, IndexError):
            sess_num = 1
        feats.append(sess_num / 15.0)

        # Pad to INPUT_DIM=42 (same as trainer)
        while len(feats) < self.input_dim:
            feats.append(0.0)

        assert len(feats) == self.input_dim, f"Feature dim mismatch: got {len(feats)}, expected {self.input_dim}"
        return feats

    def _predict_difficulty(self, seq_tensor):
        """Use trained Difficulty Recommender (CognifyTransformer, 38-dim), fallback to rule-based."""
        if self.difficulty_model is not None and not self.fallback_mode:
            with torch.no_grad():
                diff_logits = self.difficulty_model(seq_tensor)
                diff_pred = int(torch.argmax(diff_logits, dim=1).item()) + 1
                diff_conf = float(torch.softmax(diff_logits, dim=1).max().item())
                return diff_pred, diff_conf, "Transformer_DifficultyModel"
        return None, None, None

    def get_next_task(self, attempts_list, temperature=0.4):
        # Handle empty arrays / initial task loading
        if not attempts_list:
            return {
                "recommended_task_id": 0,
                "predicted_domain":    "Attention",
                "recommended_difficulty": 2,
                "domain_confidence":   1.0,
                "model_used":          "TransformerV2_ColdStart",
            }

        # ── EXPLICIT FALLBACK LOGIC FOR FRONTEND DESIGN TESTING ──
        if self.fallback_mode:
            chosen_domain = random.choice(DOMAINS)
            valid_ids = DOMAIN_TASK_MAPPING[chosen_domain]
            
            seen_task_ids = [int(self._get(a, "task_id", -1)) for a in attempts_list]
            unseen_ids = [tid for tid in valid_ids if tid not in seen_task_ids]
            pool_to_choose_from = unseen_ids if unseen_ids else valid_ids
            chosen_task_id = random.choice(pool_to_choose_from)
            
            recent5 = attempts_list[-5:]
            correct_count = sum(1 for a in recent5 if self._get(a, "correct", False) or self._get(a, "correct", 0) == 1)
            base_diff = int(self._get(attempts_list[-1], "difficulty", 3))
            
            if correct_count >= 4:
                rec_diff = min(5, base_diff + 1)
            elif correct_count <= 1:
                rec_diff = max(1, base_diff - 1)
            else:
                rec_diff = base_diff

            return {
                "recommended_task_id":     chosen_task_id,
                "predicted_domain":       chosen_domain,
                "recommended_difficulty": rec_diff,
                "domain_confidence":       0.95,
                "model_used":              "TransformerV2_SimulatedFallback",
            }

        # ── PRODUCTION PYTORCH TRANSFORMER EVALUATION LOOP ──
        windowed = attempts_list[-20:]
        seq_feats = []
        
        for i, row in enumerate(windowed):
            slice_idx = max(0, len(attempts_list) - len(windowed) + i)
            recent = attempts_list[:slice_idx]
            seq_feats.append(self.extract_features(row, recent))

        while len(seq_feats) < 20:
            seq_feats.insert(0, [0.0] * self.input_dim)

        seq_tensor = torch.tensor([seq_feats], dtype=torch.float32)

        # Build separate 38-dim sequence for Difficulty Recommender
        # Uses base 36 features + 2 streak features (approximated as zeros at inference)
        diff_seq_feats = []
        for i, row in enumerate(windowed):
            slice_idx = max(0, len(attempts_list) - len(windowed) + i)
            recent = attempts_list[:slice_idx]
            f42 = self.extract_features(row, recent)
            # Take first 36 features + append 2 streak placeholders
            f38 = f42[:36] + [0.0, 0.0]
            diff_seq_feats.append(f38)

        while len(diff_seq_feats) < 10:   # seq_len=10 for difficulty model
            diff_seq_feats.insert(0, [0.0] * 38)
        diff_seq_feats = diff_seq_feats[-10:]  # take last 10

        diff_seq_tensor = torch.tensor([diff_seq_feats], dtype=torch.float32)
        diff_pred, diff_conf, diff_tag = self._predict_difficulty(diff_seq_tensor)

        with torch.no_grad():
            task_logits, domain_logits = self.model(seq_tensor)
            domain_probs = torch.softmax(domain_logits, dim=-1).squeeze(0)

        pred_domain_idx = int(torch.argmax(domain_probs).item())
        confidence      = float(domain_probs[pred_domain_idx].item())

        mask = self.task_domain_mask[pred_domain_idx, :]
        masked_task_logits = task_logits.squeeze(0) * mask + (1.0 - mask) * -1e9

        if confidence < 0.18:
            valid = torch.where(mask > 0)[0].tolist()
            chosen_task_id = random.choice(valid) if valid else 0
            model_tag = "TransformerV2_ConfidenceFallback"
        else:
            task_probs = torch.softmax(masked_task_logits / max(temperature, 0.01), dim=-1)
            chosen_task_id = int(torch.multinomial(task_probs, 1).item())
            model_tag = "GatedTransformer_V2_Production"

        # ── Difficulty: try ML model first, fallback to rules ──
        if diff_pred is None:
            # Rule-based fallback
            recent5 = attempts_list[-5:]
            correct_count = sum(1 for a in recent5 if self._get(a, "correct", False) or self._get(a, "correct", 0) == 1)
            base_diff = int(self._get(attempts_list[-1], "difficulty", 3))
            
            if correct_count >= 4:
                diff_pred = min(5, base_diff + 1)
            elif correct_count <= 1:
                diff_pred = max(1, base_diff - 1)
            else:
                diff_pred = base_diff
            diff_tag = "RuleBased_Difficulty"
            diff_conf = 0.5

        return {
            "recommended_task_id":    chosen_task_id,
            "predicted_domain":       idx_to_domain.get(pred_domain_idx, "Attention"),
            "recommended_difficulty": diff_pred,
            "domain_confidence":      round(confidence, 4),
            "difficulty_confidence":  round(diff_conf, 4) if diff_conf else None,
            "model_used":             model_tag,
            "difficulty_model_used":  diff_tag,
        }

    def extract_student_context(self, attempts_list):
        """Extract rich student context for Gemini prompt builder."""
        if not attempts_list:
            return {
                "cognitive_profile": "unknown",
                "avg_fatigue": 0.0,
                "avg_perceived_difficulty": 3.0,
                "avg_sensory_load": 2.0,
                "domain_switch_count": 0,
                "recent_accuracy": 0.5,
                "frustration_rate": 0.0,
                "attention_drop_rate": 0.0,
                "consecutive_correct": 0,
                "session_position": 0.5,
                "weak_domains": [],
            }

        recent = attempts_list[-10:]
        correct_count = sum(1 for a in recent if self._get(a, "correct", False) or self._get(a, "correct", 0) == 1)
        recent_accuracy = correct_count / len(recent)
        
        frustrations = sum(1 for a in recent if self._get(a, "frustration_flag", 0) == 1)
        attention_drops = sum(1 for a in recent if self._get(a, "attention_drop_flag", 0) == 1)
        
        # Consecutive correct streak
        consecutive_correct = 0
        for a in reversed(recent):
            if self._get(a, "correct", False) or self._get(a, "correct", 0) == 1:
                consecutive_correct += 1
            else:
                break

        # Weak domains (accuracy < 0.5)
        domain_stats = {}
        for a in attempts_list:
            d = self._get(a, "domain_id", "unknown")
            if d not in domain_stats:
                domain_stats[d] = {"attempts": 0, "correct": 0}
            domain_stats[d]["attempts"] += 1
            if self._get(a, "correct", False) or self._get(a, "correct", 0) == 1:
                domain_stats[d]["correct"] += 1
        
        weak_domains = [
            d for d, s in domain_stats.items()
            if s["attempts"] >= 3 and s["correct"] / s["attempts"] < 0.5
        ]

        # V3 behavioral aggregates
        avg_fatigue = np.mean([float(self._get(a, "session_fatigue_index", 0.0)) for a in recent])
        avg_perceived_diff = np.mean([float(self._get(a, "perceived_difficulty", 3.0)) for a in recent])
        avg_sensory_load = np.mean([float(self._get(a, "sensory_load", 2.0)) for a in recent])
        total_domain_switches = max([int(self._get(a, "domain_switch_count", 0)) for a in attempts_list])
        cognitive_profile = self._get(attempts_list[-1], "cognitive_profile", "unknown")

        return {
            "cognitive_profile": cognitive_profile,
            "avg_fatigue": round(float(avg_fatigue), 3),
            "avg_perceived_difficulty": round(float(avg_perceived_diff), 2),
            "avg_sensory_load": round(float(avg_sensory_load), 2),
            "domain_switch_count": int(total_domain_switches),
            "recent_accuracy": round(float(recent_accuracy), 3),
            "frustration_rate": round(frustrations / len(recent), 3),
            "attention_drop_rate": round(attention_drops / len(recent), 3),
            "consecutive_correct": int(consecutive_correct),
            "session_position": round(
                float(self._get(attempts_list[-1], "session_position_normalized", 0.5)), 3
            ),
            "weak_domains": weak_domains,
        }

    def get_session_report(self, attempts_list):
        if not attempts_list:
            return {"error": "No attempts in session."}

        correct_count = 0
        total_time = 0.0
        frustrations = 0
        attention_drops = 0
        domain_stats = {}
        fatigue_values = []
        perceived_diff_values = []
        cognitive_profile = "unknown"

        for a in attempts_list:
            corr  = bool(self._get(a, "correct", False)) or self._get(a, "correct", 0) == 1
            resp  = float(self._get(a, "response_time_sec", float(self._get(a, "response_time_ms", 0.0)) / 1000.0))
            frust = bool(self._get(a, "frustration_flag", False)) or self._get(a, "frustration_flag", 0) == 1
            attn  = bool(self._get(a, "attention_drop_flag", False)) or self._get(a, "attention_drop_flag", 0) == 1
            d_id  = self._get(a, "domain_id", self._get(a, "domain", "unknown"))
            
            # V3 fields
            fatigue_values.append(float(self._get(a, "session_fatigue_index", 0.0)))
            perceived_diff_values.append(float(self._get(a, "perceived_difficulty", 3.0)))
            cognitive_profile = self._get(a, "cognitive_profile", cognitive_profile)

            if corr: correct_count += 1
            total_time += resp
            if frust: frustrations += 1
            if attn:  attention_drops += 1

            if d_id not in domain_stats:
                domain_stats[d_id] = {"attempts": 0, "correct": 0}
            domain_stats[d_id]["attempts"] += 1
            if corr:
                domain_stats[d_id]["correct"] += 1

        total = len(attempts_list)
        return {
            "total_attempts":   total,
            "correct_answers":  correct_count,
            "overall_accuracy": round(correct_count / total * 100, 2),
            "total_time_mins":  round(total_time / 60.0, 1),
            "frustration_count": frustrations,
            "attention_drops":  attention_drops,
            "engagement_score": round((1 - frustrations / max(total, 1)) * 100, 2),
            # ── V3 fields ──
            "cognitive_profile": cognitive_profile,
            "avg_fatigue": round(np.mean(fatigue_values), 3) if fatigue_values else 0.0,
            "avg_perceived_difficulty": round(np.mean(perceived_diff_values), 2) if perceived_diff_values else 3.0,
            "max_fatigue": round(max(fatigue_values), 3) if fatigue_values else 0.0,
            "domains": {
                dom: {
                    "attempts":  s["attempts"],
                    "correct":   s["correct"],
                    "accuracy":  round(s["correct"] / s["attempts"] * 100, 2),
                } for dom, s in domain_stats.items()
            },
        }

# ── Singleton Pattern ────────────────────────────────────────────────────────
_engine_instance = None

def get_engine() -> CognifyInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CognifyInferenceEngine()
    return _engine_instance