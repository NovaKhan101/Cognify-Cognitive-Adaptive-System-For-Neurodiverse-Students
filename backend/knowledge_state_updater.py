"""
knowledge_state_updater.py
Cognify — Deep Knowledge Tracing (DKT) State Updater
Runs after every session to update the knowledge_states table based on full attempt history.
"""

import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import DB models
from models import KnowledgeState, TaskAttempt, Student

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────
DOMAINS = [
    "Attention", "Memory", "Language", "Executive Function",
    "Social Cognition", "Sensory Processing", "Motor Skills", "Emotional Regulation"
]

DKT_INPUT_SIZE = 20
DKT_SEQ_LEN = 20

# Exact config used by the trainer
DKT_CONFIG = {
    'embed_size': 64,
    'exercise_embed_size': 16, 
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
}

MODEL_DATA_PATH = os.path.normpath(os.getenv(
    "MODEL_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "ml", "model_data")
).strip())

# Configurable weights filename to match exactly what the trainer output
DKT_WEIGHTS_FILENAME = os.getenv("DKT_WEIGHTS_FILENAME", "dkt_model_final.pth")

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (Exact copy from train_dkt_tuned.py)
# ─────────────────────────────────────────────────────────
class EADKTModel(nn.Module):
    def __init__(self, input_size, n_domains, config):
        super(EADKTModel, self).__init__()
        self.n_domains = n_domains
        hidden = config['hidden_size']
        
        # FIX 1: Context size is exactly 4 (fatigue, frustration, session_pos, overall_pos)
        context_size = input_size - (n_domains * 2) 
        
        # 1. Context embedding (processes the 4 behavioral/temporal features)
        self.input_embed = nn.Sequential(
            nn.Linear(context_size, config['embed_size']), 
            nn.ReLU(), 
            nn.Dropout(config['dropout'] * 0.5)
        )
        
        # 2. Exercise (Domain) Embedding (processes the 16-dim one-hot)
        self.exercise_embed = nn.Linear(n_domains * 2, config['exercise_embed_size'])
        
        # 3. LSTM takes combined vector
        lstm_input_size = config['embed_size'] + config['exercise_embed_size']
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden, 
            num_layers=config['num_layers'], 
            batch_first=True, 
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden)
        self.output_layer = nn.Linear(hidden, n_domains)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x):
        # Split input: first 16 dims are domain one-hot, last 4 are context
        domain_onehot = x[:, :, :self.n_domains * 2]
        context_feats = x[:, :, self.n_domains * 2:]
        
        # Embed both paths
        emb_input = self.input_embed(context_feats)
        emb_exercise = torch.relu(self.exercise_embed(domain_onehot))
        
        # Concatenate and pass to LSTM
        lstm_in = torch.cat([emb_input, emb_exercise], dim=-1)
        lstm_out, _ = self.lstm(lstm_in)
        
        return self.output_layer(self.layer_norm(lstm_out[:, -1, :]))

    def get_knowledge_state(self, x):
        with torch.no_grad(): return torch.sigmoid(self.forward(x))

# ─────────────────────────────────────────────────────────
# SINGLETON LOADER
# ─────────────────────────────────────────────────────────
_dkt_model_instance = None
_domain_to_idx = None

def load_dkt_model():
    global _dkt_model_instance, _domain_to_idx
    
    if _dkt_model_instance is not None:
        return _dkt_model_instance, _domain_to_idx
        
    weights_path = os.path.join(MODEL_DATA_PATH, DKT_WEIGHTS_FILENAME)
    meta_path = os.path.join(MODEL_DATA_PATH, "dkt_meta_v4.json")
    
    if not os.path.exists(weights_path):
        logger.warning(f"DKT weights ({DKT_WEIGHTS_FILENAME}) not found in {MODEL_DATA_PATH}")
        return None, None
        
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            _domain_to_idx = meta.get("domain_to_idx", {d: i for i, d in enumerate(DOMAINS)})
        else:
            _domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
            
        model = EADKTModel(input_size=DKT_INPUT_SIZE, n_domains=len(DOMAINS), config=DKT_CONFIG)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.eval()
        
        _dkt_model_instance = model
        logger.info(f"✅ DKT Model ({DKT_WEIGHTS_FILENAME}) loaded successfully.")
        return _dkt_model_instance, _domain_to_idx
        
    except Exception as e:
        logger.error(f"Failed to load DKT model: {e}")
        return None, None

# ─────────────────────────────────────────────────────────
# ENCODING & SEQUENCE BUILDING
# ─────────────────────────────────────────────────────────
def encode_interaction(domain_idx: int, correct: bool, fatigue: float, frustration: float, session_pos: float, overall_progress: float) -> np.ndarray:
    """Encodes a single attempt into a 20-dim numpy vector."""
    vec = np.zeros(DKT_INPUT_SIZE, dtype=np.float32)
    
    # Indices 0-7: correct domain one-hot
    if correct:
        vec[domain_idx] = 1.0
    # Indices 8-15: incorrect domain one-hot
    else:
        vec[8 + domain_idx] = 1.0
        
    # Indices 16-19: Context features
    vec[16] = float(fatigue)
    vec[17] = float(frustration)
    vec[18] = float(session_pos)
    vec[19] = float(overall_progress)
    
    return vec

def build_student_sequence(attempts_list: List[Dict]) -> np.ndarray:
    """Converts list of attempt dicts into (len, 20) array."""
    seq_feats = []
    total_attempts = len(attempts_list)
    
    for i, a in enumerate(attempts_list):
        d_name = a.get("domain_id", "Attention")
        if d_name not in DOMAINS:
            d_name = "Attention"
        d_idx = DOMAINS.index(d_name)
        
        correct = bool(a.get("correct", False))
        fatigue = float(a.get("session_fatigue_index", 0.0))
        frustration = 1.0 if a.get("frustration_flag") else 0.0
        session_pos = float(a.get("session_position_normalized", i / max(1, total_attempts)))
        overall_progress = i / max(1, total_attempts)
        
        vec = encode_interaction(d_idx, correct, fatigue, frustration, session_pos, overall_progress)
        seq_feats.append(vec)
        
    return np.array(seq_feats)

def compute_knowledge_state(attempts_list: List[Dict]) -> Dict[str, float]:
    """Runs DKT forward pass and returns {domain: mastery} dict."""
    model, _ = load_dkt_model()
    
    if model is None or len(attempts_list) < 5:
        return {d: 0.5 for d in DOMAINS}
        
    feats = build_student_sequence(attempts_list)
    
    # Pad or truncate to DKT_SEQ_LEN
    if len(feats) > DKT_SEQ_LEN:
        feats = feats[-DKT_SEQ_LEN:]
    else:
        pad_len = DKT_SEQ_LEN - len(feats)
        feats = np.pad(feats, ((pad_len, 0), (0, 0)), 'constant')
        
    x = torch.tensor([feats], dtype=torch.float32)
    
    with torch.no_grad():
        # Fixed: No domain_indices passed, they are extracted internally from x
        ks_tensor = model.get_knowledge_state(x)
        
    ks_array = ks_tensor.squeeze(0).numpy()
    return {DOMAINS[i]: float(max(0.0, min(1.0, val))) for i, val in enumerate(ks_array)}

# ─────────────────────────────────────────────────────────
# DATABASE UPSERT
# ─────────────────────────────────────────────────────────
def update_knowledge_states(student_id: str, session_id: Optional[str], db: Session) -> Dict[str, float]:
    """Queries attempts, computes DKT mastery, and upserts into knowledge_states table."""
    try:
        # 1. Fetch Student Severity (for sub-5 attempts fallback)
        student = db.query(Student).filter(Student.student_id == student_id).first()
        severity = (student.asd_severity or "moderate").lower() if student else "moderate"
        
        # 2. Fetch Attempts
        attempts = db.query(TaskAttempt).filter(
            TaskAttempt.student_id == student_id
        ).order_by(TaskAttempt.sequence_step.asc()).all()
        
        if len(attempts) < 5:
            logger.info(f"Student {student_id} has <5 attempts. Using default mastery based on severity.")
            if severity == "mild":
                base_mastery = 0.60
            elif severity == "severe":
                base_mastery = 0.35
            else:
                base_mastery = 0.50
            mastery_dict = {d: base_mastery for d in DOMAINS}
        else:
            attempts_raw = [{
                "domain_id": a.domain_id,
                "correct": a.correct,
                "session_fatigue_index": a.session_fatigue_index,
                "frustration_flag": a.frustration_flag,
                "session_position_normalized": a.session_position_norm
            } for a in attempts]
            
            mastery_dict = compute_knowledge_state(attempts_raw)
            
        # 3. Upsert into DB
        for domain_name, score in mastery_dict.items():
            domain_id = domain_name.lower().replace(" ", "_")
            
            db.execute(text("""
                INSERT INTO knowledge_states 
                (student_id, session_id, domain_id, domain_name, mastery_score, attempts_count, last_updated)
                VALUES (:student_id, :session_id, :domain_id, :domain_name, :mastery_score, :attempts_count, NOW())
                ON CONFLICT (student_id, domain_id) 
                DO UPDATE SET 
                    mastery_score = EXCLUDED.mastery_score,
                    attempts_count = EXCLUDED.attempts_count,
                    session_id = EXCLUDED.session_id,
                    last_updated = NOW()
            """), {
                "student_id": student_id,
                "session_id": session_id,
                "domain_id": domain_id,
                "domain_name": domain_name,
                "mastery_score": score,
                "attempts_count": len(attempts)
            })
            
        db.commit()
        logger.info(f"✅ Knowledge states updated for student {student_id}")
        return mastery_dict
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update knowledge states for {student_id}: {e}")
        # Return safe defaults
        return {d: 0.5 for d in DOMAINS}