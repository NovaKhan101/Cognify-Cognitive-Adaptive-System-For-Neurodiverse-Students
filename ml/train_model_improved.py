"""
COGNIFY - Phase 2: Improved Model Training (Capstone Version V3)
==============================================================
Run after: preprocess.py generates .npy files from cognify_synthetic_data.csv
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import math
import pickle
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("COGNIFY - Improved Model Training (V3 - Optimized)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ==================== CONFIG (REDUCED TO PREVENT OVERFITTING) ====================
CONFIG = {
    'sequence_length': 10,
    'hidden_size': 64,      # Reduced from 256!
    'num_layers': 2,        # Reduced from 3!
    'dropout': 0.3,
    'nhead': 4,
    'dim_feedforward': 128, # Reduced from 512!
    'batch_size': 64,
    'learning_rate': 0.001, # Increased slightly
    'epochs': 80,
    'patience': 15,
    'weight_decay': 1e-4,
}

# ==================== LOAD DATA ====================
print("\n📂 Loading preprocessed data...")

X_train = np.load('model_data/X_train.npy')
X_val   = np.load('model_data/X_val.npy')
X_test  = np.load('model_data/X_test.npy')
y_train = np.load('model_data/y_train.npy')
y_val   = np.load('model_data/y_val.npy')
y_test  = np.load('model_data/y_test.npy')

X_diff_train = np.load('model_data/X_diff_train.npy')
X_diff_val   = np.load('model_data/X_diff_val.npy')
X_diff_test  = np.load('model_data/X_diff_test.npy')
y_diff_train = np.load('model_data/y_diff_train.npy')
y_diff_val   = np.load('model_data/y_diff_val.npy')
y_diff_test  = np.load('model_data/y_diff_test.npy')

with open('model_data/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

n_features = feature_info['n_features']
print(f"   Features per timestep: {n_features}")
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==================== EXTRA STREAK FEATURES ====================
print("\n🔧 Adding streak features...")

def add_streak_features(X):
    n_samples, seq_len, n_feat = X.shape
    streaks = np.zeros((n_samples, seq_len, 2))
    for i in range(n_samples):
        correct_streak = 0
        wrong_streak = 0
        for t in range(seq_len):
            is_correct = X[i, t, 0]
            if is_correct > 0.5:
                correct_streak += 1
                wrong_streak = 0
            else:
                wrong_streak += 1
                correct_streak = 0
            streaks[i, t, 0] = correct_streak / 10.0
            streaks[i, t, 1] = wrong_streak / 10.0
    return np.concatenate([X, streaks], axis=2)

X_train_aug = add_streak_features(X_train)
X_val_aug   = add_streak_features(X_val)
X_test_aug  = add_streak_features(X_test)

X_diff_train_aug = add_streak_features(X_diff_train)
X_diff_val_aug   = add_streak_features(X_diff_val)
X_diff_test_aug  = add_streak_features(X_diff_test)

n_features_aug = X_train_aug.shape[2]
print(f"   Features after augmentation: {n_features_aug}")

# ==================== CLASS WEIGHTS ====================
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

# ==================== DATASETS ====================
def to_tensors(X, y, dtype_y=torch.float32):
    return TensorDataset(torch.FloatTensor(X), torch.tensor(y, dtype=dtype_y))

train_ds = to_tensors(X_train_aug, y_train)
val_ds   = to_tensors(X_val_aug, y_val)
test_ds  = to_tensors(X_test_aug, y_test)

diff_train_ds = to_tensors(X_diff_train_aug, y_diff_train, torch.long)
diff_val_ds   = to_tensors(X_diff_val_aug, y_diff_val, torch.long)
diff_test_ds  = to_tensors(X_diff_test_aug, y_diff_test, torch.long)

train_loader      = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader        = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
test_loader       = DataLoader(test_ds, batch_size=CONFIG['batch_size'])
diff_train_loader = DataLoader(diff_train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
diff_val_loader   = DataLoader(diff_val_ds, batch_size=CONFIG['batch_size'])
diff_test_loader  = DataLoader(diff_test_ds, batch_size=CONFIG['batch_size'])

# ==================== MODEL 1: IMPROVED LSTM (SHRUNK) ====================
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(ImprovedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = self.attention(lstm_out)
        context = (attn * lstm_out).sum(dim=1)
        return self.classifier(context)

# ==================== MODEL 2: TRANSFORMER (FIXED + SMALL) ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1)]

class CognifyTransformer(nn.Module):
    def __init__(self, input_size, nhead, dim_feedforward, num_layers, output_size, dropout=0.3):
        super(CognifyTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x))
        x = self.transformer(x)
        return self.classifier(self.pool(x.transpose(1, 2)).squeeze(-1))

# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs, patience, model_name, binary=True):
    print(f"\n🏋️  Training {model_name}...")
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss, train_preds, train_targets = 0, [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            if binary:
                loss = criterion(output.squeeze(), y_batch)
                preds = (torch.sigmoid(output.squeeze()) > 0.5).cpu().numpy()
            else:
                loss = criterion(output, y_batch)
                preds = output.argmax(dim=1).cpu().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(preds)
            train_targets.extend(y_batch.cpu().numpy())

        model.eval()
        val_loss, val_preds, val_targets = 0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                if binary:
                    loss = criterion(output.squeeze(), y_batch)
                    preds = (torch.sigmoid(output.squeeze()) > 0.5).cpu().numpy()
                else:
                    loss = criterion(output, y_batch)
                    preds = output.argmax(dim=1).cpu().numpy()
                val_loss += loss.item()
                val_preds.extend(preds)
                val_targets.extend(y_batch.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        val_acc   = accuracy_score(val_targets, val_preds)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f} | Time: {time.time()-start:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'model_data/best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                break

    print(f"   ✅ Best Val Accuracy: {best_val_acc:.3f}")
    return best_val_acc

def get_probs(model, X, batch_size=64):
    model.eval()
    all_probs = []
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for (X_batch,) in loader:
            output = model(X_batch.to(device))
            probs = torch.sigmoid(output.squeeze()).cpu().numpy()
            if np.isscalar(probs) or probs.ndim == 0: probs = np.array([probs])
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

# ==================== FIND OPTIMAL THRESHOLD ====================
def find_optimal_threshold(y_true, probs):
    best_acc = 0
    best_thresh = 0.5
    for thresh in np.arange(0.35, 0.65, 0.01):
        preds = (probs > thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc

# ==================== TRAIN LSTM ====================
print("\n" + "="*60)
print("MODEL 1: Improved Bidirectional LSTM")
print("="*60)

lstm_model = ImprovedLSTM(n_features_aug, CONFIG['hidden_size'], CONFIG['num_layers'], 1, CONFIG['dropout']).to(device)
print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}") # SHOULD BE ~200k, NOT 3.9M!

criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt_lstm = torch.optim.AdamW(lstm_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
sch_lstm = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_lstm, patience=5, factor=0.5)
train_model(lstm_model, train_loader, val_loader, criterion_weighted, opt_lstm, sch_lstm, CONFIG['epochs'], CONFIG['patience'], 'improved_lstm')
lstm_model.load_state_dict(torch.load('model_data/best_improved_lstm.pth'))

# ==================== TRAIN XGBOOST ====================
print("\n" + "="*60)
print("MODEL 2: Gradient Boosting")
print("="*60)

def flatten_sequences(X):
    return np.concatenate([X[:, -1, :], X.mean(axis=1), X.std(axis=1)], axis=1)

X_train_flat = flatten_sequences(X_train_aug)
X_val_flat   = flatten_sequences(X_val_aug)
X_test_flat  = flatten_sequences(X_test_aug)

gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, min_samples_split=20, subsample=0.8, random_state=42, verbose=0)
gb_model.fit(X_train_flat, y_train)
print(f"   ✅ Val Accuracy:  {accuracy_score(y_val, gb_model.predict(X_val_flat)):.3f}")
with open('model_data/gb_model.pkl', 'wb') as f: pickle.dump(gb_model, f)

# ==================== ENSEMBLE & OPTIMAL THRESHOLD ====================
print("\n" + "="*60)
print("ENSEMBLE: LSTM + GradientBoosting (Optimized Threshold)")
print("="*60)

lstm_probs_val = get_probs(lstm_model, X_val_aug)
gb_probs_val   = gb_model.predict_proba(X_val_flat)[:, 1]

lstm_probs_test = get_probs(lstm_model, X_test_aug)
gb_probs_test   = gb_model.predict_proba(X_test_flat)[:, 1]

# Find best weights AND best threshold on validation set
best_acc = 0
best_weights = (0.5, 0.5)
best_thresh = 0.5

for w1 in np.arange(0.3, 0.8, 0.1):
    w2 = round(1.0 - w1, 1)
    ensemble_probs_val = w1 * lstm_probs_val + w2 * gb_probs_val
    
    thresh, acc = find_optimal_threshold(y_val, ensemble_probs_val)
    if acc > best_acc:
        best_acc = acc
        best_weights = (w1, w2)
        best_thresh = thresh

print(f"   Optimal Weights: LSTM={best_weights[0]:.1f}, GB={best_weights[1]:.1f}")
print(f"   Optimal Threshold: {best_thresh:.2f} (Default is 0.50)")
print(f"   Optimized Val Accuracy: {best_acc:.3f}")

# Apply to test set
ensemble_probs_test = (best_weights[0] * lstm_probs_test + best_weights[1] * gb_probs_test)
ensemble_preds_test = (ensemble_probs_test > best_thresh).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_preds_test)

print(f"\n📊 Final Test Accuracy:")
print(f"   🏆 Optimized Ensemble: {ensemble_acc:.4f} ({ensemble_acc:.1%})")

# ==================== DIFFICULTY MODEL ====================
print("\n" + "="*60)
print("DIFFICULTY RECOMMENDER: Transformer")
print("="*60)

diff_transformer = CognifyTransformer(n_features_aug, 4, CONFIG['dim_feedforward'], 2, 5, CONFIG['dropout']).to(device)
opt_diff = torch.optim.AdamW(diff_transformer.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
sch_diff = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_diff, patience=5, factor=0.5)
train_model(diff_transformer, diff_train_loader, diff_val_loader, nn.CrossEntropyLoss(), opt_diff, sch_diff, CONFIG['epochs'], CONFIG['patience'], 'improved_diff_transformer', binary=False)
diff_transformer.load_state_dict(torch.load('model_data/best_improved_diff_transformer.pth'))
with torch.no_grad():
    diff_logits = diff_transformer(torch.FloatTensor(X_diff_test_aug).to(device))
    diff_preds_test = torch.softmax(diff_logits, dim=1).argmax(dim=1).cpu().numpy()
diff_acc = accuracy_score(y_diff_test, diff_preds_test)
print(f"\n📊 Difficulty Test Accuracy: {diff_acc:.4f} ({diff_acc:.1%})")

# ==================== SAVE ====================
print("\n💾 Saving results...")
os.makedirs('model_data', exist_ok=True)
results = {
    'optimal_threshold': float(best_thresh),
    'ensemble_weights': list(best_weights),
    'performance': {
        'ensemble_acc': float(ensemble_acc),
        'difficulty_acc': float(diff_acc)
    }
}
with open('model_data/improved_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"   🏆 Ensemble Accuracy:  {ensemble_acc:.1%}")
print(f"   🎯 Difficulty Model:   {diff_acc:.1%}")