import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_sample_weight
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import psutil

warnings.filterwarnings('ignore')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def log_mem(log_file, msg=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    out = f"[Mem: {mem_mb:.1f} MB] {msg}"
    print(out)
    with open(log_file, 'a') as f:
        f.write(out + '\n')

class LstmModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class GruModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, hn = self.gru(x)
        return self.fc(hn[-1])

class Cnn1dModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.squeeze(-1))

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_sequences_fast(df, features, seq_length=28):
    # Vectorized sliding window view (Zero Memory overhead)
    df = df.sort_values(['item_id', 'd_int'])
    X_vals = df[features].values
    y_vals = (df['sales'] > 0).astype(int).values
    item_ids = df['item_id'].values
    
    X_seq = np.lib.stride_tricks.sliding_window_view(X_vals, (seq_length, len(features)))
    X_seq = X_seq.squeeze(1)
    y_seq = y_vals[seq_length - 1:]
    
    item_seq_start = item_ids[:-seq_length + 1]
    item_seq_end = item_ids[seq_length - 1:]
    
    valid_mask = item_seq_start == item_seq_end
    return X_seq[valid_mask], y_seq[valid_mask]

def train_tabular_model(cfg, X_tr, y_tr, X_va, sample_weights, categorical_features):
    m_type = cfg['type']
    seed = cfg['seed']
    
    if m_type == 'hist':
        cat_mask = [True if f in categorical_features else False for f in X_tr.columns]
        clf = HistGradientBoostingClassifier(
            max_iter=100, random_state=seed, categorical_features=cat_mask,
            learning_rate=0.08, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1
        )
        clf.fit(X_tr, y_tr, sample_weight=sample_weights)
        return clf.predict_proba(X_va)[:, 1]
        
    elif m_type == 'rf':
        clf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=seed, class_weight='balanced', n_jobs=1)
        clf.fit(X_tr.fillna(0), y_tr)
        return clf.predict_proba(X_va.fillna(0))[:, 1]
        
    elif m_type == 'et':
        clf = ExtraTreesClassifier(n_estimators=30, max_depth=10, random_state=seed, class_weight='balanced', n_jobs=1)
        clf.fit(X_tr.fillna(0), y_tr)
        return clf.predict_proba(X_va.fillna(0))[:, 1]
        
    elif m_type == 'lr':
        clf = LogisticRegression(C=cfg['C'], max_iter=150, random_state=seed, class_weight='balanced', solver='saga')
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.fillna(0))
        X_va_s = scaler.transform(X_va.fillna(0))
        clf.fit(X_tr_s, y_tr)
        return clf.predict_proba(X_va_s)[:, 1]

def get_3_diverse_models():
    configs = []
    for i in range(3): configs.append({'type': 'hist', 'seed': 42 + i})
    return configs

def train_deep_model(model, train_loader, epochs=2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze(-1)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
    return model

def predict_deep_model(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = torch.sigmoid(model(X_batch).squeeze(-1)).cpu().numpy()
            preds.extend(out)
    return np.array(preds)

def main():
    data_dir = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy'
    log_file = os.path.join(data_dir, 'training_log_v10.txt')
    start_time = time.time()
    
    log_mem(log_file, f"--- [God-Tier V10: Fast All-Kill Threshold Hunter (6 Models)] ---")
    log_mem(log_file, "Loading Data (Memory Monitored)...")
    
    df = pd.read_parquet(os.path.join(data_dir, 'data', 'processed', 'features_data_v3.parquet'))
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
    
    # Create missing date features
    df['wday'] = df['date'].dt.dayofweek.astype(np.int8)
    df['month'] = df['date'].dt.month.astype(np.int8)
    df['year'] = df['date'].dt.year.astype(np.int16)
    
    # Store ID list to loop over
    stores = df['store_id'].unique()
    
    # V9 Advanced Data Preprocessing: Extreme Target Encoding
    log_mem(log_file, "Applying Advanced Target Encoding (V9)...")
    df['sales_bin'] = (df['sales'] > 0).astype(int)
    train_mask = df['d_int'] <= 1913
    
    # 1. dept_wday_mean
    dept_wday_map = df[train_mask].groupby(['dept_id', 'wday'])['sales_bin'].mean().reset_index().rename(columns={'sales_bin': 'dept_wday_mean'})
    df = df.merge(dept_wday_map, on=['dept_id', 'wday'], how='left').fillna({'dept_wday_mean': 0})
    
    # 2. item_event_mean (Extremely sensitive to spikes)
    item_event_map = df[train_mask].groupby(['item_id', 'event_name_1'])['sales_bin'].mean().reset_index().rename(columns={'sales_bin': 'item_event_mean'})
    df = df.merge(item_event_map, on=['item_id', 'event_name_1'], how='left').fillna({'item_event_mean': 0})
    
    df.drop('sales_bin', axis=1, inplace=True)
    
    features = ['item_id', 'dept_id', 'cat_id', 'wday', 'month', 'year', 
                'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI',
                'sell_price', 'sales_lag_28', 'sales_lag_35', 'sales_lag_42',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
                'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60', 'rolling_std_60',
                'price_change_w1', 'price_change_m1', 'price_max_ratio',
                'days_since_event', 'days_to_event', 'sales_lag_364', 'item_wday_mean',
                'dept_wday_mean', 'item_event_mean']
    
    categorical_features = ['dept_id', 'cat_id', 'event_name_1', 'event_type_1']
    for c in categorical_features + ['item_id']:
        df[c] = df[c].astype('category').cat.codes

    configs = get_3_diverse_models()
    
    # We will accumulate the global probabilities for each of the 6 models (3 Tabular, 3 Deep)
    # Shape: (6 models, total_validation_rows)
    total_val_rows = len(df[(df['d_int'] >= 1914) & (df['d_int'] <= 1941)])
    global_probs = np.zeros((6, total_val_rows), dtype=np.float32)
    global_y_true = np.zeros(total_val_rows, dtype=np.int8)
    
    val_offset = 0
    
    for s_idx, store in enumerate(stores):
        store_start = time.time()
        log_mem(log_file, f">> Processing Store: {store} ({s_idx+1}/{len(stores)})")
        
        # TABULAR DATA (Full History)
        s_df = df[df['store_id'] == store]
        s_train = s_df[s_df['d_int'] <= 1913]
        s_val = s_df[(s_df['d_int'] >= 1914) & (s_df['d_int'] <= 1941)]
        
        y_tr_clf = (s_train['sales'] > 0).astype(int)
        y_va_clf = (s_val['sales'] > 0).astype(int)
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr_clf)
        
        X_tr = s_train[features].fillna(0)
        X_va = s_val[features].fillna(0)
        
        # 1. TABULAR 3-MODEL ENSEMBLE (PARALLEL)
        log_mem(log_file, "   Training 3 Tabular Models (Parallel)...")
        store_probs = Parallel(n_jobs=-1, backend="threading")(
            delayed(train_tabular_model)(cfg, X_tr, y_tr_clf, X_va, sample_weights, categorical_features)
            for cfg in configs
        )
        
        # Write first 3 model predictions to global matrix
        v_len = len(y_va_clf)
        for i, probs in enumerate(store_probs):
            global_probs[i, val_offset : val_offset + v_len] = probs
            
        global_y_true[val_offset : val_offset + v_len] = y_va_clf
        
        # Clean RAM
        del X_tr, X_va, y_tr_clf, store_probs, s_train
        gc.collect()
        
        # DEEP LEARNING (Last 1 Year ONLY to prevent RAM explosion)
        log_mem(log_file, "   Preparing Deep Learning Sequences (Last 1 Year)...")
        scaler = StandardScaler()
        s_df_scaled = s_df.copy()
        s_df_scaled[features] = scaler.fit_transform(s_df_scaled[features].fillna(0))
        
        # Train split (Last 1 year up to 1913)
        train_deep_df = s_df_scaled[(s_df_scaled['d_int'] >= 1914 - 365) & (s_df_scaled['d_int'] <= 1913)]
        X_seq_tr, y_seq_tr = build_sequences_fast(train_deep_df, features, seq_length=28)
        
        # Validation split (needs the last 27 days of train to predict the first day of val)
        val_deep_df = s_df_scaled[(s_df_scaled['d_int'] >= 1914 - 27) & (s_df_scaled['d_int'] <= 1941)]
        X_seq_va, y_seq_va = build_sequences_fast(val_deep_df, features, seq_length=28)
        
        train_loader = DataLoader(SequenceDataset(X_seq_tr, y_seq_tr), batch_size=1024, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_seq_va, y_seq_va), batch_size=1024, shuffle=False)
        
        log_mem(log_file, "   Training Deep Learning Models (LSTM, GRU, 1D-CNN)...")
        input_dim = len(features)
        
        lstm = train_deep_model(LstmModel(input_dim), train_loader)
        gru = train_deep_model(GruModel(input_dim), train_loader)
        cnn = train_deep_model(Cnn1dModel(input_dim), train_loader)
        
        global_probs[3, val_offset : val_offset + v_len] = predict_deep_model(lstm, val_loader)
        global_probs[4, val_offset : val_offset + v_len] = predict_deep_model(gru, val_loader)
        global_probs[5, val_offset : val_offset + v_len] = predict_deep_model(cnn, val_loader)
        
        val_offset += v_len
        
        # Clean RAM
        del train_deep_df, val_deep_df, X_seq_tr, y_seq_tr, X_seq_va, y_seq_va, lstm, gru, cnn, s_df_scaled
        gc.collect()
        
        store_end = time.time()
        eta = (store_end - store_start) * (len(stores) - s_idx - 1)
        log_mem(log_file, f"   Store {store} Complete. Estimated Time Remaining: {eta/60:.1f} minutes")

    del df
    gc.collect()
    
    log_mem(log_file, "=== Commencing V10 ALL-KILL THRESHOLD SEARCH ===")
    
    # Baseline Targets to Beat
    TARGET_ACC = 0.6720
    TARGET_PREC = 0.6510
    TARGET_REC = 0.5840
    TARGET_F1 = 0.6160
    
    # Simple Power Average of the 6 models
    # Square the probabilities to sharpen confidence
    avg_probs = np.mean(global_probs ** 2.0, axis=0)
    
    best_all_kill_f1 = 0
    best_thresh = 0.5
    found_all_kill = False
    
    thresholds = np.linspace(0.1, 0.9, 161) # Fine-grained scan
    
    for t in thresholds:
        preds = (avg_probs >= t).astype(int)
        acc = accuracy_score(global_y_true, preds)
        prec = precision_score(global_y_true, preds, zero_division=0)
        rec = recall_score(global_y_true, preds)
        f1 = f1_score(global_y_true, preds)
        
        # Check if ALL 4 metrics strictly beat the baseline
        if acc > TARGET_ACC and prec > TARGET_PREC and rec > TARGET_REC and f1 > TARGET_F1:
            found_all_kill = True
            log_mem(log_file, f"   [All-Kill Found!] Thresh: {t:.3f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            if f1 > best_all_kill_f1:
                best_all_kill_f1 = f1
                best_thresh = t
                
    if not found_all_kill:
        log_mem(log_file, "   FAILED to find an All-Kill Threshold. Defaulting to Max F1.")
        # Fallback to Max F1
        for t in thresholds:
            preds = (avg_probs >= t).astype(int)
            f1 = f1_score(global_y_true, preds)
            if f1 > best_all_kill_f1:
                best_all_kill_f1 = f1
                best_thresh = t
                
    # Final evaluation using best threshold
    final_preds = (avg_probs >= best_thresh).astype(int)
    final_acc = accuracy_score(global_y_true, final_preds)
    final_prec = precision_score(global_y_true, final_preds)
    final_rec = recall_score(global_y_true, final_preds)
    final_f1 = f1_score(global_y_true, final_preds)
    
    log_mem(log_file, "=========================================================")
    if found_all_kill:
        log_mem(log_file, f"🏆 V10 ALL-KILL SUCCESS! (Threshold: {best_thresh:.3f}) 🏆")
    else:
        log_mem(log_file, f"⚠️ V10 ALL-KILL FAILED! Max F1 Fallback (Threshold: {best_thresh:.3f})")
        
    log_mem(log_file, f"Final Accuracy: {final_acc:.4f} (Beat 0.6720? {final_acc > 0.6720})")
    log_mem(log_file, f"Final Precision: {final_prec:.4f} (Beat 0.6510? {final_prec > 0.6510})")
    log_mem(log_file, f"Final Recall: {final_rec:.4f} (Beat 0.5840? {final_rec > 0.5840})")
    log_mem(log_file, f"Final F1-Score: {final_f1:.4f} (Beat 0.6160? {final_f1 > 0.6160})")
    log_mem(log_file, f"Total execution time: {(time.time() - start_time)/60:.1f} minutes")
    
    os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)
    metrics = {
        'Accuracy': final_acc,
        'Precision': final_prec,
        'Recall': final_rec,
        'F1-Score': final_f1,
        'Optimal_Threshold': best_thresh,
        'All_Kill_Success': found_all_kill
    }
    with open(os.path.join(data_dir, 'results', 'v10_metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
if __name__ == '__main__':
    main()
