import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
import torch.nn as pd_nn

# For deep learning
import sys
sys.path.append('/Users/gyuminkang/Desktop/m5-forecasting-accuracy/scripts')
from binary_ensemble_v10 import LstmModel, GruModel, Cnn1dModel, train_deep_model, predict_deep_model, SequenceDataset
from torch.utils.data import DataLoader

def main():
    data_dir = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy'
    df = pd.read_parquet(os.path.join(data_dir, 'data', 'processed', 'features_data_v3.parquet'))
    
    # Filter for CA_1 only to make it fast
    df = df[df['store_id'] == 'CA_1'].copy()
    
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
    
    # Create missing date features
    df['wday'] = df['date'].dt.dayofweek.astype(np.int8)
    df['month'] = df['date'].dt.month.astype(np.int8)
    df['year'] = df['date'].dt.year.astype(np.int16)
    
    # Advanced Target Encoding (V9/V10)
    dept_wday_mean = df[df['d_int'] < 1914].groupby(['dept_id', 'wday'])['sales'].mean().reset_index()
    dept_wday_mean.rename(columns={'sales': 'dept_wday_mean'}, inplace=True)
    df = df.merge(dept_wday_mean, on=['dept_id', 'wday'], how='left')

    item_event_mean = df[df['d_int'] < 1914].groupby(['item_id', 'event_name_1'])['sales'].mean().reset_index()
    item_event_mean.rename(columns={'sales': 'item_event_mean'}, inplace=True)
    df = df.merge(item_event_mean, on=['item_id', 'event_name_1'], how='left')

    df['dept_wday_mean'] = df['dept_wday_mean'].fillna(0)
    df['item_event_mean'] = df['item_event_mean'].fillna(0)
    
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
        
    s_train = df[df['d_int'] < 1914]
    s_val = df[(df['d_int'] >= 1914) & (df['d_int'] <= 1941)]
    
    y_tr_clf = (s_train['sales'] > 0).astype(np.int8)
    X_tr = s_train[features].fillna(0)
    X_va = s_val[features].fillna(0)
    
    # 1. Train Single Model (LightGBM equivalent -> HistGradientBoosting)
    print("Training Single Model...")
    # Map column indices for categorical features: 'dept_id' is 1, 'cat_id' is 2, 'event_name_1' is 6, 'event_type_1' is 7
    clf_single = HistGradientBoostingClassifier(random_state=42, categorical_features=[1, 2, 6, 7])
    clf_single.fit(X_tr, y_tr_clf)
    single_probs = clf_single.predict_proba(X_va)[:, 1]
    
    # 2. Train V10 Models (3 Hist + 3 Deep)
    print("Training V10 Tabular Models...")
    v10_probs = np.zeros((6, len(X_va)), dtype=np.float32)
    for i in range(3):
        clf = HistGradientBoostingClassifier(random_state=42+i, categorical_features=[1, 2, 6, 7])
        clf.fit(X_tr, y_tr_clf)
        v10_probs[i] = clf.predict_proba(X_va)[:, 1]
        
    print("Training V10 Deep Models...")
    # Normalize
    s_df_scaled = df[['id', 'd_int', 'sales'] + features].copy()
    s_df_scaled.fillna(0, inplace=True)
    for col in features:
        if col not in categorical_features:
            s_df_scaled[col] = (s_df_scaled[col] - s_df_scaled[col].mean()) / (s_df_scaled[col].std() + 1e-8)
            
    train_deep_df = s_df_scaled[s_df_scaled['d_int'] < 1914]
    val_deep_df = s_df_scaled[(s_df_scaled['d_int'] >= 1914-28) & (s_df_scaled['d_int'] <= 1941)]
    
    def create_sequences(d_df):
        pivot = d_df.pivot(index='id', columns='d_int', values=features).fillna(0)
        arr = pivot.values.reshape(len(pivot.index), len(features), -1).transpose(0, 2, 1)
        labels = d_df.pivot(index='id', columns='d_int', values='sales').fillna(0).values
        labels = (labels > 0).astype(np.float32)
        X_seq, y_seq = [], []
        for i in range(arr.shape[1] - 28):
            X_seq.append(arr[:, i:i+28, :])
            y_seq.append(labels[:, i+28])
        return np.concatenate(X_seq, axis=0), np.concatenate(y_seq, axis=0)
        
    X_seq_tr, y_seq_tr = create_sequences(train_deep_df)
    X_seq_va, _ = create_sequences(val_deep_df)
    
    train_loader = DataLoader(SequenceDataset(X_seq_tr, y_seq_tr), batch_size=512, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_seq_va, np.zeros(len(X_seq_va))), batch_size=512, shuffle=False)
    
    input_dim = len(features)
    lstm = train_deep_model(LstmModel(input_dim), train_loader, epochs=1)
    gru = train_deep_model(GruModel(input_dim), train_loader, epochs=1)
    cnn = train_deep_model(Cnn1dModel(input_dim), train_loader, epochs=1)
    
    v10_probs[3] = predict_deep_model(lstm, val_loader)
    v10_probs[4] = predict_deep_model(gru, val_loader)
    v10_probs[5] = predict_deep_model(cnn, val_loader)
    
    # Power Averaging for V10
    v10_avg_probs = np.mean(v10_probs ** 2.0, axis=0)
    
    # 3. Extract specific item
    s_val['single_prob'] = single_probs
    s_val['v10_prob'] = v10_avg_probs
    s_val['actual_sold'] = (s_val['sales'] > 0).astype(int)
    
    item_df = s_val[s_val['id'] == 'HOBBIES_1_001_CA_1_evaluation'].copy()
    
    out_df = item_df[['date', 'actual_sold', 'single_prob', 'v10_prob']]
    out_path = os.path.join(data_dir, 'results', 'sample_item_probs.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")

if __name__ == '__main__':
    main()
