import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    for c in int_cols:
        c_min = df[c].min()
        c_max = df[c].max()
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[c] = df[c].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[c] = df[c].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[c] = df[c].astype(np.int32)
    return df

def main():
    data_dir = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy'
    
    print("1. V2 피처 데이터(features_data_v2.parquet)를 베이스로 불러옵니다...")
    df = pd.read_parquet(os.path.join(data_dir, 'data', 'processed', 'features_data_v2.parquet'))
    
    print("2. ID와 날짜 순으로 정렬합니다...")
    df.sort_values(['id', 'date'], inplace=True)
    
    print("3. 피처 추가 [연간 계절성]: 정확히 1년 전(364일 전) 동일 요일 판매량...")
    grouped_sales = df.groupby('id')['sales']
    df['sales_lag_364'] = grouped_sales.shift(364).astype(np.float32)
    
    print("4. 피처 추가 [타겟 인코딩]: 이 상품은 특정 요일에 보통 몇 개 팔릴까? (훈련 데이터 기준)...")
    # 누수를 막기 위해 훈련 기간(d <= 1913) 데이터만 사용해서 통계를 냅니다
    df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
    df['wday'] = df['date'].dt.dayofweek.astype(np.int8)
    
    train_mask = df['d_int'] <= 1913
    train_df = df[train_mask]
    
    # 1) 상품+요일별 평균 판매량
    item_wday_means = train_df.groupby(['item_id', 'wday'])['sales'].mean().reset_index(name='item_wday_mean')
    df = df.merge(item_wday_means, on=['item_id', 'wday'], how='left')
    
    # 2) 예외 처리: 데이터가 없는 경우 상품 전체 평균으로 채움
    item_means = train_df.groupby('item_id')['sales'].mean().reset_index(name='item_mean')
    df = df.merge(item_means, on='item_id', how='left')
    df['item_wday_mean'] = df['item_wday_mean'].fillna(df['item_mean']).fillna(0).astype(np.float32)
    
    # 필요없는 컬럼 삭제
    df.drop(['item_mean', 'd_int', 'wday'], axis=1, inplace=True)
    
    print("5. Downcast 최적화 진행...")
    df = downcast_dtypes(df)
    
    print(f"최종 생성된 V3 데이터 형태: {df.shape}")
    
    print("6. features_data_v3.parquet 로 저장합니다...")
    output_path = os.path.join(data_dir, 'data', 'processed', 'features_data_v3.parquet')
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    main()
