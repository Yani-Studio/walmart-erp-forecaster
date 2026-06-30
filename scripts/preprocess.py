import os
import gc
import pandas as pd
import numpy as np

def downcast_dtypes(df):
    """
    메모리 사용량을 줄이기 위해 데이터 타입을 최적화합니다.
    """
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
    
    print("1. 데이터를 로드합니다...")
    # 달력 데이터 로드 및 downcast
    calendar = pd.read_csv(os.path.join(data_dir, 'calendar.csv'))
    calendar = downcast_dtypes(calendar)
    
    # 가격 데이터 로드 및 downcast
    prices = pd.read_csv(os.path.join(data_dir, 'sell_prices.csv'))
    prices = downcast_dtypes(prices)
    
    # 평가용 타겟 데이터 로드 (validation 보다 최신 데이터)
    train = pd.read_csv(os.path.join(data_dir, 'sales_train_evaluation.csv'))
    print(f"원본 Train 데이터 형태: {train.shape}")
    
    # 식별자 열 추출
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    print("2. Wide to Long 변환을 수행합니다 (Melt)...")
    # 모든 날짜 칼럼을 행으로 늘어뜨림
    train_melt = pd.melt(train, 
                         id_vars=id_vars, 
                         var_name='d', 
                         value_name='sales')
    
    print(f"Melt 이후 데이터 형태: {train_melt.shape}")
    
    # train 데이터프레임 제거하여 메모리 확보
    del train
    gc.collect()
    
    # 범주형(Categorical) 변수 최적화
    print("3. 범주형 변수를 Category 타입으로 변환합니다...")
    for col in id_vars:
        train_melt[col] = train_melt[col].astype('category')
        
    train_melt = downcast_dtypes(train_melt)
    
    print("4. 달력(Calendar) 데이터를 결합합니다...")
    # 불필요한 칼럼 제외 후 결합 (메모리 절약)
    calendar_cols_to_keep = ['date', 'wm_yr_wk', 'd', 'event_name_1', 'event_type_1', 
                             'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
    calendar_subset = calendar[calendar_cols_to_keep]
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        calendar_subset.loc[:, col] = calendar_subset[col].astype('category')
    
    train_melt = pd.merge(train_melt, calendar_subset, on='d', copy=False)
    
    del calendar, calendar_subset
    gc.collect()
    
    print("5. 가격(Prices) 데이터를 결합합니다...")
    # prices 데이터를 category형으로 변환하여 메모리 압축
    prices['store_id'] = prices['store_id'].astype('category')
    prices['item_id'] = prices['item_id'].astype('category')
    
    train_melt = pd.merge(train_melt, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left', copy=False)
    
    del prices
    gc.collect()
    
    print("6. 결측치 정리: 출시되지 않은 상품의 과거 데이터를 삭제합니다...")
    # sell_price가 NaN이라는 것은 아직 상점에 출시되지 않았음을 의미하므로 불필요한 0 데이터입니다.
    initial_rows = len(train_melt)
    train_melt.dropna(subset=['sell_price'], inplace=True)
    train_melt.reset_index(drop=True, inplace=True)
    final_rows = len(train_melt)
    
    print(f"삭제된 행: {initial_rows - final_rows:,} (출시 전 데이터 제거로 인한 용량 절약)")
    print(f"최종 전처리된 데이터 형태: {train_melt.shape}")
    
    print("7. Parquet 파일로 저장합니다...")
    output_path = os.path.join(data_dir, 'preprocessed_data.parquet')
    
    # 범주형 데이터로 인해 parquet 저장 시 에러를 막기 위한 조치
    # category 타입이 유지된 상태로 저장
    train_melt.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    main()
