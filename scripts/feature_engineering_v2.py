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
    
    print("1. 기존 피처 데이터(features_data.parquet)를 베이스로 불러옵니다...")
    df = pd.read_parquet(os.path.join(data_dir, 'features_data.parquet'))
    
    # 빠른 그룹화를 위해 정렬
    print("2. ID와 날짜 순으로 정렬합니다...")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['id', 'date'], inplace=True)
    
    print("3. 피처 추가 [가격 모멘텀]: 지난주/지난달 대비 가격 변동률, 역대 최고가 대비 비율...")
    # 그룹화 객체 재사용
    grouped_price = df.groupby('id')['sell_price']
    
    # 지난주(Lag 7) 가격 대비 변동률
    df['price_change_w1'] = df['sell_price'] / grouped_price.shift(7)
    # 지난달(Lag 28) 가격 대비 변동률
    df['price_change_m1'] = df['sell_price'] / grouped_price.shift(28)
    # 역대 최고가 대비 할인율(비율)
    df['price_max_ratio'] = df['sell_price'] / grouped_price.transform('max')
    
    print("4. 피처 추가 [이벤트 D-Day]: 다음 행사까지 남은 일수, 지난 행사 후 지난 일수...")
    # 달력에서 고유한 날짜와 이벤트 정보만 추출
    calendar = df[['date', 'event_name_1']].drop_duplicates().sort_values('date').copy()
    calendar['is_event'] = calendar['event_name_1'].notna().astype(int)
    
    # days since last event
    calendar['days_since_event'] = 0
    last_event_idx = -9999
    days_since = []
    for i, val in enumerate(calendar['is_event']):
        if val == 1:
            last_event_idx = i
            days_since.append(0)
        else:
            if last_event_idx == -9999:
                days_since.append(9999) # 이전 이벤트가 없으면 큰 수
            else:
                days_since.append(i - last_event_idx)
    calendar['days_since_event'] = days_since
    
    # days to next event
    calendar['days_to_event'] = 0
    next_event_idx = 999999
    days_to = []
    for i, val in reversed(list(enumerate(calendar['is_event']))):
        if val == 1:
            next_event_idx = i
            days_to.append(0)
        else:
            if next_event_idx == 999999:
                days_to.append(9999) # 이후 이벤트가 없으면 큰 수
            else:
                days_to.append(next_event_idx - i)
    calendar['days_to_event'] = list(reversed(days_to))
    
    calendar = calendar[['date', 'days_since_event', 'days_to_event']]
    
    # 본 데이터에 머지
    df = pd.merge(df, calendar, on='date', how='left')
    
    # 결측치 처리 (fillna)
    df['price_change_w1'] = df['price_change_w1'].fillna(1.0).astype(np.float32)
    df['price_change_m1'] = df['price_change_m1'].fillna(1.0).astype(np.float32)
    df['price_max_ratio'] = df['price_max_ratio'].fillna(1.0).astype(np.float32)
    df['days_since_event'] = df['days_since_event'].astype(np.int16)
    df['days_to_event'] = df['days_to_event'].astype(np.int16)
    
    print("5. Downcast 최적화 진행...")
    df = downcast_dtypes(df)
    
    print(f"최종 생성된 V2 데이터 형태: {df.shape}")
    
    print("6. features_data_v2.parquet 로 저장합니다...")
    output_path = os.path.join(data_dir, 'features_data_v2.parquet')
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    main()
