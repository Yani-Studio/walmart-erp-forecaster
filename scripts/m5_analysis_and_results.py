# %% [markdown]
# # M5 Forecasting - Accuracy: Data & Performance Analysis
# 이 노트북은 원본 데이터의 구조와 V4(최종 진화형) 앙상블 모델의 최종 성능을 시각화합니다.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'AppleGothic' # Mac용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 50)

data_dir = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy'

# %% [markdown]
# ## 1. 원본 데이터 구조 시각화 (Total Sales Over Time)

# %%
# 과거 1941일치 전체 판매량 데이터 로드
sales = pd.read_csv(os.path.join(data_dir, 'data', 'raw', 'sales_train_evaluation.csv'))
calendar = pd.read_csv(os.path.join(data_dir, 'data', 'raw', 'calendar.csv'))

# 날짜 컬럼(d_1 ~ d_1941)만 추출하여 총합 계산
d_cols = [c for c in sales.columns if c.startswith('d_')]
total_sales_per_day = sales[d_cols].sum(axis=0)

# 달력 데이터와 결합하여 실제 날짜 맵핑
total_sales_df = pd.DataFrame({'d': d_cols, 'sales': total_sales_per_day.values})
total_sales_df = total_sales_df.merge(calendar[['d', 'date']], on='d', how='left')
total_sales_df['date'] = pd.to_datetime(total_sales_df['date'])

plt.figure(figsize=(15, 5))
plt.plot(total_sales_df['date'], total_sales_df['sales'], color='#2ca02c', alpha=0.8, linewidth=1)
plt.title('전체 매장 총 판매량 추이 (Day 1 ~ Day 1941)', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# %% [markdown]
# 크리스마스마다 휴점(0)이 발생하는 거대한 계절성 패턴과 3년간 꾸준히 우상향하는 트렌드를 볼 수 있습니다.

# %% [markdown]
# ## 2. 매장별 예측 성능(RMSE) 비교 (Validation: 1914 ~ 1941)

# %%
# V4 앙상블 예측 결과 로드
preds = pd.read_csv(os.path.join(data_dir, 'results', 'current_best_ensemble_v4.csv'))
preds['store_id'] = preds['id'].str.extract(r'([A-Z]{2}_\d)')

# 개별 행 RMSE 계산 (오차 제곱)
preds['sq_error'] = (preds['sales'] - preds['simple_avg_pred'])**2

# 매장별 평균 오차(RMSE) 집계
store_rmse = preds.groupby('store_id')['sq_error'].mean().apply(np.sqrt).reset_index()
store_rmse.rename(columns={'sq_error': 'rmse'}, inplace=True)
store_rmse = store_rmse.sort_values('rmse')

plt.figure(figsize=(12, 6))
sns.barplot(x='store_id', y='rmse', data=store_rmse, palette='viridis')
plt.title('매장별 예측 오차(RMSE) 비교 - V4 앙상블', fontsize=16, fontweight='bold')
plt.ylabel('RMSE (낮을수록 좋음)')
plt.xlabel('매장 (Store)')

for index, row in enumerate(store_rmse.itertuples()):
    plt.text(index, row.rmse + 0.05, f'{row.rmse:.2f}', color='black', ha="center")
    
plt.tight_layout()
plt.show()

# %% [markdown]
# CA_4 매장이 오차 1.61로 가장 예측이 잘 맞았고, WI_2 매장이 가장 변동성이 커 예측이 어려웠음을 알 수 있습니다.

# %% [markdown]
# ## 3. 실제 판매량 vs 모델 예측량 오버레이 시각화 (특정 상품)

# %%
# 예시 상품: HOBBIES_1_001_CA_1
sample_id = 'HOBBIES_1_001_CA_1_evaluation'
sample_data = preds[preds['id'] == sample_id].copy()
sample_data = sample_data.merge(calendar[['d', 'date']], on='d', how='left')
sample_data['date'] = pd.to_datetime(sample_data['date'])
sample_data = sample_data.sort_values('date')

plt.figure(figsize=(14, 5))
plt.plot(sample_data['date'], sample_data['sales'], label='실제 판매량 (Actual)', color='blue', marker='o', alpha=0.6)
plt.plot(sample_data['date'], sample_data['simple_avg_pred'], label='V4 앙상블 예측 (Predicted)', color='red', linestyle='--', marker='x', linewidth=2)
plt.title(f'예측 vs 실제 판매량 비교 ({sample_id})', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 모델이 간헐적으로 0, 1, 2개가 팔리는 희소한 데이터에 대해 아주 부드럽고 보수적인 기댓값(약 0.6~0.7)을 내놓고 있습니다.
# 이는 캐글의 WRMSSE 평가 지표 하에서 가장 오차를 줄이는 극강의 '가중 평균' 최적화 형태입니다.
