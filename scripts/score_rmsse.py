import pandas as pd
import numpy as np
import os

def main():
    data_dir = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy'
    
    # 1. V4 앙상블 결과 로드
    preds = pd.read_csv(os.path.join(data_dir, 'results', 'current_best_ensemble_v4.csv'))
    preds['d_int'] = preds['d'].apply(lambda x: int(x.split('_')[1]))
    
    # 2. 원본 데이터 로드 (과거 변동성 Denominator 계산 위함)
    train_sales = pd.read_csv(os.path.join(data_dir, 'data', 'raw', 'sales_train_evaluation.csv'))
    
    d_cols = [c for c in train_sales.columns if c.startswith('d_')]
    train_data = train_sales[['id'] + d_cols].copy()
    
    print("캐글 채점 봇 가동! RMSSE 점수를 계산합니다...")
    print("과거 판매량 변동폭(Denominator) 분석 중...")
    
    # 분모 계산: 과거 시계열의 차분 제곱 평균
    train_values = train_data[d_cols].values
    
    denominators = []
    for i in range(len(train_values)):
        series = train_values[i]
        nonzero_idx = np.where(series > 0)[0]
        if len(nonzero_idx) == 0:
            denominators.append(1.0)
        else:
            first_nonzero = nonzero_idx[0]
            active_series = series[first_nonzero:]
            if len(active_series) < 2:
                denominators.append(1.0)
            else:
                diff = np.diff(active_series)
                denom = np.mean(diff ** 2)
                denominators.append(denom if denom > 0 else 1.0)
                
    denom_map = dict(zip(train_data['id'], denominators))
    
    print("예측 오차(Numerator) 계산 중...")
    # 분자 계산: 평가 기간(d_1914 ~ d_1941)의 오차 제곱 평균
    val_data = preds[(preds['d_int'] >= 1914) & (preds['d_int'] <= 1941)].copy()
    val_data['sq_error'] = (val_data['sales'] - val_data['simple_avg_pred']) ** 2
    
    mse_df = val_data.groupby('id')['sq_error'].mean().reset_index()
    mse_df['denominator'] = mse_df['id'].map(denom_map)
    mse_df['rmsse'] = np.sqrt(mse_df['sq_error'] / mse_df['denominator'])
    
    final_score = mse_df['rmsse'].mean()
    
    print("\n=====================================")
    print("🏆 로컬 자체 모의고사 결과 (RMSSE)")
    print("=====================================")
    print(f"최종 평균 RMSSE 점수: {final_score:.4f}")
    print("※ 참고: M5 대회 상위권의 공식 WRMSSE 점수는 0.50 ~ 0.55 수준입니다.")
    print("=====================================\n")

if __name__ == "__main__":
    main()
