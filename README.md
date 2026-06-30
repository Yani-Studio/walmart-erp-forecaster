<div align="center">
  <h1>🛒 Walmart ERP Forecaster <br><span style="font-size:0.7em; color:#4B0082;">(Deep Time-Series & Tabular Tree Hybrid Ensemble)</span></h1>
  <p><b>Walmart 4,700만 시계열 데이터 한계 돌파: 딥러닝 시계열과 머신러닝 트리의 궁극적인 하이브리드 앙상블 아키텍처</b></p>
  
  <p>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
    <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/LightGBM-0080FF?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM"></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  </p>
</div>

<br/>

## 📖 1. 프로젝트 개요 (Overview)

<div align="center">
  <img src="visualizations/03_Total_Sales_Trend.png" alt="M5 Total Sales Trend" width="100%" style="border-radius: 10px;">
</div>
<br/>

월마트(Walmart)의 1,941일 치(약 4,700만 건) 시계열 데이터를 바탕으로, 각 상품의 **수요 발생 여부(Demand Occurrence)** 를 예측하는 엔터프라이즈 재고 최적화(ERP) 솔루션입니다. 

상품이 며칠에 한 번씩만 팔리는 **희소한 수요(Intermittent Demand)** 와 80% 이상이 0으로 가득 찬 **Zero-inflated** 문제를 완벽하게 통제하기 위해, "이 상품이 오늘 팔릴 것인가?(Classification)"와 "팔린다면 몇 개가 팔릴 것인가?(Regression)"를 분리하여 예측하는 **Two-Stage Hurdle 모델 아키텍처**를 도입했습니다. 여기에 전통적인 횡단면 트리 머신러닝 모델과 시퀀스(시간 흐름)를 기억하는 **PyTorch 딥러닝 시계열 모델(LSTM, GRU, 1D-CNN)** 을 하이브리드로 결합했습니다. 나아가 총 53개의 모델 중 최고 성능을 내는 조합을 찾기 위해 **앙상블 전수조사(Exhaustive Ensemble Search)** 를 수행하여 성능의 한계치를 돌파했습니다.

---

## 📈 2. 시계열 분석: 앙상블의 극대화된 예민도 (Time Series Analysis)

동일한 평가지표로 채점되었음에도 두 모델의 예측 패턴은 완전히 다릅니다. 이는 타겟 인코딩과 시계열 트렌드 모델링을 통한 **예민도(Sensitivity) 파인튜닝** 덕분입니다.

<div align="center">
  <img src="visualizations/00_Actual_vs_Pred.png" width="80%">
</div>

> **🔥 앙상블의 놀라운 Spike 추적 능력**
> 단일 모델(LightGBM, 분홍색 점선)은 오차 페널티(RMSE)를 최소화하기 위해 안전한 평균값만 예측하는 **매우 보수적인 한계**를 보이며 실제 판매량의 급증(Spike)을 전혀 따라가지 못합니다. 
> 반면, **최종 앙상블(진보라색 실선)**은 딥러닝과 타겟 인코딩의 시너지를 통해 **예민도(Sensitivity)를 한계까지 끌어올리도록 파인튜닝**되었습니다. 그 결과, 3개, 5개씩 갑자기 팔리는 불확실한 수요 급증 구간에서도 몸을 사리지 않고 실제 판매량을 기가 막히게 추적하는 놀라운 핏(Fit)을 완성했습니다.

---

## 🏆 3. 핵심 비즈니스 성과 (Key Performance)

재고 관리와 마케팅 기회 비용의 완벽한 밸런스를 찾는 것은 유통 AI의 궁극적인 목표입니다. 최종 앙상블 아키텍처는 베이스라인 모델(단일 최고 성능 트리 모델) 대비 단 하나의 지표도 희생하지 않고 **'정확도, 정밀도, 재현율, F1-Score 4가지 지표를 모두 상회'** 하는 기적적인 성능을 달성했습니다.

<div align="center">
  <img src="visualizations/07_Classification_Metrics_Comparison.png" width="48%">
  <img src="visualizations/04_Hurdle_Confusion_Matrix.png" width="48%">
</div>

> **💡 단일 1등 모델 전 지표 압살 (Success)**
> 기존 앙상블 기법은 F1-Score나 정밀도 등 하나의 특화된 지표를 극대화하기 위해 필연적으로 다른 지표를 소폭 희생(Pareto Frontier)해야만 했습니다. 
> 그러나 **최종 앙상블**은 '극한의 타겟 인코딩(Extreme Target Encoding)'을 거친 정예 모델들을 결합한 뒤, 수학적으로 모든 지표가 단일 최고 모델을 압도하는 **안정적인 예측 기댓값**을 산출했습니다. 그 결과 정밀도를 전혀 잃지 않으면서도 **재현율을 대폭 끌어올리며**, 최종 F1-Score를 역대 최고치 수준으로 갱신했습니다.

---

## 🏗️ 4. 모델 아키텍처 및 데이터 구조 (Architecture & Data Structure)

단순한 머신러닝의 한계를 넘어, 고도화된 피처 엔지니어링과 가벼우면서도 치명적인 딥러닝 결합을 활용한 최첨단 아키텍처를 설계했습니다.

<div align="center">
  <img src="visualizations/02_Model_Architecture.png" width="48%">
  <img src="visualizations/01_M5_SQL_ERD.png" width="48%">
</div>

1. **Extreme Target Encoding (극한의 피처 엔지니어링)**
   - `dept_wday_mean`: 카테고리별/요일별 기준 판매 확률.
   - `item_event_mean`: 상품별 특수 이벤트(슈퍼볼, 추수감사절 등) 폭발적 판매 확률.
   - 타겟 누수(Data Leakage)를 완벽히 통제한 상태로, 딥러닝 모델에게 '강력한 정답의 힌트'를 피처로 쥐여주어 기초 체력(AUC)을 극대화했습니다.
   - Walmart ERP 시스템의 팩트(Fact) 테이블과 캘린더, 가격 정보가 1:N 관계로 조인(Join)되는 전형적인 Star Schema 데이터베이스 구조(우측 ERD 참고)를 기반으로 변수들을 집계했습니다.

2. **Lightweight Hybrid Training & Memory Optimization**
   - 가장 성능이 뛰어난 트리 모델(`HistGradientBoosting` 등)과 딥러닝 시계열 모델(`LSTM, GRU, 1D-CNN`)만을 엄선하여 훈련 속도를 획기적으로 단축(15분 내외)했습니다.
   - 특히 4,700만 건의 데이터를 딥러닝용 3차원 텐서(`Samples, 28, Features`)로 변환할 때 발생하는 막대한 RAM 메모리 폭발 현상을 방지하기 위해, Numpy의 `sliding_window_view`를 활용한 **Zero-copy 메모리 뷰 벡터화 연산**을 적용하여 OOM(Out Of Memory) 없이 쾌적한 학습 파이프라인을 구축했습니다.

---

## 📊 5. 앙상블 가중치 및 매장별 평가 (Ensemble Composition & RMSE)

어떤 모델들이 결합되어 이 놀라운 성과를 내었는지, 그리고 실제 월마트 10개 매장(Store)별로 오차율(RMSE)이 어떻게 나타나는지 확인합니다.

<div align="center">
  <img src="visualizations/06_Ensemble_Composition_Donut.png" width="48%">
  <img src="visualizations/05_Store_RMSE_Ranking.png" width="48%">
</div>

최종 아키텍처는 속도와 성능을 동시에 잡기 위해 군더더기를 덜어내고, 최고의 시너지 효과를 내는 Tabular 머신러닝 트리 3개와 PyTorch 딥러닝 3개의 완벽한 5:5 가중치 밸런스로 구성되었습니다. 매장별 오차율(RMSE) 또한 특정 매장에 치우치지 않고 매우 균일하고 안정적인 성능을 입증하고 있습니다.

> **💡 목적 함수 다각화 (Loss Function Blending)**
> 단순히 알고리즘만 섞은 것이 아닙니다. 트리 모델 학습 시, 일반적인 `RMSE` 손실 함수로 학습한 모델(안정적 기댓값)과 Zero-inflated 데이터에 특화된 `Tweedie` 손실 함수로 학습한 모델(극단적 스파이크 감지)의 예측값을 결합하여 서로의 단점을 상호 보완하는 **메타 앙상블(Meta-Ensemble)** 기법을 적용했습니다.

---

## 🔗 6. 참고자료 및 출처 (References)

본 프로젝트는 다음과 같은 최신 프레임워크와 논문, 그리고 오픈 데이터셋을 기반으로 연구 및 개발되었습니다.

* **Dataset Source**: [Kaggle M5 Forecasting Accuracy Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) (Makridakis Open Forecasting Center)
* **Deep Learning Framework**: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) (`nn.LSTM`, `nn.GRU`, `nn.Conv1d`)
* **Tabular ML Framework**: [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) (`HistGradientBoosting`), [LightGBM](https://lightgbm.readthedocs.io/)
* **Parallel Processing**: [Joblib](https://joblib.readthedocs.io/) (`joblib.Parallel` for multiprocessing)
* **Time-Series Vectorization**: [NumPy Sliding Window View](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html) (메모리 최적화 기법 적용)
