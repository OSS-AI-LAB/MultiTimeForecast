# TFT 통신사 손익전망 시계열 분석

darts 라이브러리의 TFT (Temporal Fusion Transformer) 모델을 활용하여 통신사 5G, 3G, LTE 등의 월별 데이터로 손익전망 시계열 데이터를 생성하는 프로젝트입니다.

## 프로젝트 구조

```
timesFM/
├── data/                   # 데이터 파일들
│   ├── raw/               # 원본 데이터
│   └── processed/         # 전처리된 데이터
├── src/                   # 소스 코드
│   ├── data_processing.py # 데이터 전처리
│   ├── tft_model.py       # TFT 모델 래퍼
│   ├── forecasting.py     # 예측 로직
│   └── visualization.py   # 시각화
├── notebooks/             # Jupyter 노트북
├── results/               # 결과 파일들
└── config/                # 설정 파일들
```

## 설치 및 실행

### 시스템 요구사항
- Python 3.11 이상
- 최소 8GB RAM (GPU 사용 권장)

### 1. 가상환경 생성 및 활성화:
```bash
# Python 3.11 확인
python --version

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### 2. 패키지 설치:
```bash
# pip 업그레이드
pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt
```

### 3. Jupyter 노트북 실행:
```bash
jupyter notebook
```

## 주요 기능

- **다변량 시계열 데이터 처리**: 5G, 3G, LTE 등 다양한 통신 기술 데이터
- **TFT 모델 활용**: darts 라이브러리의 TFT (Temporal Fusion Transformer) 모델을 통한 시계열 예측
- **손익전망 분석**: 수익성 지표 및 비용 분석
- **시각화**: 대화형 차트 및 리포트 생성
- **어텐션 메커니즘**: 변수 간 관계를 학습하는 고급 딥러닝 모델
- **자동 리포트 생성**: 예측 결과를 바탕으로 한 종합 분석 리포트
- **대화형 대시보드**: Plotly를 활용한 인터랙티브 시각화

## 예측 변수

### 타겟 변수 (예측 대상)
- `5g_users`, `lte_users`, `3g_users` - 기술별 사용자 수
- `5g_revenue`, `lte_revenue`, `3g_revenue` - 기술별 매출
- `5g_cost`, `lte_cost`, `3g_cost` - 기술별 비용

### 공변량 (예측에 도움되는 변수)
- `month`, `quarter`, `year` - 시간 특성
- `total_users`, `total_revenue`, `total_cost` - 총계 지표
- `profit`, `profit_margin` - 수익성 지표
- `5g_share`, `lte_share`, `3g_share` - 기술별 점유율
- `5g_arpu`, `lte_arpu`, `3g_arpu` - 기술별 ARPU (Average Revenue Per User)

## 데이터 요구사항

월별 데이터는 다음 형식을 권장합니다:
- 날짜 컬럼 (YYYY-MM 형식)
- 기술별 사용자 수
- 매출 데이터
- 비용 데이터
- 기타 관련 지표들

## 결과 파일

실행 후 `results/` 디렉토리에 다음 파일들이 생성됩니다:

### 예측 결과
- `forecast_results.csv` - 12개월 예측 결과
- `profitability_analysis.csv` - 수익성 분석 결과

### 기술별 분석
- `5g_analysis.csv` - 5G 기술별 상세 분석
- `lte_analysis.csv` - LTE 기술별 상세 분석  
- `3g_analysis.csv` - 3G 기술별 상세 분석

### 시각화 및 리포트
- `interactive_dashboard.html` - 대화형 대시보드 (Plotly)
- `analysis_report.txt` - 텍스트 분석 리포트
- `historical_trends.png` - 과거 데이터 추이 차트
- `forecast_comparison.png` - 예측 결과 비교 차트
- `profitability_analysis.png` - 수익성 분석 차트
- `technology_comparison.png` - 기술별 비교 차트

### 모델 파일
- `tft_model.pth` - 훈련된 TFT 모델

## TFT 모델의 장점

### 1. 어텐션 메커니즘
- 변수 간 관계를 자동으로 학습
- 중요한 시점과 변수에 집중하여 예측 정확도 향상

### 2. 다변량 처리
- 여러 변수를 동시에 예측
- 변수 간 상관관계를 고려한 예측

### 3. 시간 특성 활용
- 월, 분기, 연도 등의 시간 정보 활용
- 계절성 및 트렌드를 자동으로 학습

### 4. 공변량 지원
- 예측에 도움이 되는 추가 변수 활용
- 외부 요인을 고려한 예측

### 5. 확장성
- 새로운 변수 추가 용이
- 다양한 시계열 길이 지원

## 모델 설정 옵션

### 기본 설정
```python
config = {
    'input_chunk_length': 12,      # 입력 시퀀스 길이 (개월)
    'output_chunk_length': 12,     # 출력 시퀀스 길이 (개월)
    'hidden_size': 64,             # 은닉층 크기
    'num_attention_heads': 4,      # 어텐션 헤드 수
    'num_encoder_layers': 2,       # 인코더 레이어 수
    'num_decoder_layers': 2,       # 디코더 레이어 수
    'dropout': 0.1,                # 드롭아웃 비율
    'batch_size': 32,              # 배치 크기
    'n_epochs': 100,               # 훈련 에포크 수
    'learning_rate': 0.001         # 학습률
}
```

### 고성능 설정 (더 정확한 예측)
```python
config = {
    'hidden_size': 128,
    'num_attention_heads': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'n_epochs': 200,
    'batch_size': 16
}
```

## 사용 예시

### 1. 기본 실행
```bash
python main.py
```

### 2. Jupyter 노트북 사용
```bash
jupyter notebook notebooks/telecom_forecasting_demo.ipynb
```

### 3. 커스텀 설정으로 실행
```python
from src.forecasting import TelecomForecaster

# 커스텀 설정
config = {
    'n_epochs': 150,
    'hidden_size': 96,
    'num_attention_heads': 6
}

forecaster = TelecomForecaster(config)
results = forecaster.run_full_pipeline(
    file_path='your_data.csv',
    target_columns=['5g_users', 'lte_users', '3g_users'],
    forecast_steps=24  # 24개월 예측
)
``` 