# TimesFM 통신사 재무 예측 시스템 - 사용자 가이드

## 📋 목차
1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 선택 및 설정](#모델-선택-및-설정)
4. [시스템 실행](#시스템-실행)
5. [주피터 노트북 사용법](#주피터-노트북-사용법)
6. [결과 해석](#결과-해석)
7. [문제 해결](#문제-해결)

---

## 🛠️ 환경 설정

### 1. Python 환경 확인
```bash
# Python 버전 확인 (3.11 이상 필요)
python --version

# pip 업그레이드
pip install --upgrade pip
```

### 2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 필요한 패키지 설치
```bash
# 프로젝트 디렉토리로 이동
cd /path/to/timesFM

# 패키지 설치
pip install -r requirements.txt
```

### 4. 설치 확인
```bash
# Python에서 패키지 import 테스트
python -c "import pandas, torch, darts; print('설치 완료!')"

# GPU 사용 가능 여부 확인 (선택사항)
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

---

## 📊 데이터 준비

### 1. 필수 컬럼 구조

데이터 파일은 다음 컬럼들을 포함해야 합니다:

| 컬럼명 | 설명 | 데이터 타입 | 예시 |
|--------|------|-------------|------|
| `BASE_YM` | 기준년월 | YYYYMM | 202306 |
| `BASE_YY` | 기준년도 | YYYY | 2025 |
| `ENTR_3_PROD_LEVEL_NM` | 제품 레벨 | 문자열 | 3G, LTE, 5G |
| `PROFT_SRC_NM` | 손익원천명 | 문자열 | 서비스이용료 |
| `GL_ACC_LSN_NO` | 계정과목 코드 | 문자열 | 415020400 |
| `GL_ACC_LSN_NM` | 계정과목명 | 문자열 | 무선전화_기본료 |
| `SUM_DIV_NM` | 집계 구분명 | 문자열 | 월별매출 |
| `PRFIT_PERSP_1_INDX_VAL` | 매출액 | 숫자 | 1697722067 |

### 2. 지원 파일 형식

- **CSV 파일**: `.csv` (UTF-8, CP949, EUC-KR 등 다양한 인코딩 지원)
- **Excel 파일**: `.xlsx`, `.xls` (DRM 보호 파일 포함)

### 3. 데이터 준비 체크리스트

- [ ] 필수 컬럼이 모두 포함되어 있는지 확인
- [ ] 모든 수치 데이터는 숫자 형식 (쉼표, 통화 기호 제거)
- [ ] 결측치가 없는지 확인
- [ ] 최소 12개월 이상의 데이터 확보
- [ ] 데이터가 시간순으로 정렬되어 있는지 확인

### 4. 파일 저장 위치

데이터 파일을 다음 경로에 저장하세요:
```
timesFM/
└── data/
    └── raw/
        └── telecom_financial_data.xlsx  # 여기에 파일 저장
```

---

## 🤖 모델 선택 및 설정

### 1. 지원 모델

시스템은 다음 5가지 모델을 지원합니다:

| 모델 | 타입 | 특징 | 적합한 데이터 |
|------|------|------|---------------|
| **TFT** | 다변량 | 복잡한 패턴 학습, 변수 간 관계 고려 | 다변량, 복잡한 시계열 |
| **Prophet** | 단변량 | 계절성 처리 우수, 해석 가능 | 계절성이 강한 단변량 |
| **LSTM** | 단변량 | 긴 시퀀스 처리, 안정적 | 긴 의존성이 있는 시계열 |
| **GRU** | 단변량 | LSTM보다 빠름, 적은 파라미터 | 중간 길이 의존성 |
| **Transformer** | 단변량 | 병렬 처리, 어텐션 메커니즘 | 복잡한 패턴, 충분한 데이터 |

### 2. 모델 선택 전략

#### 전략 1: TFT 전용 모드 (`strategy: "tft_only"`)
```yaml
model:
  strategy: "tft_only"
```
- **용도**: 빠른 예측, 단일 모델 신뢰도
- **장점**: 빠른 실행 (5-10분), 안정적
- **단점**: 앙상블 효과 없음

#### 전략 2: 기존 앙상블 모드 (`strategy: "ensemble"`)
```yaml
model:
  strategy: "ensemble"
  use_ensemble: true
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]
```
- **용도**: TFT + Prophet 조합
- **장점**: 검증된 조합, 안정적 (10-15분)
- **단점**: 제한된 모델 조합

#### 전략 3: 다중 모델 앙상블 (`strategy: "multi_model"`)
```yaml
model:
  strategy: "multi_model"
  multi_model_ensemble:
    enabled: true
    models: ["tft", "prophet", "lstm", "gru", "transformer"]
    weights: [0.4, 0.2, 0.15, 0.15, 0.1]
```
- **용도**: 최고 성능 추구
- **장점**: 다양한 모델의 장점 활용
- **단점**: 긴 실행 시간 (30-60분), 복잡성

#### 전략 4: 자동 선택 모드 (`strategy: "auto_select"`)
```yaml
model:
  strategy: "auto_select"
```
- **용도**: 성능 기반 자동 선택
- **장점**: 데이터에 최적화된 모델 선택
- **단점**: 평가 시간 필요

### 3. 권장 사용 시나리오

#### 시나리오 1: 빠른 프로토타이핑
```yaml
model:
  strategy: "tft_only"
  use_ensemble: false
```

#### 시나리오 2: 안정적인 프로덕션
```yaml
model:
  strategy: "ensemble"
  use_ensemble: true
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]
```

#### 시나리오 3: 최고 성능 추구
```yaml
model:
  strategy: "multi_model"
  multi_model_ensemble:
    enabled: true
    models: ["tft", "prophet", "lstm", "gru", "transformer"]
    weights: [0.4, 0.2, 0.15, 0.15, 0.1]
```

### 4. 모델별 설정

#### TFT 모델 설정
```yaml
model:
  tft:
    input_chunk_length: 6     # 입력 시퀀스 길이
    output_chunk_length: 3    # 출력 시퀀스 길이
    hidden_size: 64          # 히든 레이어 크기
    lstm_layers: 1           # LSTM 레이어 수
    num_attention_heads: 4   # 어텐션 헤드 수
    dropout: 0.1            # 드롭아웃 비율
    n_epochs: 50            # 훈련 에포크
    batch_size: 32          # 배치 크기
```

#### LSTM 모델 설정
```yaml
model:
  lstm:
    input_chunk_length: 6
    hidden_dim: 64
    n_rnn_layers: 2
    dropout: 0.1
    n_epochs: 50
    batch_size: 32
```

#### GRU 모델 설정
```yaml
model:
  gru:
    input_chunk_length: 6
    hidden_dim: 64
    n_rnn_layers: 2
    dropout: 0.1
    n_epochs: 50
    batch_size: 32
```

#### Transformer 모델 설정
```yaml
model:
  transformer:
    input_chunk_length: 6
    output_chunk_length: 3
    d_model: 64
    nhead: 8
    num_encoder_layers: 4
    num_decoder_layers: 4
    dim_feedforward: 256
    dropout: 0.1
    n_epochs: 50
    batch_size: 32
```

---

## 🚀 시스템 실행

### 1. 기본 실행

```bash
# 메인 스크립트 실행
python main.py
```

### 2. 실행 과정

시스템이 실행되면 다음과 같은 단계를 거칩니다:

1. **데이터 처리기 초기화**
2. **원본 데이터 처리**
   - 다양한 파일 형식 지원
   - 자동 인코딩 감지
   - 계정과목 필터링
   - 특성 엔지니어링
3. **예측기 초기화**
   - 선택된 모델 설정
   - 앙상블 구성
4. **예측 파이프라인 실행**
   - 모델 훈련
   - 예측 수행
   - 앙상블 결합
5. **시각화 리포트 생성**
   - 예측 차트
   - 정확도 분석
   - 상관관계 분석

### 3. 설정 파일 수정

`config/config.yaml` 파일에서 다음 설정을 조정할 수 있습니다:

#### 데이터 필터링 설정
```yaml
data:
  account_filtering:
    min_total_value: 1000000  # 최소 총 매출액
    min_occurrence: 3         # 최소 발생 횟수
    exclude_patterns: ["<할인>", "<포인트>"]  # 제외 패턴
```

#### 모델 전략 설정
```yaml
model:
  strategy: "ensemble"  # "tft_only", "ensemble", "multi_model", "auto_select"
  use_ensemble: true    # 앙상블 사용 여부
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]      # TFT 70%, Prophet 30%
  multi_model_ensemble:
    enabled: false           # 다중 모델 앙상블 사용 여부
    models: ["tft", "prophet", "lstm", "gru", "transformer"]
    weights: [0.4, 0.2, 0.15, 0.15, 0.1]
```

#### 예측 설정
```yaml
forecasting:
  forecast_horizon: 12       # 예측 기간 (개월)
  validation_periods: 6      # 검증 기간
```

---

## 📓 주피터 노트북 사용법

### 1. 주피터 노트북 실행

```bash
# 프로젝트 디렉토리에서
jupyter notebook
```

또는 특정 노트북 실행:
```bash
jupyter notebook notebooks/telecom_forecasting_demo.ipynb
```

### 2. 노트북 기본 사용법

#### 셀 실행 방법
- **Shift + Enter**: 현재 셀 실행 후 다음 셀로 이동
- **Ctrl + Enter**: 현재 셀 실행 후 현재 셀에 머무름
- **Alt + Enter**: 현재 셀 실행 후 새 셀 생성

#### 셀 타입
- **Code**: Python 코드 실행
- **Markdown**: 문서 작성 (설명, 제목 등)

### 3. 단계별 분석 과정

#### Step 1: 환경 설정 및 데이터 로드
```python
# 필요한 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processor import TelecomDataProcessor
from src.models import TelecomForecaster
from src.visualizer import TelecomVisualizer

# 한글 폰트 설정 (한글 출력 시)
plt.rcParams['font.family'] = 'DejaVu Sans'
```

#### Step 2: 데이터 처리
```python
# 데이터 처리기 생성
processor = TelecomDataProcessor()

# 데이터 처리
processed_data = processor.process_data()

# 특성 정보 확인
feature_info = processor.get_feature_info()
print(f"처리된 계정과목: {len(feature_info['account_columns'])}개")
print(f"처리된 제품: {len(feature_info['product_columns'])}개")
```

#### Step 3: 모델 전략 설정
```python
# 설정 파일에서 모델 전략 확인
import yaml
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

strategy = config['model']['strategy']
print(f"현재 모델 전략: {strategy}")

if strategy == "multi_model":
    models = config['model']['multi_model_ensemble']['models']
    weights = config['model']['multi_model_ensemble']['weights']
    print(f"사용 모델: {models}")
    print(f"가중치: {weights}")
```

#### Step 4: 예측 모델 실행
```python
# 예측기 생성
forecaster = TelecomForecaster()

# 타겟 컬럼 정의 (상위 10개 계정과목)
target_columns = feature_info['account_columns'][:10]

# 예측 파이프라인 실행
results = forecaster.run_forecast_pipeline(
    processed_data=processed_data,
    target_columns=target_columns,
    forecast_horizon=12
)

print("예측 완료!")
```

#### Step 5: 결과 시각화
```python
# 시각화기 생성
visualizer = TelecomVisualizer()

# 리포트 생성
report_path = visualizer.generate_report(
    processed_data=processed_data,
    results=results,
    target_columns=target_columns
)

print(f"리포트 생성 완료: {report_path}")
```

### 4. 노트북 저장 및 공유

#### 노트북 저장
- **Ctrl + S**: 노트북 저장
- **File → Download as**: 다양한 형식으로 다운로드

#### 노트북 공유
```bash
# HTML 형식으로 변환
jupyter nbconvert --to html notebooks/telecom_forecasting_demo.ipynb

# PDF 형식으로 변환 (LaTeX 설치 필요)
jupyter nbconvert --to pdf notebooks/telecom_forecasting_demo.ipynb
```

---

## 📈 결과 해석

### 1. 예측 결과 파일

분석 완료 후 `results/` 폴더에 다음 파일들이 생성됩니다:

#### 예측 결과
- `forecast_results.csv`: 예측 결과
- `evaluation_results.csv`: 모델 평가 결과

#### 시각화 파일
- `forecast_plot.html`: 예측 결과 차트
- `accuracy_plot.html`: 모델 정확도 비교
- `seasonal_plot.html`: 계절성 분석
- `dashboard.html`: 종합 대시보드

### 2. 모델별 성능 비교

#### 성능 지표
- **MAE**: 평균 절대 오차
- **MAPE**: 평균 절대 백분율 오차
- **RMSE**: 평균 제곱근 오차
- **SMAPE**: 대칭 평균 절대 백분율 오차

#### 모델별 특징
```python
# 모델별 성능 확인
evaluation_results = results['evaluation_results']

for model_name, model_results in evaluation_results.items():
    print(f"\n{model_name.upper()} 모델 성능:")
    for col, metrics in model_results.items():
        mape = metrics.get('mape', 'N/A')
        print(f"  {col}: MAPE = {mape:.2f}%")
```

### 3. 결과 해석 방법

#### 예측 정확도 평가
```python
# 예측 정확도 확인
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 실제값과 예측값 비교
actual = processed_data[target_columns].iloc[-6:]  # 최근 6개월
predicted = results['ensemble_forecast'].iloc[:6]  # 예측 6개월

for col in target_columns:
    if col in actual.columns and col in predicted.columns:
        mae = mean_absolute_error(actual[col], predicted[col])
        print(f"{col}: MAE = {mae:,.0f}원")
```

#### 트렌드 분석
- **상승 트렌드**: 지속적인 성장 예상
- **하락 트렌드**: 시장 포화 또는 기술 전환
- **안정적**: 성숙한 시장

#### 계정과목별 분석
- **무선전화_기본료**: 기본 서비스 수익
- **무선전화_플랫폼이용료**: 데이터 서비스 수익
- **무선전화_통화서비스**: 음성 서비스 수익

### 4. 비즈니스 인사이트 도출

#### 기술별 전략
- **5G**: 신규 투자 및 마케팅 강화
- **LTE**: 안정적 운영 및 비용 최적화
- **3G**: 점진적 축소 및 사용자 이전

#### 수익성 개선 방안
- **ARPU 향상**: 부가서비스 개발
- **비용 최적화**: 네트워크 효율성 개선
- **고객 유지**: 충성도 프로그램 강화

---

## 🔧 문제 해결

### 1. 일반적인 오류 및 해결방법

#### 데이터 로드 오류
```python
# 오류: 파일을 찾을 수 없음
# 해결: 파일 경로 확인
import os
print(os.path.exists('data/raw/telecom_financial_data.xlsx'))

# 오류: 엑셀 파일 읽기 실패
# 해결: openpyxl 설치 확인
pip install openpyxl
```

#### 인코딩 오류
```python
# 오류: 'utf-8' codec can't decode byte
# 해결: 자동 인코딩 감지 사용
# 시스템이 자동으로 처리하므로 별도 조치 불필요
```

#### 메모리 부족 오류
```python
# 해결: 설정 파일에서 배치 크기 줄이기
# config/config.yaml 수정
model:
  tft:
    batch_size: 16  # 기본값 32에서 줄임
  lstm:
    batch_size: 16
  gru:
    batch_size: 16
  transformer:
    batch_size: 16
```

#### 훈련 시간이 너무 긴 경우
```python
# 해결: 설정 파일에서 에포크 수 줄이기
# config/config.yaml 수정
model:
  tft:
    n_epochs: 25  # 기본값 50에서 줄임
  lstm:
    n_epochs: 25
  gru:
    n_epochs: 25
  transformer:
    n_epochs: 25
```

### 2. 모델별 최적화

#### TFT 모델 최적화
```yaml
model:
  tft:
    input_chunk_length: 4   # 입력 시퀀스 길이 줄임
    output_chunk_length: 2  # 출력 시퀀스 길이 줄임
    hidden_size: 32        # 히든 크기 줄임
    lstm_layers: 1         # 레이어 수 줄임
```

#### LSTM/GRU 모델 최적화
```yaml
model:
  lstm:
    hidden_dim: 32         # 히든 크기 줄임
    n_rnn_layers: 1        # 레이어 수 줄임
  gru:
    hidden_dim: 32
    n_rnn_layers: 1
```

#### Transformer 모델 최적화
```yaml
model:
  transformer:
    d_model: 32           # 모델 크기 줄임
    num_encoder_layers: 2 # 인코더 레이어 줄임
    num_decoder_layers: 2 # 디코더 레이어 줄임
```

### 3. 성능 최적화

#### GPU 사용 (선택사항)
```python
# CUDA 사용 가능 여부 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# GPU 사용 설정 (자동으로 처리됨)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

#### 데이터 크기 최적화
```python
# 설정 파일에서 chunk length 조정
# config/config.yaml 수정
model:
  tft:
    input_chunk_length: 4   # 입력 시퀀스 길이 줄임
    output_chunk_length: 2  # 출력 시퀀스 길이 줄임
```

### 4. 디버깅 팁

#### 로그 확인
```python
# 상세한 로그 출력
import logging
logging.basicConfig(level=logging.INFO)

# 데이터 처리 과정 확인
processor = TelecomDataProcessor()
processed_data = processor.process_data()
```

#### 단계별 테스트
```python
# 각 단계별로 테스트
# 1. 데이터 처리 테스트
processor = TelecomDataProcessor()
processed_data = processor.process_data()
print("1. 데이터 처리 완료")

# 2. 예측 모델 테스트
forecaster = TelecomForecaster()
target_columns = ['무선전화_기본료', '무선전화_플랫폼이용료']
results = forecaster.run_forecast_pipeline(
    processed_data=processed_data,
    target_columns=target_columns,
    forecast_horizon=6
)
print("2. 예측 완료")
```

---

## 📞 추가 지원

### 1. 문서 및 참고자료
- [Darts 라이브러리 공식 문서](https://unit8co.github.io/darts/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [MODEL_USAGE_GUIDE.md](MODEL_USAGE_GUIDE.md): 상세한 모델 사용 가이드

### 2. 커뮤니티 지원
- GitHub Issues: 프로젝트 저장소의 Issues 탭
- Stack Overflow: `darts`, `pytorch`, `timeseries` 태그

### 3. 성능 튜닝 가이드
- 데이터 품질 향상
- 모델 하이퍼파라미터 조정
- 하드웨어 최적화

---

## 📝 체크리스트

### 초기 설정
- [ ] Python 3.11+ 설치
- [ ] 가상환경 생성 및 활성화
- [ ] 필요한 패키지 설치
- [ ] 데이터 파일 준비

### 데이터 준비
- [ ] 필수 컬럼 포함 확인
- [ ] 데이터 품질 검증
- [ ] 파일 경로 설정
- [ ] 파일 형식 확인

### 모델 설정
- [ ] 모델 전략 선택 (tft_only, ensemble, multi_model, auto_select)
- [ ] 앙상블 설정 확인
- [ ] 모델별 파라미터 조정
- [ ] 실행 시간 및 리소스 고려

### 시스템 실행
- [ ] 설정 파일 확인
- [ ] 메인 스크립트 실행
- [ ] 예측 완료 확인
- [ ] 결과 파일 생성 확인

### 결과 확인
- [ ] 예측 결과 파일 확인
- [ ] 시각화 파일 확인
- [ ] 모델별 성능 비교
- [ ] 결과 해석 및 인사이트 도출
- [ ] 보고서 작성

---

**마지막 업데이트**: 2024년 12월
**버전**: 3.0 (다중 모델 지원) 