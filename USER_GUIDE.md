# TimesFM 통신사 손익전망 분석 - 사용자 가이드

## 📋 목차
1. [환경 설정](#환경-설정)
2. [엑셀 데이터 준비](#엑셀-데이터-준비)
3. [데이터 입력 방법](#데이터-입력-방법)
4. [주피터 노트북 사용법](#주피터-노트북-사용법)
5. [결과 해석](#결과-해석)
6. [문제 해결](#문제-해결)

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
python -m venv timesfm_env

# 가상환경 활성화
# macOS/Linux:
source timesfm_env/bin/activate
# Windows:
timesfm_env\Scripts\activate
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
# Jupyter 설치 확인
jupyter --version

# Python에서 패키지 import 테스트
python -c "import pandas, torch, darts; print('설치 완료!')"

# GPU 사용 가능 여부 확인 (선택사항)
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

---

## 📊 엑셀 데이터 준비

### 1. 필수 컬럼 구조

엑셀 파일은 다음 컬럼들을 포함해야 합니다:

| 컬럼명 | 설명 | 데이터 타입 | 예시 |
|--------|------|-------------|------|
| `date` | 날짜 | YYYY-MM-DD | 2020-01-01 |
| `5g_users` | 5G 사용자 수 | 숫자 | 1000000 |
| `lte_users` | LTE 사용자 수 | 숫자 | 5000000 |
| `3g_users` | 3G 사용자 수 | 숫자 | 1000000 |
| `5g_revenue` | 5G 매출 | 숫자 | 50000000 |
| `lte_revenue` | LTE 매출 | 숫자 | 200000000 |
| `3g_revenue` | 3G 매출 | 숫자 | 30000000 |
| `5g_cost` | 5G 비용 | 숫자 | 30000000 |
| `lte_cost` | LTE 비용 | 숫자 | 120000000 |
| `3g_cost` | 3G 비용 | 숫자 | 20000000 |

### 2. 엑셀 파일 형식 예시

**Sheet1: telecom_data**
| date | 5g_users | lte_users | 3g_users | 5g_revenue | lte_revenue | 3g_revenue | 5g_cost | lte_cost | 3g_cost |
|------|----------|-----------|----------|------------|-------------|------------|---------|----------|---------|
| 2020-01-01 | 1000000 | 5000000 | 1000000 | 50000000 | 200000000 | 30000000 | 30000000 | 120000000 | 20000000 |
| 2020-02-01 | 1100000 | 5100000 | 950000 | 55000000 | 204000000 | 28500000 | 33000000 | 122400000 | 19000000 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 3. 데이터 준비 체크리스트

- [ ] 날짜는 YYYY-MM-DD 형식으로 입력
- [ ] 모든 수치 데이터는 숫자 형식 (쉼표, 통화 기호 제거)
- [ ] 결측치가 없는지 확인
- [ ] 최소 12개월 이상의 데이터 확보
- [ ] 데이터가 시간순으로 정렬되어 있는지 확인

### 4. 엑셀 파일 저장 방법

1. **Excel 2016 이상 사용 권장**
2. **파일 형식**: `.xlsx` 또는 `.csv`
3. **인코딩**: UTF-8 (한글 포함 시)
4. **저장 위치**: `data/raw/` 폴더
5. **파일명 예시**: `telecom_data.xlsx`, `my_company_data.csv`

---

## 📁 데이터 입력 방법

### 1. 파일 위치 설정

데이터 파일을 다음 경로에 저장하세요:
```
timesFM/
└── data/
    └── raw/
        └── your_telecom_data.xlsx  # 여기에 파일 저장
```

### 2. 데이터 검증

```python
# 데이터 검증 스크립트
import pandas as pd
from src.data_processing import TelecomDataProcessor

# 데이터 로더 생성
processor = TelecomDataProcessor()

# 데이터 로드
df = processor.load_data('data/raw/your_telecom_data.xlsx')  # 파일명을 실제 파일명으로 변경하세요

# 데이터 검증
is_valid = processor.validate_data(df)
print(f"데이터 유효성: {is_valid}")

# 데이터 미리보기
print(df.head())
print(f"데이터 크기: {df.shape}")
```

### 3. 자동 데이터 정리

```python
# 데이터 정리 및 전처리
df_clean = processor.clean_data(df)
df_features = processor.create_features(df_clean)

print("정리된 데이터:")
print(df_features.head())
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
from src.data_processing import TelecomDataProcessor
from src.forecasting import TelecomForecaster

# 한글 폰트 설정 (한글 출력 시)
plt.rcParams['font.family'] = 'DejaVu Sans'
```

#### Step 2: 데이터 로드 및 검증
```python
# 데이터 로더 생성
processor = TelecomDataProcessor()

# 데이터 로드
df = processor.load_data('data/raw/your_telecom_data.xlsx')  # 파일명을 실제 파일명으로 변경하세요

# 데이터 검증
if processor.validate_data(df):
    print("✅ 데이터 검증 통과")
    print(f"데이터 크기: {df.shape}")
    print(f"기간: {df['date'].min()} ~ {df['date'].max()}")
else:
    print("❌ 데이터 검증 실패")
```

#### Step 3: 데이터 전처리
```python
# 데이터 정리
df_clean = processor.clean_data(df)

# 특성 생성
df_features = processor.create_features(df_clean)

# 전처리된 데이터 확인
print("전처리된 데이터:")
print(df_features.head())
```

#### Step 4: 시각화 및 탐색적 분석
```python
# 기술별 사용자 수 추이
plt.figure(figsize=(12, 6))
plt.plot(df_features['date'], df_features['5g_users'], label='5G')
plt.plot(df_features['date'], df_features['lte_users'], label='LTE')
plt.plot(df_features['date'], df_features['3g_users'], label='3G')
plt.title('기술별 사용자 수 추이')
plt.xlabel('날짜')
plt.ylabel('사용자 수')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 수익성 분석
plt.figure(figsize=(12, 6))
plt.plot(df_features['date'], df_features['profit'], label='총 이익')
plt.plot(df_features['date'], df_features['profit_margin'], label='이익률')
plt.title('수익성 추이')
plt.xlabel('날짜')
plt.ylabel('금액/비율')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### Step 5: 모델 훈련 및 예측
```python
# 예측기 생성
forecaster = TelecomForecaster()

# 전체 파이프라인 실행
results = forecaster.run_full_pipeline(
    file_path='data/raw/your_telecom_data.xlsx',  # 파일명을 실제 파일명으로 변경하세요
    target_columns=['5g_users', 'lte_users', '3g_users'],
    forecast_steps=12  # 12개월 예측
)

print("예측 완료!")
```

#### Step 6: 결과 시각화
```python
# 예측 결과 시각화
forecaster.plot_forecasts(results)

# 수익성 분석 시각화
forecaster.plot_profitability_analysis(results)

# 기술별 비교 시각화
forecaster.plot_technology_comparison(results)
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
- `forecast_results.csv`: 12개월 예측 결과
- `profitability_analysis.csv`: 수익성 분석 결과

#### 시각화 파일
- `interactive_dashboard.html`: 대화형 대시보드
- `historical_trends.png`: 과거 데이터 추이
- `forecast_comparison.png`: 예측 결과 비교
- `profitability_analysis.png`: 수익성 분석 차트

### 2. 결과 해석 방법

#### 예측 정확도 평가
```python
# 예측 정확도 확인
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 실제값과 예측값 비교
actual = df_features['5g_users'].tail(12)
predicted = results['forecast']['5g_users']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
```

#### 트렌드 분석
- **상승 트렌드**: 지속적인 성장 예상
- **하락 트렌드**: 시장 포화 또는 기술 전환
- **안정적**: 성숙한 시장

#### 수익성 분석
- **이익률 증가**: 효율성 개선
- **이익률 감소**: 경쟁 심화 또는 비용 증가
- **안정적 이익률**: 성숙한 비즈니스 모델

### 3. 비즈니스 인사이트 도출

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
print(os.path.exists('data/raw/your_telecom_data.xlsx'))  # 파일명을 실제 파일명으로 변경하세요

# 오류: 엑셀 파일 읽기 실패
# 해결: openpyxl 설치 확인
pip install openpyxl
```

#### 메모리 부족 오류
```python
# 해결: 배치 크기 줄이기
config = {
    'batch_size': 16,  # 기본값 32에서 줄임
    'hidden_size': 32  # 기본값 64에서 줄임
}
```

#### 훈련 시간이 너무 긴 경우
```python
# 해결: 에포크 수 줄이기
config = {
    'n_epochs': 50,  # 기본값 100에서 줄임
    'learning_rate': 0.01  # 학습률 증가
}
```

### 2. 성능 최적화

#### GPU 사용 (선택사항)
```python
# CUDA 사용 가능 여부 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# GPU 사용 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

#### 데이터 크기 최적화
```python
# 대용량 데이터 처리
config = {
    'input_chunk_length': 6,  # 입력 시퀀스 길이 줄임
    'output_chunk_length': 6,  # 출력 시퀀스 길이 줄임
}
```

### 3. 디버깅 팁

#### 로그 확인
```python
# 상세한 로그 출력
import logging
logging.basicConfig(level=logging.INFO)

# 데이터 검증 상세 정보
processor = TelecomDataProcessor()
df = processor.load_data('data/raw/your_telecom_data.xlsx')  # 파일명을 실제 파일명으로 변경하세요
print("컬럼 목록:", df.columns.tolist())
print("데이터 타입:", df.dtypes)
print("결측치:", df.isnull().sum())
```

#### 단계별 테스트
```python
# 각 단계별로 테스트
# 1. 데이터 로드 테스트
df = processor.load_data('data/raw/your_telecom_data.xlsx')  # 파일명을 실제 파일명으로 변경하세요
print("1. 데이터 로드 완료")

# 2. 데이터 검증 테스트
is_valid = processor.validate_data(df)
print(f"2. 데이터 검증: {is_valid}")

# 3. 데이터 정리 테스트
df_clean = processor.clean_data(df)
print("3. 데이터 정리 완료")

# 4. 특성 생성 테스트
df_features = processor.create_features(df_clean)
print("4. 특성 생성 완료")
```

---

## 📞 추가 지원

### 1. 문서 및 참고자료
- [Darts 라이브러리 공식 문서](https://unit8co.github.io/darts/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)

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
- [ ] Jupyter 노트북 실행 확인

### 데이터 준비
- [ ] 엑셀 파일 형식 확인
- [ ] 필수 컬럼 포함 확인
- [ ] 데이터 품질 검증
- [ ] 파일 경로 설정

### 분석 실행
- [ ] 데이터 로드 및 검증
- [ ] 전처리 완료
- [ ] 모델 훈련 완료
- [ ] 예측 결과 생성

### 결과 확인
- [ ] 예측 결과 파일 확인
- [ ] 시각화 파일 확인
- [ ] 결과 해석 및 인사이트 도출
- [ ] 보고서 작성

---

**마지막 업데이트**: 2024년 12월
**버전**: 1.0 