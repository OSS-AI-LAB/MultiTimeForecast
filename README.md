# TimesFM - 통신사 재무 예측 시스템

Darts 라이브러리의 TFTModel을 활용한 통신사 계정과목별 매출 예측 시스템입니다.

## 🎯 프로젝트 개요

이 프로젝트는 통신사의 재무 데이터를 분석하여 GL_ACC_LSN_NM(계정과목명) 기준으로 매출을 예측하는 시스템입니다.

### 주요 특징

- **계정과목별 예측**: GL_ACC_LSN_NM을 기준으로 한 다변량 시계열 예측
- **Darts TFTModel**: Temporal Fusion Transformer를 활용한 고성능 예측
- **앙상블 모델**: TFT + Prophet 모델 앙상블
- **동적 특성 엔지니어링**: 시간적 특성, 지연 특성, 이동평균 등
- **다양한 파일 형식 지원**: CSV, Excel (.xlsx, .xls), DRM 보호 파일 포함
- **자동 인코딩 감지**: chardet를 사용한 자동 인코딩 감지

## 📊 데이터 구조

### 입력 데이터 형식

```csv
BASE_YM,BASE_YY,ENTR_3_PROD_LEVEL_NM,PROFT_SRC_NM,GL_ACC_LSN_NO,GL_ACC_LSN_NM,SUM_DIV_NM,PRFIT_PERSP_1_INDX_VAL
202306,2025,3G,서비스이용료,415020400,무선전화_기본료,월별매출,1697722067
202306,2025,LTE,서비스이용료,415050400,무선전화_플랫폼이용료,월별매출,8842835544
...
```

### 컬럼 설명

- `BASE_YM`: 기준년월 (YYYYMM)
- `BASE_YY`: 기준년도
- `ENTR_3_PROD_LEVEL_NM`: 제품 레벨 (3G, LTE, 5G, Enterprise무선 등)
- `PROFT_SRC_NM`: 손익원천명
- `GL_ACC_LSN_NO`: 계정과목 코드
- `GL_ACC_LSN_NM`: 계정과목명 (예측 대상)
- `SUM_DIV_NM`: 집계 구분명
- `PRFIT_PERSP_1_INDX_VAL`: 매출액

## 🏗️ 프로젝트 구조

```
timesFM/
├── config/
│   └── config.yaml              # 설정 파일
├── data/
│   ├── raw/                     # 원본 데이터
│   └── processed/               # 처리된 데이터
├── src/
│   ├── data_processor.py        # 데이터 처리 모듈
│   ├── models.py               # 예측 모델 (Darts TFTModel)
│   └── visualizer.py           # 시각화 모듈
├── notebooks/
│   └── telecom_forecasting_demo.ipynb  # 주피터 노트북 데모
├── results/                    # 예측 결과 및 차트
├── logs/                       # 로그 파일
├── main.py                     # 메인 실행 스크립트
├── requirements.txt            # 의존성 목록
└── README.md                   # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

다음 파일 형식을 지원합니다:
- `data/raw/telecom_financial_data.csv` (UTF-8, CP949, EUC-KR 등 다양한 인코딩 지원)
- `data/raw/telecom_financial_data.xlsx` (Excel 파일, DRM 보호 파일 포함)
- `data/raw/telecom_financial_data.xls` (Excel 파일)

**참고**: DRM 보호된 Excel 파일도 자동으로 처리됩니다.

### 3. 실행

```bash
python main.py
```

## ⚙️ 설정

`config/config.yaml` 파일에서 다음 설정을 조정할 수 있습니다:

### 데이터 설정
```yaml
data:
  account_filtering:
    min_total_value: 1000000  # 최소 총 매출액
    min_occurrence: 3         # 최소 발생 횟수
    exclude_patterns: ["<할인>", "<포인트>"]  # 제외 패턴
```

### 모델 설정
```yaml
model:
  use_ensemble: true  # true: TFT + Prophet 앙상블, false: TFT만 사용
  tft:
    input_chunk_length: 6     # 입력 시퀀스 길이
    output_chunk_length: 3    # 출력 시퀀스 길이
    hidden_size: 64          # 히든 레이어 크기
    num_attention_heads: 4   # 어텐션 헤드 수
    n_epochs: 50            # 훈련 에포크
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]     # TFT 70%, Prophet 30%
```

### 예측 설정
```yaml
forecasting:
  forecast_horizon: 12       # 예측 기간 (개월)
  validation_periods: 6      # 검증 기간
```

## 📈 예측 결과

### 출력 파일

- `results/forecast_results.csv`: 예측 결과
- `results/evaluation_results.csv`: 모델 평가 결과
- `results/forecast_plot.html`: 예측 차트
- `results/accuracy_plot.html`: 모델 정확도 비교
- `results/correlation_plot.html`: 계정과목 간 상관관계
- `results/seasonal_plot.html`: 계절성 분석
- `results/hierarchical_plot.html`: 계층적 예측 분석
- `results/dashboard.html`: 종합 대시보드

### 예측 성능 지표

- **MAE**: 평균 절대 오차
- **MAPE**: 평균 절대 백분율 오차
- **RMSE**: 평균 제곱근 오차
- **SMAPE**: 대칭 평균 절대 백분율 오차

## 🔧 주요 기능

### 1. 데이터 전처리

- **다양한 파일 형식 지원**: CSV, Excel (.xlsx, .xls) 파일 지원
- **자동 인코딩 감지**: chardet를 사용한 자동 인코딩 감지
- **DRM 보호 파일 처리**: 보호된 Excel 파일 자동 처리
- **계정과목 필터링**: 중요도 기반 계정과목 선택
- **피벗 변환**: 계정과목별 시계열 데이터 생성
- **특성 엔지니어링**: 시간적 특성, 지연 특성, 이동평균
- **정규화**: RobustScaler를 통한 특성 스케일링

### 2. 모델링

- **TFTModel**: Darts의 Temporal Fusion Transformer
- **Prophet**: Facebook의 시계열 예측 모델
- **앙상블**: TFT + Prophet 가중 평균

### 3. 시각화

- **예측 결과**: 실제 vs 예측 비교
- **모델 성능**: 정확도 지표 비교
- **상관관계**: 계정과목 간 관계 분석
- **계절성**: 시계열 분해 분석

## 📊 예시 결과

### 예측 결과 예시
```
=== 예측 완료 ===
예측 기간: 12개월
예측된 계정과목: 10개

최종 예측값 (12개월 후):
  무선전화_기본료: 1,750,000,000원
  무선전화_플랫폼이용료: 9,200,000,000원
  무선전화_통화서비스: 1,500,000,000원
  무선전화_데이터(TRAFFIC)이용료: 28,000,000원
  유선전화_기본료: 55,000,000원

성장률 분석 (12개월):
  무선전화_기본료: +3.12%
  무선전화_플랫폼이용료: +4.05%
  무선전화_통화서비스: +75.45%
  무선전화_데이터(TRAFFIC)이용료: +3.70%
  유선전화_기본료: +2.94%
```

## 📓 주피터 노트북 사용

프로젝트에는 `notebooks/telecom_forecasting_demo.ipynb` 파일이 포함되어 있어 단계별 분석 과정을 확인할 수 있습니다.

```bash
# 주피터 노트북 실행
jupyter notebook notebooks/telecom_forecasting_demo.ipynb
```

## 🤝 기여

프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🛠️ 문제 해결

### 인코딩 오류 해결
```
ERROR: 'utf-8' codec can't decode byte 0xd0 in position 10: invalid continuation byte
```

**해결 방법**:
1. 의존성 재설치: `pip install -r requirements.txt`
2. 시스템 재실행: `python main.py`

### DRM 보호 파일 오류 해결
Excel 파일이 DRM으로 보호되어 있는 경우에도 자동으로 처리됩니다.

### 기타 문제
- 가상환경이 활성화되어 있는지 확인
- Python 3.11 이상 버전 사용
- 충분한 메모리 확보 (최소 8GB 권장)

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요. 