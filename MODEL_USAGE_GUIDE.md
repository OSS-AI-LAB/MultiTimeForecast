# 모델 사용 가이드

## 개요
현재 시스템은 다음 5가지 모델을 지원합니다:
- **TFT (Temporal Fusion Transformer)**: 다변량 시계열 예측
- **Prophet**: Facebook의 시계열 예측 모델
- **LSTM**: Long Short-Term Memory 네트워크
- **GRU**: Gated Recurrent Unit
- **Transformer**: Attention 기반 모델

## 모델 선택 전략

### 1. TFT 전용 모드 (`strategy: "tft_only"`)
```yaml
model:
  strategy: "tft_only"
```
- **용도**: 빠른 예측, 단일 모델 신뢰도
- **장점**: 빠른 실행, 안정적
- **단점**: 앙상블 효과 없음

### 2. 기존 앙상블 모드 (`strategy: "ensemble"`)
```yaml
model:
  strategy: "ensemble"
  use_ensemble: true
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]
```
- **용도**: TFT + Prophet 조합
- **장점**: 검증된 조합, 안정적
- **단점**: 제한된 모델 조합

### 3. 다중 모델 앙상블 (`strategy: "multi_model"`)
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
- **단점**: 긴 실행 시간, 복잡성

### 4. 자동 선택 모드 (`strategy: "auto_select"`)
```yaml
model:
  strategy: "auto_select"
```
- **용도**: 성능 기반 자동 선택
- **장점**: 데이터에 최적화된 모델 선택
- **단점**: 평가 시간 필요

## 권장 사용 시나리오

### 시나리오 1: 빠른 프로토타이핑
```yaml
model:
  strategy: "tft_only"
  use_ensemble: false
```

### 시나리오 2: 안정적인 프로덕션
```yaml
model:
  strategy: "ensemble"
  use_ensemble: true
  ensemble:
    methods: ["tft", "prophet"]
    weights: [0.7, 0.3]
```

### 시나리오 3: 최고 성능 추구
```yaml
model:
  strategy: "multi_model"
  multi_model_ensemble:
    enabled: true
    models: ["tft", "prophet", "lstm", "gru", "transformer"]
    weights: [0.4, 0.2, 0.15, 0.15, 0.1]
```

### 시나리오 4: 데이터 특성에 맞는 자동 선택
```yaml
model:
  strategy: "auto_select"
```

## 모델별 특징

### TFT (Temporal Fusion Transformer)
- **장점**: 다변량 시계열, 복잡한 패턴 학습
- **단점**: 긴 훈련 시간, 많은 메모리
- **적합한 데이터**: 다변량, 복잡한 시계열

### Prophet
- **장점**: 계절성 처리 우수, 해석 가능
- **단점**: 단변량만 지원
- **적합한 데이터**: 계절성이 강한 단변량 시계열

### LSTM
- **장점**: 긴 시퀀스 처리, 안정적
- **단점**: 단변량, 순차 처리
- **적합한 데이터**: 긴 의존성이 있는 시계열

### GRU
- **장점**: LSTM보다 빠름, 적은 파라미터
- **단점**: 단변량, 복잡한 패턴 처리 제한
- **적합한 데이터**: 중간 길이 의존성 시계열

### Transformer
- **장점**: 병렬 처리, 어텐션 메커니즘
- **단점**: 많은 데이터 필요, 긴 훈련 시간
- **적합한 데이터**: 복잡한 패턴, 충분한 데이터

## 실행 시간 비교

| 모델 조합 | 예상 실행 시간 | 메모리 사용량 |
|-----------|---------------|---------------|
| TFT만 | 5-10분 | 중간 |
| TFT + Prophet | 10-15분 | 중간 |
| 모든 모델 | 30-60분 | 높음 |

## 성능 최적화 팁

1. **데이터 크기에 따른 조정**:
   - 작은 데이터 (< 50 포인트): TFT만 사용
   - 중간 데이터 (50-200 포인트): TFT + Prophet
   - 큰 데이터 (> 200 포인트): 다중 모델 앙상블

2. **계산 리소스 고려**:
   - CPU만: TFT + Prophet
   - GPU 있음: 모든 모델 사용 가능

3. **실시간 요구사항**:
   - 빠른 응답 필요: TFT만
   - 정확도 우선: 다중 모델 앙상블

## 문제 해결

### 일반적인 오류
1. **메모리 부족**: 배치 크기 줄이기
2. **훈련 시간 오래**: 에포크 수 줄이기
3. **수렴 안됨**: 학습률 조정

### 디버깅
- 로그 레벨을 DEBUG로 설정
- 각 모델별 성능 확인
- 데이터 품질 검증 