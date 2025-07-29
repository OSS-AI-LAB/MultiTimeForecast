"""
TFT (Temporal Fusion Transformer) 모델 래퍼
darts 라이브러리의 TFTModel을 통신사 시계열 예측에 맞게 래핑
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, mae, rmse, mse
from sklearn.preprocessing import StandardScaler

class TFTWrapper:
    """TFT 모델 래퍼 클래스"""
    
    def __init__(self, 
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 12,
                 hidden_size: int = 64,
                 num_attention_heads: int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Args:
            input_chunk_length: 입력 시퀀스 길이
            output_chunk_length: 출력 시퀀스 길이
            hidden_size: 은닉층 크기
            num_attention_heads: 어텐션 헤드 수
            num_encoder_layers: 인코더 레이어 수
            num_decoder_layers: 디코더 레이어 수
            dropout: 드롭아웃 비율
            batch_size: 배치 크기
            n_epochs: 훈련 에포크 수
            learning_rate: 학습률
            random_state: 랜덤 시드
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 모델 초기화
        self.model = None
        self.scaler = StandardScaler()
        self.target_columns = []
        
        print("TFT 모델이 초기화되었습니다.")
    
    def _build_model(self) -> TFTModel:
        """TFT 모델 구축"""
        return TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            pl_trainer_kwargs={"accelerator": "auto", "devices": "auto"}
        )
    
    def prepare_darts_data(self, 
                          df: pd.DataFrame,
                          target_columns: List[str],
                          covariate_columns: Optional[List[str]] = None) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """
        darts TimeSeries 형식으로 데이터 준비
        
        Args:
            df: 데이터프레임
            target_columns: 타겟 컬럼들
            covariate_columns: 공변량 컬럼들
            
        Returns:
            (타겟 시계열 리스트, 공변량 시계열 리스트)
        """
        # 날짜를 인덱스로 설정
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        
        # 타겟 시계열 생성
        target_series_list = []
        for col in target_columns:
            if col in df_copy.columns:
                series = TimeSeries.from_series(df_copy[col])
                target_series_list.append(series)
        
        # 공변량 시계열 생성
        covariate_series_list = []
        if covariate_columns:
            for col in covariate_columns:
                if col in df_copy.columns:
                    series = TimeSeries.from_series(df_copy[col])
                    covariate_series_list.append(series)
        
        # 시간 특성 추가
        time_covariates = self._create_time_covariates(df_copy.index)
        covariate_series_list.extend(time_covariates)
        
        return target_series_list, covariate_series_list
    
    def _create_time_covariates(self, time_index: pd.DatetimeIndex) -> List[TimeSeries]:
        """시간 특성 생성"""
        covariates = []
        
        # 월, 분기, 연도 특성
        month_series = datetime_attribute_timeseries(time_index, attribute='month', freq='M')
        quarter_series = datetime_attribute_timeseries(time_index, attribute='quarter', freq='M')
        year_series = datetime_attribute_timeseries(time_index, attribute='year', freq='M')
        
        covariates.extend([month_series, quarter_series, year_series])
        
        return covariates
    
    def train(self, 
              df: pd.DataFrame,
              target_columns: List[str],
              covariate_columns: Optional[List[str]] = None,
              validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        모델 훈련
        
        Args:
            df: 훈련 데이터
            target_columns: 타겟 컬럼들
            covariate_columns: 공변량 컬럼들
            validation_split: 검증 데이터 비율
            
        Returns:
            훈련 히스토리
        """
        self.target_columns = target_columns
        
        print("데이터 준비 중...")
        target_series_list, covariate_series_list = self.prepare_darts_data(
            df, target_columns, covariate_columns
        )
        
        # 데이터 분할
        split_point = int(len(df) * (1 - validation_split))
        
        train_targets = [series[:split_point] for series in target_series_list]
        val_targets = [series[split_point:] for series in target_series_list]
        
        train_covariates = [series[:split_point] for series in covariate_series_list]
        val_covariates = [series[split_point:] for series in covariate_series_list]
        
        # 모델 초기화
        print("TFT 모델 구축 중...")
        self.model = self._build_model()
        
        # 모델 훈련
        print("모델 훈련 시작...")
        self.model.fit(
            series=train_targets,
            past_covariates=train_covariates if train_covariates else None,
            val_series=val_targets,
            val_past_covariates=val_covariates if val_covariates else None
        )
        
        print("훈련 완료!")
        
        # 검증 성능 평가
        if val_targets:
            val_predictions = self.model.predict(
                n=self.output_chunk_length,
                series=train_targets,
                past_covariates=train_covariates if train_covariates else None
            )
            
            metrics = {}
            for i, (pred, actual) in enumerate(zip(val_predictions, val_targets)):
                col_name = target_columns[i] if i < len(target_columns) else f"target_{i}"
                metrics[f"{col_name}_mape"] = mape(actual, pred)
                metrics[f"{col_name}_mae"] = mae(actual, pred)
                metrics[f"{col_name}_rmse"] = rmse(actual, pred)
            
            print("검증 성능:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return {"training_completed": True}
    
    def predict(self, 
               df: pd.DataFrame,
               n_steps: int = 12,
               covariate_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        예측 수행
        
        Args:
            df: 입력 데이터
            n_steps: 예측할 스텝 수
            covariate_columns: 공변량 컬럼들
            
        Returns:
            예측 결과 데이터프레임
        """
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다. 먼저 train()을 실행하세요.")
        
        print("예측 데이터 준비 중...")
        target_series_list, covariate_series_list = self.prepare_darts_data(
            df, self.target_columns, covariate_columns
        )
        
        print("예측 수행 중...")
        predictions = self.model.predict(
            n=n_steps,
            series=target_series_list,
            past_covariates=covariate_series_list if covariate_series_list else None
        )
        
        # 예측 결과를 데이터프레임으로 변환
        forecast_df = self._convert_predictions_to_df(predictions, n_steps)
        
        return forecast_df
    
    def _convert_predictions_to_df(self, 
                                 predictions: List[TimeSeries], 
                                 n_steps: int) -> pd.DataFrame:
        """예측 결과를 데이터프레임으로 변환"""
        # 예측 날짜 생성
        last_date = predictions[0].time_index[-1] if predictions else pd.Timestamp.now()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_steps,
            freq='M'
        )
        
        # 예측값 추출
        forecast_data = {}
        for i, (pred_series, col_name) in enumerate(zip(predictions, self.target_columns)):
            if i < len(predictions):
                values = pred_series.values()[-n_steps:]  # 마지막 n_steps만 사용
                forecast_data[col_name] = values.flatten()
        
        forecast_data['date'] = forecast_dates
        
        return pd.DataFrame(forecast_data)
    
    def evaluate(self, 
                df: pd.DataFrame,
                target_columns: List[str],
                covariate_columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            df: 평가 데이터
            target_columns: 타겟 컬럼들
            covariate_columns: 공변량 컬럼들
            
        Returns:
            평가 지표들
        """
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 데이터 준비
        target_series_list, covariate_series_list = self.prepare_darts_data(
            df, target_columns, covariate_columns
        )
        
        # 예측 수행
        predictions = self.model.predict(
            n=self.output_chunk_length,
            series=target_series_list,
            past_covariates=covariate_series_list if covariate_series_list else None
        )
        
        # 평가 지표 계산
        metrics = {}
        for i, (pred, actual) in enumerate(zip(predictions, target_series_list)):
            col_name = target_columns[i] if i < len(target_columns) else f"target_{i}"
            metrics[f"{col_name}_mape"] = mape(actual, pred)
            metrics[f"{col_name}_mae"] = mae(actual, pred)
            metrics[f"{col_name}_rmse"] = rmse(actual, pred)
            metrics[f"{col_name}_mse"] = mse(actual, pred)
        
        return metrics
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        self.model = TFTModel.load(filepath)
        print(f"모델이 {filepath}에서 로드되었습니다.")

def main():
    """테스트 코드"""
    print("TFT 모델 래퍼 테스트")
    
    # 샘플 데이터 생성
    from data_processing import generate_sample_data
    
    sample_data = generate_sample_data()
    
    # TFT 모델 초기화
    tft_model = TFTWrapper(
        input_chunk_length=12,
        output_chunk_length=12,
        n_epochs=50  # 테스트용으로 에포크 수 줄임
    )
    
    # 타겟 컬럼 정의
    target_columns = ['5g_users', 'lte_users', '3g_users']
    
    # 모델 훈련
    history = tft_model.train(sample_data, target_columns)
    
    # 예측
    forecast = tft_model.predict(sample_data, n_steps=12)
    print("예측 완료!")
    print(forecast.head())

if __name__ == "__main__":
    main() 