"""
통신사 재무 예측 모델 모듈
Darts 라이브러리의 TFTModel을 활용한 다변량 시계열 예측
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from darts import TimeSeries
from darts.models import TFTModel, Prophet
from darts.metrics import mae, mape, rmse, smape
from darts.utils.statistics import check_seasonality
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelecomForecaster:
    """통신사 재무 예측 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """초기화"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def prepare_time_series(self, df: pd.DataFrame, target_columns: List[str]) -> Dict[str, TimeSeries]:
        """데이터프레임을 Darts TimeSeries로 변환"""
        time_series_dict = {}
        
        for col in target_columns:
            if col in df.columns:
                # 시계열 데이터 추출
                series_data = df[col].values
                dates = df.index
                
                # TimeSeries 객체 생성
                ts = TimeSeries(
                    times=dates,
                    values=series_data,
                    freq='MS'  # ✅
                )
                
                time_series_dict[col] = ts
                logger.info(f"TimeSeries 생성: {col} - {len(ts)}개 데이터포인트")
        
        return time_series_dict
    
    def create_tft_model(self, target_columns: List[str]) -> TFTModel:
        """TFT 모델 생성"""
        tft_config = self.config['model']['tft']
        
        # TFT 모델 초기화
        model = TFTModel(
            input_chunk_length=tft_config['input_chunk_length'],
            output_chunk_length=tft_config['output_chunk_length'],
            hidden_size=tft_config['hidden_size'],
            lstm_layers=tft_config['lstm_layers'],
            num_attention_heads=tft_config['num_attention_heads'],
            dropout=tft_config['dropout'],
            batch_size=tft_config['batch_size'],
            n_epochs=tft_config['n_epochs'],
            optimizer_kwargs=tft_config['optimizer_kwargs'],
            random_state=tft_config['random_state'],
            add_relative_index=True,  # 미래 공변량 문제 해결
            pl_trainer_kwargs={
                "accelerator": "auto",
                "devices": "auto"
            }
        )
        
        logger.info("TFT 모델 생성 완료")
        return model
    
    def create_prophet_model(self) -> Prophet:
        """Prophet 모델 생성"""
        prophet_config = self.config['model']['prophet']
        
        model = Prophet(
            yearly_seasonality=prophet_config['yearly_seasonality'],
            weekly_seasonality=prophet_config['weekly_seasonality'],
            daily_seasonality=prophet_config['daily_seasonality'],
            seasonality_mode=prophet_config['seasonality_mode']
        )
        
        logger.info("Prophet 모델 생성 완료")
        return model
    
    def train_models(self, time_series_dict: Dict[str, TimeSeries], 
                    target_columns: List[str]) -> Dict[str, object]:
        """모델 훈련"""
        models = {}
        
        # TFT 모델 훈련 (다변량)
        logger.info("TFT 모델 훈련 시작...")
        tft_model = self.create_tft_model(target_columns)
        
        # 모든 타겟 시계열을 하나의 다변량 시계열로 결합
        combined_series = []
        for col in target_columns:
            if col in time_series_dict:
                combined_series.append(time_series_dict[col])
        
        if combined_series:
            # 다변량 시계열 생성
            multivariate_series = TimeSeries(
                times=combined_series[0].time_index,
                values=np.column_stack([ts.values() for ts in combined_series]),
                components=target_columns,
                freq='MS'  # ✅
            )
            
            # TFT 모델 훈련
            tft_model.fit(multivariate_series)
            models['tft'] = tft_model
            logger.info("TFT 모델 훈련 완료")
        
        # 앙상블 사용 여부에 따라 Prophet 모델 훈련
        use_ensemble = self.config['model']['use_ensemble']
        if use_ensemble:
            logger.info("Prophet 모델 훈련 시작...")
            prophet_models = {}
            
            for col in target_columns:
                if col in time_series_dict:
                    prophet_model = self.create_prophet_model()
                    prophet_model.fit(time_series_dict[col])
                    prophet_models[col] = prophet_model
            
            models['prophet'] = prophet_models
            logger.info("Prophet 모델 훈련 완료")
        else:
            logger.info("앙상블 비활성화: TFT 모델만 사용")
        
        return models
    
    def make_predictions(self, models: Dict[str, object], 
                        time_series_dict: Dict[str, TimeSeries],
                        target_columns: List[str],
                        forecast_horizon: int = 12) -> Dict[str, pd.DataFrame]:
        """예측 수행"""
        predictions = {}
        
        # TFT 모델 예측
        if 'tft' in models:
            logger.info("TFT 모델 예측 시작...")
            tft_model = models['tft']
            
            # 다변량 시계열 생성
            combined_series = []
            for col in target_columns:
                if col in time_series_dict:
                    combined_series.append(time_series_dict[col])
            
            if combined_series:
                multivariate_series = TimeSeries(
                    times=combined_series[0].time_index,
                    values=np.column_stack([ts.values() for ts in combined_series]),
                    components=target_columns,
                    freq='MS'  # ✅
                )
                
                # 예측 수행
                tft_forecast = tft_model.predict(n=forecast_horizon)
                predictions['tft'] = tft_forecast.pd_dataframe()
                logger.info("TFT 모델 예측 완료")
        
        # 앙상블 사용 여부에 따라 Prophet 모델 예측
        use_ensemble = self.config['model']['use_ensemble']
        if use_ensemble and 'prophet' in models:
            logger.info("Prophet 모델 예측 시작...")
            prophet_models = models['prophet']
            prophet_predictions = {}
            
            for col in target_columns:
                if col in prophet_models and col in time_series_dict:
                    prophet_model = prophet_models[col]
                    prophet_forecast = prophet_model.predict(n=forecast_horizon)
                    prophet_predictions[col] = prophet_forecast.pd_dataframe()
            
            predictions['prophet'] = prophet_predictions
            logger.info("Prophet 모델 예측 완료")
        
        return predictions
    
    def ensemble_predictions(self, predictions: Dict[str, pd.DataFrame],
                           target_columns: List[str]) -> pd.DataFrame:
        """앙상블 예측"""
        use_ensemble = self.config['model']['use_ensemble']
        
        # TFT 예측 결과
        tft_pred = predictions.get('tft', pd.DataFrame())
        
        # 앙상블 사용하지 않는 경우 TFT 결과만 반환
        if not use_ensemble:
            logger.info("앙상블 비활성화: TFT 예측 결과만 사용")
            return tft_pred
        
        # 앙상블 사용하는 경우
        ensemble_config = self.config['model']['ensemble']
        weights = ensemble_config['weights']
        
        # Prophet 예측 결과 통합
        prophet_pred = pd.DataFrame()
        if 'prophet' in predictions:
            prophet_predictions = predictions['prophet']
            prophet_data = []
            for col in target_columns:
                if col in prophet_predictions:
                    prophet_data.append(prophet_predictions[col])
            if prophet_data:
                prophet_pred = pd.concat(prophet_data, axis=1)
        
        # 앙상블 계산
        if not tft_pred.empty and not prophet_pred.empty:
            # 컬럼명 맞추기
            common_cols = list(set(tft_pred.columns) & set(prophet_pred.columns))
            
            ensemble_result = pd.DataFrame()
            for col in common_cols:
                if col in tft_pred.columns and col in prophet_pred.columns:
                    ensemble_result[col] = (
                        weights[0] * tft_pred[col] + 
                        weights[1] * prophet_pred[col]
                    )
            
            logger.info("앙상블 예측 완료")
            return ensemble_result
        
        # 하나의 모델만 있는 경우
        elif not tft_pred.empty:
            return tft_pred
        elif not prophet_pred.empty:
            return prophet_pred
        else:
            return pd.DataFrame()
    
    def evaluate_models(self, models: Dict[str, object],
                       time_series_dict: Dict[str, TimeSeries],
                       target_columns: List[str],
                       test_size: int = 6) -> Dict[str, Dict[str, float]]:
        """모델 평가"""
        evaluation_results = {}
        
        # TFT 모델 평가
        if 'tft' in models:
            logger.info("TFT 모델 평가 시작...")
            tft_model = models['tft']
            
            # 다변량 시계열 생성
            combined_series = []
            for col in target_columns:
                if col in time_series_dict:
                    combined_series.append(time_series_dict[col])
            
            if combined_series:
                multivariate_series = TimeSeries(
                    times=combined_series[0].time_index,
                    values=np.column_stack([ts.values() for ts in combined_series]),
                    components=target_columns,
                    freq='MS'  # ✅
                )
                
                # 훈련/테스트 분할
                train_series = multivariate_series[:-test_size]
                test_series = multivariate_series[-test_size:]
                
                # 예측
                forecast = tft_model.predict(n=test_size)
                
                # 평가 지표 계산
                tft_metrics = {}
                for col in target_columns:
                    if col in test_series.columns and col in forecast.columns:
                        actual = test_series[col].values()
                        predicted = forecast[col].values()
                        
                        tft_metrics[col] = {
                            'mae': mae(actual, predicted),
                            'mape': mape(actual, predicted),
                            'rmse': rmse(actual, predicted),
                            'smape': smape(actual, predicted)
                        }
                
                evaluation_results['tft'] = tft_metrics
                logger.info("TFT 모델 평가 완료")
        
        # 앙상블 사용 여부에 따라 Prophet 모델 평가
        use_ensemble = self.config['model']['use_ensemble']
        if use_ensemble and 'prophet' in models:
            logger.info("Prophet 모델 평가 시작...")
            prophet_models = models['prophet']
            prophet_metrics = {}
            
            for col in target_columns:
                if col in prophet_models and col in time_series_dict:
                    prophet_model = prophet_models[col]
                    series = time_series_dict[col]
                    
                    # 훈련/테스트 분할
                    train_series = series[:-test_size]
                    test_series = series[-test_size:]
                    
                    # 예측
                    forecast = prophet_model.predict(n=test_size)
                    
                    # 평가 지표 계산
                    actual = test_series.values()
                    predicted = forecast.values()
                    
                    prophet_metrics[col] = {
                        'mae': mae(actual, predicted),
                        'mape': mape(actual, predicted),
                        'rmse': rmse(actual, predicted),
                        'smape': smape(actual, predicted)
                    }
            
            evaluation_results['prophet'] = prophet_metrics
            logger.info("Prophet 모델 평가 완료")
        
        return evaluation_results
    
    def run_forecast_pipeline(self, processed_data: pd.DataFrame,
                            target_columns: List[str],
                            forecast_horizon: int = 12) -> Dict:
        """전체 예측 파이프라인 실행"""
        logger.info("=== 예측 파이프라인 시작 ===")
        
        # 1. TimeSeries 변환
        time_series_dict = self.prepare_time_series(processed_data, target_columns)
        
        # 2. 모델 훈련
        models = self.train_models(time_series_dict, target_columns)
        
        # 3. 예측 수행
        predictions = self.make_predictions(models, time_series_dict, target_columns, forecast_horizon)
        
        # 4. 앙상블 예측
        ensemble_forecast = self.ensemble_predictions(predictions, target_columns)
        
        # 5. 모델 평가
        evaluation_results = self.evaluate_models(models, time_series_dict, target_columns)
        
        # 6. 결과 저장
        results = {
            'models': models,
            'predictions': predictions,
            'ensemble_forecast': ensemble_forecast,
            'evaluation_results': evaluation_results,
            'time_series_dict': time_series_dict
        }
        
        logger.info("=== 예측 파이프라인 완료 ===")
        return results 