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
        """데이터프레임을 Darts TimeSeries로 변환 (단순화 버전)"""
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
                    freq='MS'
                )
                
                time_series_dict[col] = ts
                logger.info(f"TimeSeries 생성: {col} - {len(ts)}개 데이터포인트")
        
        # TFTModel은 add_relative_index=True로 설정했으므로 
        # future_covariates를 자동으로 생성함 - 수동 생성 불필요
        logger.info("add_relative_index=True로 future_covariates 자동 생성 설정됨")
        
        return time_series_dict
    
    def create_tft_model(self, target_columns: List[str], data_length: int = 29) -> TFTModel:
        """TFT 모델 생성 - 최신 파라미터 및 동적 조정 적용"""
        tft_config = self.config['model']['tft']
        
        # 데이터 길이에 따라 chunk length 동적 조정
        max_input_chunk = min(tft_config['input_chunk_length'], data_length // 3)
        max_output_chunk = min(tft_config['output_chunk_length'], data_length // 6)
        
        # 최소값 보장
        input_chunk_length = max(2, max_input_chunk)
        output_chunk_length = max(1, max_output_chunk)
        
        logger.info(f"데이터 길이: {data_length}, input_chunk: {input_chunk_length}, output_chunk: {output_chunk_length}")
        
        # TFT 모델 초기화 (최신 버전 파라미터)
        model = TFTModel(
            # 필수 파라미터 (동적 조정)
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            
            # 모델 아키텍처 파라미터
            hidden_size=tft_config['hidden_size'],
            lstm_layers=tft_config['lstm_layers'],
            num_attention_heads=tft_config['num_attention_heads'],
            full_attention=tft_config.get('full_attention', False),
            feed_forward=tft_config.get('feed_forward', 'GatedResidualNetwork'),
            dropout=tft_config['dropout'],
            hidden_continuous_size=tft_config.get('hidden_continuous_size', 8),
            
            # 특성 관련 파라미터
            add_relative_index=True,  # future_covariates 자동 생성
            use_static_covariates=tft_config.get('use_static_covariates', True),
            norm_type=tft_config.get('norm_type', 'LayerNorm'),
            
            # 훈련 관련 파라미터 (PyTorch Lightning kwargs로 전달)
            n_epochs=tft_config['n_epochs'],
            batch_size=min(tft_config['batch_size'], 16),  # 작은 데이터셋에 맞게 조정
            optimizer_kwargs=tft_config['optimizer_kwargs'],
            random_state=tft_config['random_state'],
            
            # PyTorch Lightning Trainer 설정
            pl_trainer_kwargs={
                "accelerator": "auto",
                "devices": "auto",
                "enable_progress_bar": True,
                "enable_model_summary": False
            },
            
            # 기타 설정
            save_checkpoints=False,  # 체크포인트 저장 비활성화
            force_reset=True,        # 기존 모델 초기화
            show_warnings=False      # 경고 메시지 숨김
        )
        
        logger.info("TFT 모델 생성 완료 (최신 파라미터, 동적 조정)")
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
        """모델 훈련 (단순화 버전)"""
        models = {}
        
        # 데이터 길이 확인
        if target_columns and target_columns[0] in time_series_dict:
            data_length = len(time_series_dict[target_columns[0]])
        else:
            data_length = 29  # 기본값
        
        # TFT 모델 훈련 (다변량)
        logger.info("TFT 모델 훈련 시작...")
        tft_model = self.create_tft_model(target_columns, data_length)
        
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
                freq='MS'
            )
            
            # TFT 모델 훈련 - add_relative_index=True이므로 자동으로 future_covariates 생성
            tft_model.fit(multivariate_series)
            
            models['tft'] = tft_model
            models['multivariate_series'] = multivariate_series
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
        """예측 수행 - add_relative_index 사용으로 단순화"""
        predictions = {}
        
        # TFT 모델 예측
        if 'tft' in models and 'multivariate_series' in models:
            logger.info("TFT 모델 예측 시작...")
            tft_model = models['tft']
            
            # add_relative_index=True이므로 자동으로 future_covariates 생성됨
            tft_forecast = tft_model.predict(n=forecast_horizon)
            predictions['tft'] = tft_forecast.to_dataframe()
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
                    prophet_predictions[col] = prophet_forecast.to_dataframe()
            
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
                    col_data = prophet_predictions[col]
                    if isinstance(col_data, pd.DataFrame):
                        # DataFrame인 경우 첫 번째 컬럼 사용
                        prophet_data.append(col_data.iloc[:, 0].rename(col))
                    else:
                        prophet_data.append(col_data)
            
            if prophet_data:
                prophet_pred = pd.concat(prophet_data, axis=1)
        
        # 앙상블 계산
        if not tft_pred.empty and not prophet_pred.empty:
            # 컬럼명 맞추기
            common_cols = list(set(tft_pred.columns) & set(prophet_pred.columns))
            
            ensemble_result = pd.DataFrame(index=tft_pred.index)
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
    
    def safe_metric_calculation(self, actual: np.ndarray, predicted: np.ndarray, metric_name: str) -> float:
        """안전한 메트릭 계산 (오류 처리 포함)"""
        try:
            # 배열 차원 확인 및 1차원으로 변환
            if actual.ndim > 1:
                actual = actual.flatten()
            if predicted.ndim > 1:
                predicted = predicted.flatten()
            
            # NaN이나 무한대 값 제거
            mask = np.isfinite(actual) & np.isfinite(predicted)
            if not np.any(mask):
                return np.nan
            
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return np.nan
            
            # TimeSeries 객체 생성 (차원 문제 해결)
            try:
                # 1차원 배열로 변환
                actual_1d = actual_clean.flatten() if actual_clean.ndim > 1 else actual_clean
                predicted_1d = predicted_clean.flatten() if predicted_clean.ndim > 1 else predicted_clean
                
                actual_ts = TimeSeries.from_values(actual_1d)
                predicted_ts = TimeSeries.from_values(predicted_1d)
            except Exception as e:
                logger.warning(f"TimeSeries 생성 실패 ({metric_name}): {e}")
                return np.nan
            
            # 길이가 다른 경우 짧은 쪽에 맞춤
            min_len = min(len(actual_ts), len(predicted_ts))
            if min_len == 0:
                return np.nan
                
            actual_ts = actual_ts[:min_len]
            predicted_ts = predicted_ts[:min_len]
            
            # 메트릭 계산
            if metric_name == 'mae':
                return mae(actual_ts, predicted_ts)
            elif metric_name == 'mape':
                return mape(actual_ts, predicted_ts)
            elif metric_name == 'rmse':
                return rmse(actual_ts, predicted_ts)
            elif metric_name == 'smape':
                return smape(actual_ts, predicted_ts)
            else:
                return np.nan
                
        except Exception as e:
            logger.warning(f"메트릭 계산 실패 ({metric_name}): {e}")
            return np.nan
    
    def evaluate_models(self, models: Dict[str, object],
                       time_series_dict: Dict[str, TimeSeries],
                       target_columns: List[str],
                       test_size: int = 6) -> Dict[str, Dict[str, Dict[str, float]]]:
        """모델 평가 - 개선된 버전"""
        evaluation_results = {}
        
        # TFT 모델 평가
        if 'tft' in models and 'multivariate_series' in models:
            logger.info("TFT 모델 평가 시작...")
            try:
                tft_model = models['tft']
                multivariate_series = models['multivariate_series']
                
                # 시계열 길이 확인
                if len(multivariate_series) <= test_size:
                    logger.warning(f"시계열이 너무 짧아 평가 불가: {len(multivariate_series)} <= {test_size}")
                    evaluation_results['tft'] = {}
                else:
                    # 훈련/테스트 분할
                    train_series = multivariate_series[:-test_size]
                    test_series = multivariate_series[-test_size:]
                    
                    # 예측
                    forecast = tft_model.predict(n=test_size)
                    
                    # 평가 지표 계산
                    tft_metrics = {}
                    for col in target_columns:
                        try:
                            if hasattr(test_series, 'components') and col in test_series.components:
                                # 다변량 시계열에서 특정 컴포넌트 추출
                                actual_values = test_series.to_dataframe()[col].values
                                predicted_values = forecast.to_dataframe()[col].values
                            elif col in test_series.to_dataframe().columns:
                                actual_values = test_series.to_dataframe()[col].values
                                predicted_values = forecast.to_dataframe()[col].values
                            else:
                                logger.warning(f"컬럼 {col}을 찾을 수 없음")
                                continue
                            
                            # 메트릭 계산
                            col_metrics = {}
                            for metric_name in ['mae', 'mape', 'rmse', 'smape']:
                                col_metrics[metric_name] = self.safe_metric_calculation(
                                    actual_values, predicted_values, metric_name
                                )
                            
                            tft_metrics[col] = col_metrics
                            
                        except Exception as e:
                            logger.warning(f"TFT 평가 실패 (컬럼: {col}): {e}")
                            tft_metrics[col] = {
                                'mae': np.nan, 'mape': np.nan, 'rmse': np.nan, 'smape': np.nan
                            }
                    
                    evaluation_results['tft'] = tft_metrics
                    logger.info("TFT 모델 평가 완료")
                    
            except Exception as e:
                logger.error(f"TFT 모델 평가 전체 실패: {e}")
                evaluation_results['tft'] = {}
        
        # 앙상블 사용 여부에 따라 Prophet 모델 평가
        use_ensemble = self.config['model']['use_ensemble']
        if use_ensemble and 'prophet' in models:
            logger.info("Prophet 모델 평가 시작...")
            try:
                prophet_models = models['prophet']
                prophet_metrics = {}
                
                for col in target_columns:
                    if col in prophet_models and col in time_series_dict:
                        try:
                            prophet_model = prophet_models[col]
                            series = time_series_dict[col]
                            
                            # 시계열 길이 확인
                            if len(series) <= test_size:
                                logger.warning(f"시계열이 너무 짧아 평가 불가 (컬럼: {col}): {len(series)} <= {test_size}")
                                continue
                            
                            # 훈련/테스트 분할
                            train_series = series[:-test_size]
                            test_series = series[-test_size:]
                            
                            # 예측
                            forecast = prophet_model.predict(n=test_size)
                            
                            # 평가 지표 계산
                            actual_values = test_series.values()
                            predicted_values = forecast.values()
                            
                            col_metrics = {}
                            for metric_name in ['mae', 'mape', 'rmse', 'smape']:
                                col_metrics[metric_name] = self.safe_metric_calculation(
                                    actual_values, predicted_values, metric_name
                                )
                            
                            prophet_metrics[col] = col_metrics
                            
                        except Exception as e:
                            logger.warning(f"Prophet 평가 실패 (컬럼: {col}): {e}")
                            prophet_metrics[col] = {
                                'mae': np.nan, 'mape': np.nan, 'rmse': np.nan, 'smape': np.nan
                            }
                
                evaluation_results['prophet'] = prophet_metrics
                logger.info("Prophet 모델 평가 완료")
                
            except Exception as e:
                logger.error(f"Prophet 모델 평가 전체 실패: {e}")
                evaluation_results['prophet'] = {}
        
        return evaluation_results
    
    def run_forecast_pipeline(self, processed_data: pd.DataFrame,
                            target_columns: List[str],
                            forecast_horizon: int = 12) -> Dict:
        """전체 예측 파이프라인 실행"""
        logger.info("=== 예측 파이프라인 시작 ===")
        
        try:
            # 1. TimeSeries 변환
            time_series_dict = self.prepare_time_series(processed_data, target_columns)
            
            # 2. 모델 훈련
            models = self.train_models(time_series_dict, target_columns)
            
            # 3. 예측 수행
            predictions = self.make_predictions(models, time_series_dict, target_columns, forecast_horizon)
            
            # 4. 앙상블 예측
            ensemble_forecast = self.ensemble_predictions(predictions, target_columns)
            
            # 5. 모델 평가 (에러가 발생해도 계속 진행)
            try:
                evaluation_results = self.evaluate_models(models, time_series_dict, target_columns)
            except Exception as e:
                logger.warning(f"모델 평가 중 오류 발생, 건너뜀: {e}")
                evaluation_results = {}
            
            # 6. 결과 저장
            results = {
                'models': models,
                'predictions': predictions,
                'ensemble_forecast': ensemble_forecast,
                'evaluation_results': evaluation_results,
                'time_series_dict': time_series_dict,
                'hierarchical_data': {}  # 빈 dict로 초기화
            }
            
            logger.info("=== 예측 파이프라인 완료 ===")
            return results
            
        except Exception as e:
            logger.error(f"예측 파이프라인 실행 중 오류: {e}")
            raise