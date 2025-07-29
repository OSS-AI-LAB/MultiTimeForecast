"""
통신사 손익전망 예측 메인 로직
TimesFM 모델을 활용한 다변량 시계열 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_processing import TelecomDataProcessor, generate_sample_data
from tft_model import TFTWrapper

class TelecomForecaster:
    """통신사 손익전망 예측 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.data_processor = TelecomDataProcessor(config)
        self.model = None
        self.target_columns = []
        self.forecast_horizon = 12
        self.covariate_columns = []
        
        # 기본 설정
        self.default_config = {
            'forecast_horizon': 12,
            'validation_split': 0.2,
            'n_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'num_attention_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dropout': 0.1,
            'input_chunk_length': 12,
            'output_chunk_length': 12
        }
        
        # 설정 업데이트
        self.default_config.update(self.config)
        self.config = self.default_config
        
    def prepare_forecasting_data(self, 
                               file_path: str,
                               target_columns: List[str],
                               covariate_columns: Optional[List[str]] = None,
                               forecast_horizon: int = 12) -> pd.DataFrame:
        """
        예측용 데이터 준비
        
        Args:
            file_path: 데이터 파일 경로
            target_columns: 예측할 타겟 컬럼들
            covariate_columns: 공변량 컬럼들
            forecast_horizon: 예측 기간
            
        Returns:
            전처리된 데이터프레임
        """
        self.target_columns = target_columns
        self.covariate_columns = covariate_columns or []
        self.forecast_horizon = forecast_horizon
        
        # 데이터 로드 및 전처리
        df = self.data_processor.load_data(file_path)
        
        if not self.data_processor.validate_data(df):
            raise ValueError("데이터 검증에 실패했습니다.")
        
        df_clean = self.data_processor.clean_data(df)
        df_processed = self.data_processor.create_features(df_clean)
        
        return df_processed
    
    def train_model(self, 
                   df_processed: pd.DataFrame,
                   model_config: Optional[Dict] = None) -> Dict[str, List[float]]:
        """
        TFT 모델 훈련
        
        Args:
            df_processed: 전처리된 데이터프레임
            model_config: 모델 설정
            
        Returns:
            훈련 히스토리
        """
        # 모델 설정
        if model_config is None:
            model_config = {}
        
        model_params = {
            'input_chunk_length': self.config['input_chunk_length'],
            'output_chunk_length': self.config['output_chunk_length'],
            'hidden_size': self.config['hidden_size'],
            'num_attention_heads': self.config['num_attention_heads'],
            'num_encoder_layers': self.config['num_encoder_layers'],
            'num_decoder_layers': self.config['num_decoder_layers'],
            'dropout': self.config['dropout'],
            'batch_size': self.config['batch_size'],
            'n_epochs': self.config['n_epochs'],
            'learning_rate': self.config['learning_rate']
        }
        model_params.update(model_config)
        
        # 모델 초기화
        self.model = TFTWrapper(**model_params)
        
        # 모델 훈련
        history = self.model.train(
            df_processed,
            self.target_columns,
            self.covariate_columns,
            validation_split=self.config['validation_split']
        )
        
        return history
    
    def generate_forecast(self, 
                         df_processed: pd.DataFrame,
                         forecast_steps: int = 12) -> pd.DataFrame:
        """
        미래 예측 생성
        
        Args:
            df_processed: 전처리된 데이터
            forecast_steps: 예측할 스텝 수
            
        Returns:
            예측 결과 데이터프레임
        """
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다. 먼저 train_model()을 실행하세요.")
        
        # 예측 수행
        forecast_df = self.model.predict(
            df_processed, 
            n_steps=forecast_steps,
            covariate_columns=self.covariate_columns
        )
        
        return forecast_df
    
    def analyze_profitability(self, 
                            df_processed: pd.DataFrame,
                            forecast_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        수익성 분석
        
        Args:
            df_processed: 전처리된 과거 데이터
            forecast_df: 예측 데이터
            
        Returns:
            수익성 분석 결과
        """
        # 과거 데이터와 예측 데이터 결합
        combined_df = pd.concat([
            df_processed[['date'] + self.target_columns],
            forecast_df[['date'] + self.target_columns]
        ], ignore_index=True)
        
        # 수익성 지표 계산
        if 'total_revenue' in combined_df.columns and 'total_cost' in combined_df.columns:
            combined_df['profit'] = combined_df['total_revenue'] - combined_df['total_cost']
            combined_df['profit_margin'] = combined_df['profit'] / combined_df['total_revenue']
        else:
            # 수익성 지표가 없는 경우 기본 계산
            revenue_cols = [col for col in combined_df.columns if 'revenue' in col]
            cost_cols = [col for col in combined_df.columns if 'cost' in col]
            
            if revenue_cols and cost_cols:
                combined_df['total_revenue'] = combined_df[revenue_cols].sum(axis=1)
                combined_df['total_cost'] = combined_df[cost_cols].sum(axis=1)
                combined_df['profit'] = combined_df['total_revenue'] - combined_df['total_cost']
                combined_df['profit_margin'] = combined_df['profit'] / combined_df['total_revenue']
        
        # 기술별 분석
        tech_analysis = {}
        
        # 5G 분석
        if '5g_users' in combined_df.columns:
            tech_analysis['5g'] = combined_df[['date', '5g_users', '5g_revenue', '5g_cost']].copy()
            if '5g_revenue' in combined_df.columns and '5g_cost' in combined_df.columns:
                tech_analysis['5g']['5g_profit'] = combined_df['5g_revenue'] - combined_df['5g_cost']
                tech_analysis['5g']['5g_profit_margin'] = tech_analysis['5g']['5g_profit'] / combined_df['5g_revenue']
        
        # LTE 분석
        if 'lte_users' in combined_df.columns:
            tech_analysis['lte'] = combined_df[['date', 'lte_users', 'lte_revenue', 'lte_cost']].copy()
            if 'lte_revenue' in combined_df.columns and 'lte_cost' in combined_df.columns:
                tech_analysis['lte']['lte_profit'] = combined_df['lte_revenue'] - combined_df['lte_cost']
                tech_analysis['lte']['lte_profit_margin'] = tech_analysis['lte']['lte_profit'] / combined_df['lte_revenue']
        
        # 3G 분석
        if '3g_users' in combined_df.columns:
            tech_analysis['3g'] = combined_df[['date', '3g_users', '3g_revenue', '3g_cost']].copy()
            if '3g_revenue' in combined_df.columns and '3g_cost' in combined_df.columns:
                tech_analysis['3g']['3g_profit'] = combined_df['3g_revenue'] - combined_df['3g_cost']
                tech_analysis['3g']['3g_profit_margin'] = tech_analysis['3g']['3g_profit'] / combined_df['3g_revenue']
        
        return {
            'combined': combined_df,
            'tech_analysis': tech_analysis
        }
    
    def save_results(self, 
                    forecast_df: pd.DataFrame,
                    analysis_results: Dict[str, pd.DataFrame],
                    output_dir: str = 'results'):
        """
        결과 저장
        
        Args:
            forecast_df: 예측 결과
            analysis_results: 분석 결과
            output_dir: 출력 디렉토리
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 예측 결과 저장
        forecast_df.to_csv(f'{output_dir}/forecast_results.csv', index=False)
        
        # 수익성 분석 결과 저장
        analysis_results['combined'].to_csv(f'{output_dir}/profitability_analysis.csv', index=False)
        
        # 기술별 분석 결과 저장
        for tech, data in analysis_results['tech_analysis'].items():
            data.to_csv(f'{output_dir}/{tech}_analysis.csv', index=False)
        
        # 모델 저장
        if self.model:
            self.model.save_model(f'{output_dir}/tft_model.pth')
        
        print(f"결과가 {output_dir} 디렉토리에 저장되었습니다.")
    
    def run_full_pipeline(self, 
                         file_path: str,
                         target_columns: List[str],
                         forecast_steps: int = 12,
                         model_config: Optional[Dict] = None) -> Dict:
        """
        전체 예측 파이프라인 실행
        
        Args:
            file_path: 데이터 파일 경로
            target_columns: 예측할 타겟 컬럼들
            forecast_steps: 예측할 스텝 수
            model_config: 모델 설정
            
        Returns:
            전체 결과 딕셔너리
        """
        print("=== 통신사 손익전망 예측 파이프라인 시작 ===")
        
        # 1. 데이터 준비
        print("\n1. 데이터 준비 중...")
        df_processed = self.prepare_forecasting_data(
            file_path, target_columns, None, self.config['forecast_horizon']
        )
        
        # 2. 모델 훈련
        print("\n2. 모델 훈련 중...")
        history = self.train_model(df_processed, model_config)
        
        # 3. 예측 생성
        print("\n3. 예측 생성 중...")
        forecast_df = self.generate_forecast(df_processed, forecast_steps)
        
        # 4. 수익성 분석
        print("\n4. 수익성 분석 중...")
        analysis_results = self.analyze_profitability(df_processed, forecast_df)
        
        # 5. 결과 저장
        print("\n5. 결과 저장 중...")
        self.save_results(forecast_df, analysis_results)
        
        print("\n=== 파이프라인 완료 ===")
        
        return {
            'processed_data': df_processed,
            'forecast': forecast_df,
            'analysis': analysis_results,
            'history': history
        }

def main():
    """메인 실행 함수"""
    
    # 샘플 데이터 생성 (실제 데이터가 없는 경우)
    print("샘플 데이터 생성 중...")
    sample_data = generate_sample_data()
    sample_data.to_csv('data/raw/sample_telecom_data.csv', index=False)
    
    # 예측기 초기화
    forecaster = TelecomForecaster()
    
    # 타겟 컬럼 정의 (수익성 관련 지표들)
    target_columns = [
        '5g_users', 'lte_users', '3g_users',
        '5g_revenue', 'lte_revenue', '3g_revenue',
        '5g_cost', 'lte_cost', '3g_cost',
        'total_revenue', 'total_cost', 'profit', 'profit_margin'
    ]
    
    # 전체 파이프라인 실행
    results = forecaster.run_full_pipeline(
        file_path='data/raw/sample_telecom_data.csv',
        target_columns=target_columns,
        forecast_steps=12
    )
    
    # 결과 요약 출력
    print("\n=== 예측 결과 요약 ===")
    print(f"예측 기간: {len(results['forecast'])}개월")
    print(f"예측된 컬럼 수: {len(target_columns)}")
    
    # 최종 예측값 출력
    print("\n최종 예측값 (12개월 후):")
    final_forecast = results['forecast'].iloc[-1]
    for col in target_columns:
        if col in final_forecast:
            print(f"  {col}: {final_forecast[col]:.2f}")

if __name__ == "__main__":
    main() 