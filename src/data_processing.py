"""
통신사 시계열 데이터 전처리 모듈
5G, 3G, LTE 등의 월별 데이터를 처리하고 TimesFM 모델에 맞는 형식으로 변환
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TelecomDataProcessor:
    """통신사 데이터 전처리 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.required_columns = [
            'date', '5g_users', 'lte_users', '3g_users',
            '5g_revenue', 'lte_revenue', '3g_revenue',
            '5g_cost', 'lte_cost', '3g_cost'
        ]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        데이터 파일 로드
        
        Args:
            file_path: 데이터 파일 경로
            
        Returns:
            로드된 데이터프레임
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 사용하세요.")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        데이터 유효성 검사
        
        Args:
            df: 검사할 데이터프레임
            
        Returns:
            유효성 검사 결과
        """
        # 필수 컬럼 확인
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            print(f"누락된 컬럼: {missing_cols}")
            return False
        
        # 날짜 형식 확인
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                print("날짜 컬럼을 datetime 형식으로 변환할 수 없습니다.")
                return False
        
        # 수치형 데이터 확인
        numeric_cols = [col for col in df.columns if col != 'date']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"수치형이 아닌 컬럼: {col}")
                return False
        
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정리 (결측치 처리, 이상치 제거 등)
        
        Args:
            df: 정리할 데이터프레임
            
        Returns:
            정리된 데이터프레임
        """
        df_clean = df.copy()
        
        # 날짜 정렬
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # 결측치 처리
        numeric_cols = [col for col in df_clean.columns if col != 'date']
        
        # 선형 보간으로 결측치 채우기
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        
        # 남은 결측치는 이전 값으로 채우기
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
        
        # 음수 값 처리 (사용자 수, 매출, 비용은 음수가 될 수 없음)
        for col in numeric_cols:
            df_clean[col] = df_clean[col].clip(lower=0)
        
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특성 엔지니어링
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            특성이 추가된 데이터프레임
        """
        df_featured = df.copy()
        
        # 월, 분기, 연도 추출
        df_featured['month'] = df_featured['date'].dt.month
        df_featured['quarter'] = df_featured['date'].dt.quarter
        df_featured['year'] = df_featured['date'].dt.year
        
        # 기술별 총 사용자 수
        df_featured['total_users'] = (
            df_featured['5g_users'] + 
            df_featured['lte_users'] + 
            df_featured['3g_users']
        )
        
        # 기술별 총 매출
        df_featured['total_revenue'] = (
            df_featured['5g_revenue'] + 
            df_featured['lte_revenue'] + 
            df_featured['3g_revenue']
        )
        
        # 기술별 총 비용
        df_featured['total_cost'] = (
            df_featured['5g_cost'] + 
            df_featured['lte_cost'] + 
            df_featured['3g_cost']
        )
        
        # 수익성 지표
        df_featured['profit'] = df_featured['total_revenue'] - df_featured['total_cost']
        df_featured['profit_margin'] = (
            df_featured['profit'] / df_featured['total_revenue']
        ).fillna(0)
        
        # 기술별 점유율
        df_featured['5g_share'] = df_featured['5g_users'] / df_featured['total_users']
        df_featured['lte_share'] = df_featured['lte_users'] / df_featured['total_users']
        df_featured['3g_share'] = df_featured['3g_users'] / df_featured['total_users']
        
        # ARPU (Average Revenue Per User)
        df_featured['5g_arpu'] = df_featured['5g_revenue'] / df_featured['5g_users']
        df_featured['lte_arpu'] = df_featured['lte_revenue'] / df_featured['lte_users']
        df_featured['3g_arpu'] = df_featured['3g_revenue'] / df_featured['3g_users']
        
        # 결측치 처리
        arpu_cols = ['5g_arpu', 'lte_arpu', '3g_arpu']
        df_featured[arpu_cols] = df_featured[arpu_cols].fillna(0)
        
        return df_featured
    
    def prepare_timesfm_data(self, df: pd.DataFrame, 
                           target_columns: List[str],
                           forecast_horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        TimesFM 모델용 데이터 준비
        
        Args:
            df: 전처리된 데이터프레임
            target_columns: 예측할 타겟 컬럼들
            forecast_horizon: 예측 기간 (개월)
            
        Returns:
            (입력 데이터, 타겟 데이터) 튜플
        """
        # 타겟 컬럼만 선택
        target_data = df[target_columns].values
        
        # 정규화
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(target_data)
        
        # 시계열 윈도우 생성
        input_data = []
        target_data_processed = []
        
        for i in range(len(scaled_data) - forecast_horizon):
            input_data.append(scaled_data[i:i+forecast_horizon])
            target_data_processed.append(scaled_data[i+forecast_horizon])
        
        return np.array(input_data), np.array(target_data_processed)
    
    def process_pipeline(self, file_path: str, 
                        target_columns: List[str],
                        forecast_horizon: int = 12) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        전체 데이터 처리 파이프라인
        
        Args:
            file_path: 데이터 파일 경로
            target_columns: 예측할 타겟 컬럼들
            forecast_horizon: 예측 기간
            
        Returns:
            (전처리된 데이터프레임, 입력 데이터, 타겟 데이터) 튜플
        """
        # 1. 데이터 로드
        print("데이터 로드 중...")
        df = self.load_data(file_path)
        
        # 2. 데이터 검증
        print("데이터 검증 중...")
        if not self.validate_data(df):
            raise ValueError("데이터 검증에 실패했습니다.")
        
        # 3. 데이터 정리
        print("데이터 정리 중...")
        df_clean = self.clean_data(df)
        
        # 4. 특성 생성
        print("특성 생성 중...")
        df_featured = self.create_features(df_clean)
        
        # 5. TimesFM 데이터 준비
        print("TimesFM 데이터 준비 중...")
        input_data, target_data = self.prepare_timesfm_data(
            df_featured, target_columns, forecast_horizon
        )
        
        print(f"처리 완료! 입력 데이터 형태: {input_data.shape}, 타겟 데이터 형태: {target_data.shape}")
        
        return df_featured, input_data, target_data

# 샘플 데이터 생성 함수
def generate_sample_data(start_date: str = '2020-01-01', 
                        periods: int = 48) -> pd.DataFrame:
    """
    샘플 통신사 데이터 생성
    
    Args:
        start_date: 시작 날짜
        periods: 데이터 기간 (개월)
        
    Returns:
        샘플 데이터프레임
    """
    np.random.seed(42)
    
    # 날짜 생성
    dates = pd.date_range(start=start_date, periods=periods, freq='M')
    
    # 5G 데이터 (성장 추세)
    base_5g = 1000
    growth_5g = 0.15
    noise_5g = 0.1
    
    # LTE 데이터 (안정적)
    base_lte = 5000
    growth_lte = 0.02
    noise_lte = 0.05
    
    # 3G 데이터 (감소 추세)
    base_3g = 2000
    growth_3g = -0.08
    noise_3g = 0.15
    
    data = []
    for i, date in enumerate(dates):
        # 사용자 수
        users_5g = int(base_5g * (1 + growth_5g) ** i * (1 + np.random.normal(0, noise_5g)))
        users_lte = int(base_lte * (1 + growth_lte) ** i * (1 + np.random.normal(0, noise_lte)))
        users_3g = int(base_3g * (1 + growth_3g) ** i * (1 + np.random.normal(0, noise_3g)))
        
        # 매출 (사용자 수에 비례)
        revenue_5g = users_5g * 50 * (1 + np.random.normal(0, 0.05))
        revenue_lte = users_lte * 30 * (1 + np.random.normal(0, 0.03))
        revenue_3g = users_3g * 20 * (1 + np.random.normal(0, 0.08))
        
        # 비용 (매출의 60-80%)
        cost_5g = revenue_5g * (0.6 + np.random.normal(0, 0.1))
        cost_lte = revenue_lte * (0.7 + np.random.normal(0, 0.05))
        cost_3g = revenue_3g * (0.8 + np.random.normal(0, 0.1))
        
        data.append({
            'date': date,
            '5g_users': max(0, users_5g),
            'lte_users': max(0, users_lte),
            '3g_users': max(0, users_3g),
            '5g_revenue': max(0, revenue_5g),
            'lte_revenue': max(0, revenue_lte),
            '3g_revenue': max(0, revenue_3g),
            '5g_cost': max(0, cost_5g),
            'lte_cost': max(0, cost_lte),
            '3g_cost': max(0, cost_3g)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 샘플 데이터 생성 및 테스트
    sample_data = generate_sample_data()
    sample_data.to_csv('data/raw/sample_telecom_data.csv', index=False)
    print("샘플 데이터가 생성되었습니다: data/raw/sample_telecom_data.csv") 