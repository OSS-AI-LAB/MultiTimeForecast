"""
통신사 재무 데이터 처리 모듈
GL_ACC_LSN_NM 기준으로 계정과목별 시계열 데이터 변환 및 전처리
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 추가 라이브러리 import
try:
    import chardet
except ImportError:
    chardet = None

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelecomDataProcessor:
    """통신사 재무 데이터 처리 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """초기화"""
        self.config = self._load_config(config_path)
        self.scaler = None
        self.imputer = None
        self.account_columns = []
        self.product_columns = []
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """원본 데이터 로드 - 다양한 파일 형식 및 인코딩 지원"""
        if file_path is None:
            file_path = self.config['data']['raw_file']
        
        logger.info(f"원본 데이터 로드: {file_path}")
        
        # 파일 확장자 확인
        file_ext = Path(file_path).suffix.lower()
        
        try:
            # Excel 파일 처리 (DRM 보호된 파일 포함)
            if file_ext in ['.xlsx', '.xls']:
                logger.info("Excel 파일 감지 - Excel 엔진으로 로드 시도")
                try:
                    # openpyxl 엔진으로 시도
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e1:
                    logger.warning(f"openpyxl 엔진 실패: {e1}")
                    try:
                        # xlrd 엔진으로 시도
                        df = pd.read_excel(file_path, engine='xlrd')
                    except Exception as e2:
                        logger.warning(f"xlrd 엔진 실패: {e2}")
                        # 기본 엔진으로 시도
                        df = pd.read_excel(file_path)
            
            # CSV 파일 처리 (다양한 인코딩 시도)
            elif file_ext == '.csv':
                df = None
                
                # 1. chardet를 사용한 자동 인코딩 감지
                if chardet is not None:
                    try:
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            detected_encoding = result['encoding']
                            confidence = result['confidence']
                            
                        if detected_encoding and confidence > 0.7:
                            logger.info(f"자동 감지된 인코딩: {detected_encoding} (신뢰도: {confidence:.2f})")
                            try:
                                df = pd.read_csv(file_path, encoding=detected_encoding)
                                logger.info("자동 감지 인코딩으로 성공적으로 로드됨")
                            except Exception as e:
                                logger.warning(f"자동 감지 인코딩 실패: {e}")
                    except Exception as e:
                        logger.warning(f"자동 인코딩 감지 실패: {e}")
                
                # 2. 수동 인코딩 시도
                if df is None:
                    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1', 'utf-8-sig']
                    
                    for encoding in encodings:
                        try:
                            logger.info(f"수동 인코딩 {encoding}로 시도")
                            df = pd.read_csv(file_path, encoding=encoding)
                            logger.info(f"성공: {encoding} 인코딩으로 로드됨")
                            break
                        except UnicodeDecodeError:
                            logger.warning(f"인코딩 {encoding} 실패")
                            continue
                        except Exception as e:
                            logger.warning(f"인코딩 {encoding}에서 기타 오류: {e}")
                            continue
                
                if df is None:
                    raise ValueError("모든 인코딩 시도 실패")
            
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
            
            # 컬럼명 정리
            columns_config = self.config['data']['columns']
            df.columns = [col.strip() for col in df.columns]
            
            # 데이터 타입 변환
            df[columns_config['date_col']] = pd.to_datetime(df[columns_config['date_col']], format='%Y%m')
            df[columns_config['value_col']] = pd.to_numeric(df[columns_config['value_col']], errors='coerce')
            
            logger.info(f"데이터 로드 완료: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise
    
    def filter_accounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """중요 계정과목 필터링"""
        columns_config = self.config['data']['columns']
        filter_config = self.config['data']['account_filtering']
        
        # 계정과목별 총 매출액 계산
        account_totals = df.groupby(columns_config['account_name_col'])[columns_config['value_col']].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        
        # 필터링 조건 적용
        filtered_accounts = account_totals[
            (account_totals['sum'].abs() >= filter_config['min_total_value']) &
            (account_totals['count'] >= filter_config['min_occurrence'])
        ]
        
        # 제외 패턴 필터링
        for pattern in filter_config['exclude_patterns']:
            filtered_accounts = filtered_accounts[
                ~filtered_accounts[columns_config['account_name_col']].str.contains(pattern, na=False)
            ]
        
        # 필터링된 계정과목만 선택
        important_accounts = filtered_accounts[columns_config['account_name_col']].tolist()
        df_filtered = df[df[columns_config['account_name_col']].isin(important_accounts)]
        
        logger.info(f"계정과목 필터링: {len(important_accounts)}개 계정과목 선택")
        return df_filtered
    
    def create_pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """계정과목별 피벗 데이터 생성"""
        columns_config = self.config['data']['columns']
        
        # 계정과목별 피벗 테이블 생성
        pivot_df = df.pivot_table(
            index=columns_config['date_col'],
            columns=columns_config['account_name_col'],
            values=columns_config['value_col'],
            aggfunc='sum',
            fill_value=0
        )
        
        # 제품별 피벗 테이블 생성
        product_pivot = df.pivot_table(
            index=columns_config['date_col'],
            columns=columns_config['product_col'],
            values=columns_config['value_col'],
            aggfunc='sum',
            fill_value=0
        )
        
        # 계정과목별 컬럼명 저장
        self.account_columns = pivot_df.columns.tolist()
        self.product_columns = product_pivot.columns.tolist()
        
        # 두 피벗 테이블 결합
        combined_df = pd.concat([pivot_df, product_pivot], axis=1)
        combined_df = combined_df.sort_index()
        
        logger.info(f"피벗 데이터 생성: {combined_df.shape}")
        logger.info(f"계정과목 컬럼: {len(self.account_columns)}개")
        logger.info(f"제품 컬럼: {len(self.product_columns)}개")
        
        return combined_df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간적 특성 추가"""
        df = df.copy()
        
        # 기본 시간 특성
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # 계절성 특성 (사인/코사인 변환)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # 연도별 특성
        df['year_since_start'] = df['year'] - df['year'].min()
        
        logger.info("시간적 특성 추가 완료")
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """지연 특성 추가"""
        df = df.copy()
        lag_periods = self.config['preprocessing']['features']['lag_features']
        
        for col in self.account_columns + self.product_columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"지연 특성 추가 완료: {len(lag_periods)}개 기간")
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """이동평균 특성 추가"""
        df = df.copy()
        rolling_periods = self.config['preprocessing']['features']['rolling_features']
        
        for col in self.account_columns + self.product_columns:
            for period in rolling_periods:
                df[f'{col}_rolling_mean_{period}'] = df[col].rolling(window=period, min_periods=1).mean()
                df[f'{col}_rolling_std_{period}'] = df[col].rolling(window=period, min_periods=1).std()
        
        logger.info(f"이동평균 특성 추가 완료: {len(rolling_periods)}개 기간")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        df = df.copy()
        
        # 시간적 특성은 0으로 채우기
        temporal_cols = ['year', 'month', 'quarter', 'sin_month', 'cos_month', 'sin_quarter', 'cos_quarter', 'year_since_start']
        df[temporal_cols] = df[temporal_cols].fillna(0)
        
        # 계정과목 데이터는 전진 채우기 후 0으로 채우기
        account_cols = [col for col in df.columns if col not in temporal_cols]
        df[account_cols] = df[account_cols].fillna(method='ffill').fillna(0)
        
        logger.info("결측치 처리 완료")
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """특성 스케일링"""
        df = df.copy()
        scaling_config = self.config['preprocessing']['scaling']
        
        # 스케일링 대상 컬럼 선택
        exclude_cols = scaling_config['exclude_cols']
        scale_cols = [col for col in df.columns if col not in exclude_cols]
        
        if scaling_config['method'] == 'robust':
            self.scaler = RobustScaler()
        elif scaling_config['method'] == 'standard':
            self.scaler = StandardScaler()
        else:
            return df  # 스케일링 없음
        
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            if self.scaler is not None:
                df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        logger.info(f"특성 스케일링 완료: {scaling_config['method']}")
        return df
    
    def create_hierarchical_structure(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """계층적 구조 생성"""
        hierarchical_data = {}
        
        # 전체 레벨
        hierarchical_data['total'] = df[self.account_columns].sum(axis=1).to_frame('total_revenue')
        
        # 제품별 레벨
        for product in self.product_columns:
            if product in df.columns:
                hierarchical_data[f'product_{product}'] = df[product].to_frame(f'{product}_revenue')
        
        # 계정과목별 레벨
        for account in self.account_columns:
            hierarchical_data[f'account_{account}'] = df[account].to_frame(f'{account}_revenue')
        
        logger.info(f"계층적 구조 생성: {len(hierarchical_data)}개 레벨")
        return hierarchical_data
    
    def process_data(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """전체 데이터 처리 파이프라인"""
        logger.info("=== 데이터 처리 파이프라인 시작 ===")
        
        # 1. 원본 데이터 로드
        df = self.load_raw_data(file_path)
        
        # 2. 계정과목 필터링
        df_filtered = self.filter_accounts(df)
        
        # 3. 피벗 데이터 생성
        pivot_df = self.create_pivot_data(df_filtered)
        
        # 4. 시간적 특성 추가
        pivot_df = self.add_temporal_features(pivot_df)
        
        # 5. 지연 특성 추가
        pivot_df = self.add_lag_features(pivot_df)
        
        # 6. 이동평균 특성 추가
        pivot_df = self.add_rolling_features(pivot_df)
        
        # 7. 결측치 처리
        pivot_df = self.handle_missing_values(pivot_df)
        
        # 8. 특성 스케일링
        pivot_df = self.scale_features(pivot_df, fit=True)
        
        # 9. 계층적 구조 생성
        hierarchical_data = self.create_hierarchical_structure(pivot_df)
        
        # 10. 처리된 데이터 저장
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(exist_ok=True)
        
        pivot_df.to_csv(processed_dir / 'processed_data.csv')
        
        logger.info("=== 데이터 처리 파이프라인 완료 ===")
        return pivot_df, hierarchical_data
    
    def get_feature_info(self) -> Dict:
        """특성 정보 반환"""
        return {
            'account_columns': self.account_columns,
            'product_columns': self.product_columns,
            'total_features': len(self.account_columns) + len(self.product_columns),
            'scaler': type(self.scaler).__name__ if self.scaler else None
        } 