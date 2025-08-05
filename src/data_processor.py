"""
통신사 재무 데이터 처리 모듈
GL_ACC_LSN_NM 기준으로 계정과목별 시계열 데이터 변환 및 전처리
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
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

try:
    import xlwings as xw
except ImportError:
    xw = None

try:
    import win32com.client
except ImportError:
    win32com = None

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
                logger.info("Excel 파일 감지 - 다양한 방법으로 로드 시도")
                df = None
                
                # 1. pandas 엔진들로 시도
                engines = ['openpyxl', 'xlrd']
                for engine in engines:
                    try:
                        logger.info(f"pandas {engine} 엔진으로 시도")
                        df = pd.read_excel(file_path, engine=engine)
                        logger.info(f"성공: {engine} 엔진으로 로드됨")
                        break
                    except Exception as e:
                        logger.warning(f"{engine} 엔진 실패: {e}")
                        continue
                
                # 2. xlwings로 시도 (DRM 보호 파일 처리)
                if df is None and xw is not None:
                    try:
                        logger.info("xlwings로 시도 (DRM 보호 파일 처리)")
                        app = xw.App(visible=False)
                        wb = app.books.open(file_path)
                        sheet = wb.sheets[0]
                        data = sheet.used_range.options(pd.DataFrame, index=False, header=True).value
                        wb.close()
                        app.quit()
                        df = data
                        logger.info("성공: xlwings로 로드됨")
                    except Exception as e:
                        logger.warning(f"xlwings 실패: {e}")
                
                # 3. win32com으로 시도 (Windows 전용)
                if df is None and win32com is not None:
                    try:
                        logger.info("win32com으로 시도")
                        excel = win32com.client.Dispatch("Excel.Application")
                        excel.Visible = False
                        wb = excel.Workbooks.Open(os.path.abspath(file_path))
                        sheet = wb.Sheets(1)
                        data = sheet.UsedRange.Value
                        wb.Close()
                        excel.Quit()
                        
                        # 데이터를 DataFrame으로 변환
                        if data:
                            df = pd.DataFrame(data[1:], columns=data[0])
                        logger.info("성공: win32com으로 로드됨")
                    except Exception as e:
                        logger.warning(f"win32com 실패: {e}")
                
                # 4. 기본 엔진으로 최종 시도
                if df is None:
                    try:
                        logger.info("기본 엔진으로 최종 시도")
                        df = pd.read_excel(file_path)
                        logger.info("성공: 기본 엔진으로 로드됨")
                    except Exception as e:
                        logger.error(f"모든 Excel 로드 방법 실패: {e}")
                        raise
            
            # CSV 파일 처리 (다양한 인코딩 시도)
            elif file_ext == '.csv':
                df = None
                
                # 1. chardet를 사용한 자동 인코딩 감지 + 구분자 감지
                if chardet is not None:
                    try:
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            detected_encoding = result['encoding']
                            confidence = result['confidence']
                            
                        if detected_encoding and confidence > 0.7:
                            logger.info(f"자동 감지된 인코딩: {detected_encoding} (신뢰도: {confidence:.2f})")
                            
                            # 구분자 자동 감지
                            try:
                                with open(file_path, 'r', encoding=detected_encoding) as f:
                                    first_line = f.readline().strip()
                                
                                # 일반적인 구분자들로 테스트
                                delimiters = [',', ';', '\t', '|']
                                best_delimiter = ','
                                max_fields = 1
                                
                                for delimiter in delimiters:
                                    field_count = len(first_line.split(delimiter))
                                    if field_count > max_fields:
                                        max_fields = field_count
                                        best_delimiter = delimiter
                                
                                logger.info(f"자동 감지된 구분자: '{best_delimiter}' (필드 수: {max_fields})")
                                
                                try:
                                    # 다양한 CSV 파싱 옵션 시도 (따옴표 처리 중심)
                                    csv_options = [
                                        {'delimiter': best_delimiter, 'quotechar': '"', 'quoting': 1},  # QUOTE_ALL
                                        {'delimiter': best_delimiter, 'quotechar': '"', 'quoting': 0},  # QUOTE_MINIMAL
                                        {'delimiter': best_delimiter, 'quotechar': '"', 'quoting': 2},  # QUOTE_NONNUMERIC
                                        {'delimiter': best_delimiter, 'quotechar': '"'},
                                        {'delimiter': best_delimiter, 'quoting': 3},  # QUOTE_NONE
                                        {'delimiter': best_delimiter, 'escapechar': '\\'},
                                        {'delimiter': best_delimiter, 'quotechar': "'", 'quoting': 1},  # 작은따옴표
                                        {'delimiter': best_delimiter, 'quotechar': "'", 'quoting': 0}   # 작은따옴표
                                    ]
                                    
                                    for options in csv_options:
                                        try:
                                            df = pd.read_csv(file_path, encoding=detected_encoding, skipinitialspace=True, **options)
                                            logger.info(f"자동 감지 인코딩과 구분자로 성공적으로 로드됨 (옵션: {options})")
                                            break
                                        except Exception as e:
                                            logger.warning(f"CSV 옵션 {options} 실패: {e}")
                                            continue
                                    
                                    if df is None:
                                        logger.warning("모든 CSV 옵션 실패")
                                except Exception as e:
                                    logger.warning(f"자동 감지 설정 실패: {e}")
                            except Exception as e:
                                logger.warning(f"구분자 감지 실패: {e}")
                    except Exception as e:
                        logger.warning(f"자동 인코딩 감지 실패: {e}")
                
                # 2. 수동 인코딩 시도 (다양한 구분자 포함)
                if df is None:
                    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1', 'utf-8-sig']
                    delimiters = [',', ';', '\t', '|']
                    
                    for encoding in encodings:
                        for delimiter in delimiters:
                            # 다양한 CSV 파싱 옵션 시도 (따옴표 처리 중심)
                            csv_options = [
                                {'delimiter': delimiter, 'quotechar': '"', 'quoting': 1},  # QUOTE_ALL
                                {'delimiter': delimiter, 'quotechar': '"', 'quoting': 0},  # QUOTE_MINIMAL
                                {'delimiter': delimiter, 'quotechar': '"', 'quoting': 2},  # QUOTE_NONNUMERIC
                                {'delimiter': delimiter, 'quotechar': '"'},
                                {'delimiter': delimiter, 'quoting': 3},  # QUOTE_NONE
                                {'delimiter': delimiter, 'escapechar': '\\'},
                                {'delimiter': delimiter, 'quotechar': "'", 'quoting': 1},  # 작은따옴표
                                {'delimiter': delimiter, 'quotechar': "'", 'quoting': 0}   # 작은따옴표
                            ]
                            
                            for options in csv_options:
                                try:
                                    logger.info(f"인코딩 {encoding}, 구분자 '{delimiter}', 옵션 {options}로 시도")
                                    df = pd.read_csv(file_path, encoding=encoding, skipinitialspace=True, **options)
                                    logger.info(f"성공: {encoding} 인코딩, '{delimiter}' 구분자로 로드됨")
                                    break
                                except UnicodeDecodeError:
                                    logger.warning(f"인코딩 {encoding} 실패")
                                    break
                                except Exception as e:
                                    logger.warning(f"인코딩 {encoding}, 구분자 '{delimiter}', 옵션 {options}에서 기타 오류: {e}")
                                    continue
                            
                            if df is not None:
                                break
                        
                        if df is not None:
                            break
                
                if df is None:
                    # 최후의 수단: 정확한 CSV 파싱 시도
                    logger.info("최후의 수단: 정확한 CSV 파싱 시도")
                    try:
                        import csv
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            rows = list(reader)
                        
                        if len(rows) > 1:
                            header = rows[0]
                            data_rows = rows[1:]
                            df = pd.DataFrame(data_rows, columns=header)
                            logger.info(f"CSV 모듈로 성공: {len(data_rows)}개 행 로드됨")
                        else:
                            raise ValueError("데이터가 부족함")
                    except Exception as e:
                        logger.error(f"CSV 모듈 파싱도 실패: {e}")
                        raise ValueError("모든 인코딩 시도 실패")
            
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
            
            # 컬럼명 정리 (공백 제거)
            columns_config = self.config['data']['columns']
            df.columns = [col.strip() for col in df.columns]
            
            # 모든 문자열 컬럼의 공백 제거
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
            
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
        
        # 계정과목 컬럼만 스케일링 (예측 대상 컬럼)
        scale_cols = self.account_columns if hasattr(self, 'account_columns') else []
        
        if not scale_cols:
            logger.warning("계정과목 컬럼이 없어 스케일링을 건너뜁니다")
            return df
        
        if scaling_config['method'] == 'robust':
            self.scaler = RobustScaler()
        elif scaling_config['method'] == 'standard':
            self.scaler = StandardScaler()
        elif scaling_config['method'] == 'none':
            logger.info("스케일링 비활성화됨")
            return df  # 스케일링 없음
        else:
            logger.warning(f"알 수 없는 스케일링 방법: {scaling_config['method']}")
            return df  # 스케일링 없음
        
        if fit:
            # 원본 계정과목 데이터 저장 (역변환용)
            if hasattr(self, 'account_columns'):
                self._original_account_data = df[self.account_columns].copy()
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            if self.scaler is not None:
                df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        logger.info(f"특성 스케일링 완료: {scaling_config['method']} (대상: {len(scale_cols)}개 컬럼)")
        return df
    
    def inverse_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """정규화된 특성을 원본 값으로 역변환"""
        df = df.copy()
        scaling_config = self.config['preprocessing']['scaling']
        
        # 계정과목 컬럼만 역변환 (예측 대상 컬럼)
        scale_cols = self.account_columns if hasattr(self, 'account_columns') else []
        
        # 스케일링이 none인 경우 원본 값 그대로 반환
        if scaling_config['method'] == 'none':
            logger.info("스케일링이 비활성화되어 있어 원본 값 그대로 반환")
            return df
        
        if self.scaler is not None and scale_cols:
            try:
                # DataFrame에서 계정과목 컬럼만 선택하여 역변환
                available_cols = [col for col in scale_cols if col in df.columns]
                if available_cols:
                    # 스케일러가 훈련된 전체 컬럼과 현재 데이터의 컬럼 수가 다른 경우 처리
                    if len(available_cols) != len(scale_cols):
                        # 계정과목 컬럼만으로 별도 스케일러 생성
                        account_scaler = RobustScaler() if scaling_config['method'] == 'robust' else StandardScaler()
                        
                        # 원본 데이터에서 계정과목 컬럼만 추출하여 스케일러 재훈련
                        if hasattr(self, '_original_account_data'):
                            account_data = self._original_account_data[available_cols]
                            account_scaler.fit(account_data)
                            df[available_cols] = account_scaler.inverse_transform(df[available_cols])
                            logger.info(f"계정과목 전용 스케일러로 역변환 완료: {len(available_cols)}개 컬럼")
                        else:
                            logger.warning("원본 계정과목 데이터가 없어 역변환을 건너뜁니다")
                            return df
                    else:
                        # 컬럼 수가 일치하는 경우 기존 방식 사용
                        df[available_cols] = self.scaler.inverse_transform(df[available_cols])
                        logger.info(f"특성 역변환 완료: {len(available_cols)}개 컬럼")
                else:
                    logger.warning("역변환할 계정과목 컬럼이 없습니다")
            except Exception as e:
                logger.warning(f"특성 역변환 실패: {e}")
        
        return df
    

    
    def process_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
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
        

        
        # 10. 처리된 데이터 저장 (선택적)
        try:
            processed_dir = Path(self.config['data']['processed_dir'])
            # 여러 경로 시도
            possible_paths = [
                processed_dir,
                Path('../data/processed'),
                Path('../../data/processed'),
                Path(os.path.dirname(__file__), '..', 'data', 'processed')
            ]
            
            saved = False
            for path in possible_paths:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    pivot_df.to_csv(path / 'processed_data.csv')
                    logger.info(f"처리된 데이터 저장 완료: {path}")
                    saved = True
                    break
                except Exception as e:
                    logger.warning(f"저장 경로 실패 {path}: {e}")
                    continue
            
            if not saved:
                logger.warning("처리된 데이터 저장을 건너뜁니다")
        except Exception as e:
            logger.warning(f"데이터 저장 실패: {e}")
        
        logger.info("=== 데이터 처리 파이프라인 완료 ===")
        return pivot_df
    
    def get_feature_info(self) -> Dict:
        """특성 정보 반환"""
        return {
            'account_columns': self.account_columns,
            'product_columns': self.product_columns,
            'total_features': len(self.account_columns) + len(self.product_columns),
            'scaler': type(self.scaler).__name__ if self.scaler else None
        } 