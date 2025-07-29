"""
통신사 재무 예측 시각화 모듈
예측 결과 및 분석 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelecomVisualizer:
    """통신사 재무 예측 시각화 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """초기화"""
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['data']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_forecast_plot(self, actual_data: pd.DataFrame, 
                           forecast_data: pd.DataFrame,
                           target_columns: List[str]) -> go.Figure:
        """예측 결과 시각화"""
        fig = make_subplots(
            rows=len(target_columns), cols=1,
            subplot_titles=target_columns,
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(target_columns):
            if col in actual_data.columns:
                # 실제 데이터
                fig.add_trace(
                    go.Scatter(
                        x=actual_data.index,
                        y=actual_data[col],
                        mode='lines+markers',
                        name=f'{col} (실제)',
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
            
            if col in forecast_data.columns:
                # 예측 데이터
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data[col],
                        mode='lines+markers',
                        name=f'{col} (예측)',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="통신사 재무 예측 결과",
            height=300 * len(target_columns),
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_accuracy_plot(self, evaluation_results: Dict) -> go.Figure:
        """모델 정확도 비교 시각화"""
        # 평가 결과를 데이터프레임으로 변환
        accuracy_data = []
        
        for model_name, model_results in evaluation_results.items():
            for metric_name, metric_results in model_results.items():
                if isinstance(metric_results, dict):
                    for col, value in metric_results.items():
                        accuracy_data.append({
                            'Model': model_name,
                            'Metric': metric_name,
                            'Account': col,
                            'Value': value
                        })
        
        if not accuracy_data:
            return go.Figure()
        
        df_accuracy = pd.DataFrame(accuracy_data)
        
        # 메트릭별로 서브플롯 생성
        metrics = df_accuracy['Metric'].unique()
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            metric_data = df_accuracy[df_accuracy['Metric'] == metric]
            
            # 모델별 박스플롯
            for j, model in enumerate(metric_data['Model'].unique()):
                model_data = metric_data[metric_data['Model'] == model]
                
                fig.add_trace(
                    go.Box(
                        y=model_data['Value'],
                        name=model,
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="모델 성능 비교",
            height=300 * len(metrics),
            template="plotly_white"
        )
        
        return fig
    
    def create_feature_importance_plot(self, processed_data: pd.DataFrame,
                                     target_columns: List[str]) -> go.Figure:
        """특성 중요도 시각화 (상관관계 기반)"""
        # 계정과목 컬럼만 선택
        account_cols = [col for col in processed_data.columns 
                       if col not in ['year', 'month', 'quarter', 'sin_month', 'cos_month', 
                                    'sin_quarter', 'cos_quarter', 'year_since_start']]
        
        # 상관관계 계산
        correlation_matrix = processed_data[account_cols].corr()
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="계정과목 간 상관관계",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_seasonal_decomposition_plot(self, time_series_dict: Dict,
                                         target_columns: List[str]) -> go.Figure:
        """계절성 분해 시각화"""
        fig = make_subplots(
            rows=len(target_columns), cols=1,
            subplot_titles=[f"{col} - 계절성 분해" for col in target_columns],
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(target_columns):
            if col in time_series_dict:
                series = time_series_dict[col]
                values = series.values()
                dates = series.time_index
                
                # 간단한 계절성 분석 (12개월 이동평균)
                if len(values) >= 12:
                    trend = pd.Series(values).rolling(window=12, center=True).mean()
                    seasonal = pd.Series(values) - trend
                    residual = pd.Series(values) - trend - seasonal
                    
                    # 원본 데이터
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=values,
                            mode='lines',
                            name=f'{col} (원본)',
                            line=dict(color='blue', width=1),
                            showlegend=(i == 0)
                        ),
                        row=i+1, col=1
                    )
                    
                    # 트렌드
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=trend,
                            mode='lines',
                            name=f'{col} (트렌드)',
                            line=dict(color='red', width=2),
                            showlegend=(i == 0)
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            title="계절성 분해 분석",
            height=300 * len(target_columns),
            template="plotly_white"
        )
        
        return fig
    
    def create_hierarchical_forecast_plot(self, hierarchical_data: Dict,
                                        forecast_data: pd.DataFrame) -> go.Figure:
        """계층적 예측 결과 시각화"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['전체 매출', '제품별 매출', '계정과목별 매출', '예측 vs 실제'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 전체 매출
        if 'total' in hierarchical_data:
            total_data = hierarchical_data['total']
            fig.add_trace(
                go.Scatter(
                    x=total_data.index,
                    y=total_data['total_revenue'],
                    mode='lines+markers',
                    name='전체 매출',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 제품별 매출
        product_cols = [col for col in hierarchical_data.keys() if col.startswith('product_')]
        for i, product in enumerate(product_cols[:3]):  # 상위 3개 제품만
            product_data = hierarchical_data[product]
            fig.add_trace(
                go.Scatter(
                    x=product_data.index,
                    y=product_data.iloc[:, 0],
                    mode='lines',
                    name=product.replace('product_', ''),
                    line=dict(width=1)
                ),
                row=1, col=2
            )
        
        # 계정과목별 매출 (상위 5개)
        account_cols = [col for col in hierarchical_data.keys() if col.startswith('account_')]
        for i, account in enumerate(account_cols[:5]):
            account_data = hierarchical_data[account]
            fig.add_trace(
                go.Scatter(
                    x=account_data.index,
                    y=account_data.iloc[:, 0],
                    mode='lines',
                    name=account.replace('account_', ''),
                    line=dict(width=1)
                ),
                row=2, col=1
            )
        
        # 예측 vs 실제 (첫 번째 계정과목)
        if account_cols and account_cols[0] in hierarchical_data:
            account_name = account_cols[0].replace('account_', '')
            actual_data = hierarchical_data[account_cols[0]]
            
            if account_name in forecast_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=actual_data.index,
                        y=actual_data.iloc[:, 0],
                        mode='lines+markers',
                        name=f'{account_name} (실제)',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data[account_name],
                        mode='lines+markers',
                        name=f'{account_name} (예측)',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="계층적 예측 분석",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def create_dashboard(self, results: Dict) -> str:
        """대시보드 생성"""
        dashboard_config = self.config['visualization']['dashboard']
        
        # 대시보드 HTML 생성
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>통신사 재무 예측 대시보드</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .chart { margin-bottom: 30px; }
                .summary { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>통신사 재무 예측 대시보드</h1>
                    <p>생성일: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
                
                <div class="summary">
                    <h2>예측 요약</h2>
                    <p>예측 기간: """ + str(len(results.get('ensemble_forecast', pd.DataFrame()))) + """개월</p>
                    <p>예측 대상: """ + str(len(results.get('ensemble_forecast', pd.DataFrame()).columns)) + """개 계정과목</p>
                </div>
                
                <div class="chart" id="forecast-chart"></div>
                <div class="chart" id="accuracy-chart"></div>
                <div class="chart" id="correlation-chart"></div>
                <div class="chart" id="seasonal-chart"></div>
                <div class="chart" id="hierarchical-chart"></div>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = self.results_dir / "dashboard.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"대시보드 생성 완료: {dashboard_path}")
        return str(dashboard_path)
    
    def generate_report(self, processed_data: pd.DataFrame,
                       results: Dict,
                       target_columns: List[str]) -> str:
        """종합 분석 리포트 생성"""
        logger.info("=== 시각화 리포트 생성 시작 ===")
        
        # 1. 예측 결과 시각화
        forecast_fig = self.create_forecast_plot(
            processed_data, 
            results.get('ensemble_forecast', pd.DataFrame()),
            target_columns
        )
        forecast_fig.write_html(self.results_dir / "forecast_plot.html")
        
        # 2. 모델 정확도 시각화
        accuracy_fig = self.create_accuracy_plot(
            results.get('evaluation_results', {})
        )
        accuracy_fig.write_html(self.results_dir / "accuracy_plot.html")
        
        # 3. 특성 중요도 시각화
        importance_fig = self.create_feature_importance_plot(
            processed_data, target_columns
        )
        importance_fig.write_html(self.results_dir / "correlation_plot.html")
        
        # 4. 계절성 분해 시각화
        seasonal_fig = self.create_seasonal_decomposition_plot(
            results.get('time_series_dict', {}),
            target_columns
        )
        seasonal_fig.write_html(self.results_dir / "seasonal_plot.html")
        
        # 5. 계층적 예측 시각화
        hierarchical_fig = self.create_hierarchical_forecast_plot(
            results.get('hierarchical_data', {}),
            results.get('ensemble_forecast', pd.DataFrame())
        )
        hierarchical_fig.write_html(self.results_dir / "hierarchical_plot.html")
        
        # 6. 대시보드 생성
        dashboard_path = self.create_dashboard(results)
        
        # 7. 예측 결과 CSV 저장
        if 'ensemble_forecast' in results:
            results['ensemble_forecast'].to_csv(self.results_dir / "forecast_results.csv")
        
        # 8. 평가 결과 CSV 저장
        if 'evaluation_results' in results:
            evaluation_df = pd.DataFrame()
            for model_name, model_results in results['evaluation_results'].items():
                for metric_name, metric_results in model_results.items():
                    if isinstance(metric_results, dict):
                        for col, value in metric_results.items():
                            evaluation_df = evaluation_df.append({
                                'Model': model_name,
                                'Metric': metric_name,
                                'Account': col,
                                'Value': value
                            }, ignore_index=True)
            
            evaluation_df.to_csv(self.results_dir / "evaluation_results.csv", index=False)
        
        logger.info("=== 시각화 리포트 생성 완료 ===")
        return dashboard_path 