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
                           target_columns: List[str],
                           data_processor=None) -> go.Figure:
        """예측 결과 시각화 - 현대적 디자인"""
        # vertical_spacing을 동적으로 계산하여 오류 방지
        n_rows = len(target_columns)
        if n_rows <= 1:
            vertical_spacing = 0.1
        else:
            vertical_spacing = min(0.08, 1.0 / (n_rows + 1))
        
        # 정규화된 데이터를 원본 값으로 변환
        actual_display = actual_data.copy()
        forecast_display = forecast_data.copy()
        
        if data_processor and hasattr(data_processor, 'inverse_scale_features'):
            try:
                # 실제 데이터 역변환
                actual_display = data_processor.inverse_scale_features(actual_display)
                # 예측 데이터 역변환
                forecast_display = data_processor.inverse_scale_features(forecast_display)
                logger.info("예측 결과를 원본 값으로 변환 완료")
            except Exception as e:
                logger.warning(f"데이터 역변환 실패: {e}")
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=[f"<b>{col}</b>" for col in target_columns],
            vertical_spacing=vertical_spacing,
            specs=[[{"secondary_y": False}] for _ in target_columns]
        )
        
        # 현대적인 색상 팔레트
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, col in enumerate(target_columns):
            if col in actual_display.columns:
                # 실제 데이터
                fig.add_trace(
                    go.Scatter(
                        x=actual_display.index,
                        y=actual_display[col],
                        mode='lines+markers',
                        name=f'{col} (실제)',
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=6, color=colors[i % len(colors)]),
                        showlegend=(i == 0),
                        hovertemplate='<b>%{x}</b><br>실제값: %{y:,.0f}원<extra></extra>'
                    ),
                    row=i+1, col=1
                )
            
            if col in forecast_display.columns:
                # 예측 데이터
                fig.add_trace(
                    go.Scatter(
                        x=forecast_display.index,
                        y=forecast_display[col],
                        mode='lines+markers',
                        name=f'{col} (예측)',
                        line=dict(color=colors[i % len(colors)], width=3, dash='dash'),
                        marker=dict(size=6, color=colors[i % len(colors)], symbol='diamond'),
                        showlegend=(i == 0),
                        hovertemplate='<b>%{x}</b><br>예측값: %{y:,.0f}원<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>통신사 재무 예측 결과</b>",
                x=0.5,
                font=dict(size=24, color='#2c3e50')
            ),
            height=350 * len(target_columns),
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # 각 서브플롯 스타일링
        for i in range(len(target_columns)):
            fig.update_xaxes(
                title_text="날짜",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
            fig.update_yaxes(
                title_text="금액 (원)",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
        
        return fig
    
    def create_accuracy_plot(self, evaluation_results: Dict) -> go.Figure:
        """모델 정확도 비교 시각화 - 현대적 디자인"""
        # 평가 결과를 데이터프레임으로 변환
        accuracy_data = []
        
        for model_name, model_results in evaluation_results.items():
            for metric_name, metric_results in model_results.items():
                if isinstance(metric_results, dict):
                    for col, value in metric_results.items():
                        accuracy_data.append({
                            'Model': model_name.upper(),
                            'Metric': metric_name.upper(),
                            'Account': col,
                            'Value': value
                        })
        
        if not accuracy_data:
            return go.Figure()
        
        df_accuracy = pd.DataFrame(accuracy_data)
        
        # 메트릭별로 서브플롯 생성
        metrics = df_accuracy['Metric'].unique()
        n_rows = len(metrics)
        if n_rows <= 1:
            vertical_spacing = 0.1
        else:
            vertical_spacing = min(0.12, 1.0 / (n_rows + 1))
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=[f"<b>{metric}</b>" for metric in metrics],
            vertical_spacing=vertical_spacing
        )
        
        # 현대적인 색상 팔레트
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
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
                        showlegend=(i == 0),
                        marker_color=colors[j % len(colors)],
                        line_color=colors[j % len(colors)],
                        fillcolor='rgba(52, 152, 219, 0.1)',
                        hovertemplate='<b>%{fullData.name}</b><br>값: %{y:,.2f}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>모델 성능 비교</b>",
                x=0.5,
                font=dict(size=24, color='#2c3e50')
            ),
            height=350 * len(metrics),
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=100, b=80),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 각 서브플롯 스타일링
        for i in range(len(metrics)):
            fig.update_xaxes(
                title_text="모델",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
            fig.update_yaxes(
                title_text="값",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
        
        return fig
    
    def create_feature_importance_plot(self, processed_data: pd.DataFrame,
                                     target_columns: List[str]) -> go.Figure:
        """특성 중요도 시각화 (상관관계 기반) - 현대적 디자인"""
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
            colorscale=[
                [0, '#e74c3c'],    # 빨간색 (음의 상관관계)
                [0.5, '#ecf0f1'],  # 회색 (무상관)
                [1, '#3498db']     # 파란색 (양의 상관관계)
            ],
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="<b>%{text}</b>",
            textfont={"size": 11, "color": "#2c3e50"},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>상관계수: %{z:.3f}<extra></extra>'
        ))
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>계정과목 간 상관관계 분석</b>",
                x=0.5,
                font=dict(size=24, color='#2c3e50')
            ),
            width=900,
            height=700,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=100, r=100, t=120, b=100),
            xaxis=dict(
                title="계정과목",
                tickangle=45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="계정과목",
                tickfont=dict(size=10)
            )
        )
        
        return fig
    
    def create_seasonal_decomposition_plot(self, time_series_dict: Dict,
                                         target_columns: List[str]) -> go.Figure:
        """계절성 분해 시각화"""
        n_rows = len(target_columns)
        if n_rows <= 1:
            vertical_spacing = 0.1
        else:
            vertical_spacing = min(0.05, 1.0 / (n_rows + 1))
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=[f"{col} - 계절성 분해" for col in target_columns],
            vertical_spacing=vertical_spacing
        )
        
        for i, col in enumerate(target_columns):
            if col in time_series_dict:
                series = time_series_dict[col]
                values = series.values()
                # 다변량 시계열인 경우 1차원으로 변환
                if values.ndim > 1:
                    values = values.flatten()
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
        """대시보드 생성 - 현대적 디자인"""
        dashboard_config = self.config['visualization']['dashboard']
        
        # 대시보드 HTML 생성
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>통신사 재무 예측 대시보드</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body { 
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #2c3e50;
                }
                
                .container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    padding: 20px;
                }
                
                .header { 
                    text-align: center; 
                    margin-bottom: 40px;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                }
                
                .header h1 {
                    font-size: 2.5rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 10px;
                }
                
                .header p {
                    font-size: 1.1rem;
                    color: #7f8c8d;
                    font-weight: 500;
                }
                
                .summary { 
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px; 
                    border-radius: 20px; 
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                }
                
                .summary h2 {
                    font-size: 1.8rem;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #2c3e50;
                }
                
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                
                .summary-item {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                }
                
                .summary-item h3 {
                    font-size: 2rem;
                    font-weight: 700;
                    margin-bottom: 5px;
                }
                
                .summary-item p {
                    font-size: 0.9rem;
                    opacity: 0.9;
                }
                
                .chart { 
                    margin-bottom: 30px;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                }
                
                .chart h3 {
                    font-size: 1.5rem;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #2c3e50;
                    text-align: center;
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 10px;
                    }
                    
                    .header h1 {
                        font-size: 2rem;
                    }
                    
                    .summary-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 통신사 재무 예측 대시보드</h1>
                    <p>AI 기반 시계열 예측 분석 리포트</p>
                    <p style="margin-top: 10px; font-size: 0.9rem; color: #95a5a6;">생성일: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
                
                <div class="summary">
                    <h2>📈 예측 요약</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <h3>""" + str(len(results.get('ensemble_forecast', pd.DataFrame()))) + """</h3>
                            <p>예측 기간 (개월)</p>
                        </div>
                        <div class="summary-item">
                            <h3>""" + str(len(results.get('ensemble_forecast', pd.DataFrame()).columns)) + """</h3>
                            <p>예측 대상 계정과목</p>
                        </div>
                        <div class="summary-item">
                            <h3>""" + str(len(results.get('evaluation_results', {}))) + """</h3>
                            <p>사용 모델 수</p>
                        </div>
                    </div>
                </div>
                
                <div class="chart">
                    <h3>📊 예측 결과</h3>
                    <div id="forecast-chart"></div>
                </div>
                
                <div class="chart">
                    <h3>🎯 모델 성능 비교</h3>
                    <div id="accuracy-chart"></div>
                </div>
                
                <div class="chart">
                    <h3>🔗 상관관계 분석</h3>
                    <div id="correlation-chart"></div>
                </div>
                
                <div class="chart">
                    <h3>📅 계절성 분석</h3>
                    <div id="seasonal-chart"></div>
                </div>
                
                <div class="chart">
                    <h3>🏗️ 계층적 분석</h3>
                    <div id="hierarchical-chart"></div>
                </div>
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
                       target_columns: List[str],
                       data_processor=None) -> str:
        """종합 분석 리포트 생성"""
        logger.info("=== 시각화 리포트 생성 시작 ===")
        
        # 1. 예측 결과 시각화
        forecast_fig = self.create_forecast_plot(
            processed_data, 
            results.get('ensemble_forecast', pd.DataFrame()),
            target_columns,
            data_processor
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