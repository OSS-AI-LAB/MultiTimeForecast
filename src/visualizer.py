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

# 분할된 모듈 import
try:
    from .chart_creators import ChartCreators
except ImportError:
    from chart_creators import ChartCreators

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
        """예측 결과 시각화 - 분할된 모듈 사용"""
        return ChartCreators.create_forecast_plot(actual_data, forecast_data, target_columns, data_processor)
    
    def create_accuracy_plot(self, evaluation_results: Dict) -> go.Figure:
        """모델 정확도 비교 시각화 - 분할된 모듈 사용"""
        return ChartCreators.create_accuracy_plot(evaluation_results)
    
    def create_model_comparison_summary(self, evaluation_results: Dict) -> go.Figure:
        """모델 비교 요약 - 승률과 성능 개선율"""
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
        
        # 모델별 성능 요약 계산
        model_summary = {}
        metrics = df_accuracy['Metric'].unique()
        
        for metric in metrics:
            metric_data = df_accuracy[df_accuracy['Metric'] == metric]
            model_means = metric_data.groupby('Model')['Value'].mean()
            
            # 성능 순위 계산
            is_lower_better = metric in ['MAE', 'RMSE', 'MAPE']
            if is_lower_better:
                sorted_models = model_means.sort_values()
            else:
                sorted_models = model_means.sort_values(ascending=False)
            
            # 각 모델의 승률 계산 (다른 모델 대비 더 좋은 성능을 보인 비율)
            win_rates = {}
            for model in model_means.index:
                wins = 0
                total_comparisons = 0
                
                for other_model in model_means.index:
                    if model != other_model:
                        total_comparisons += 1
                        if is_lower_better:
                            if model_means[model] < model_means[other_model]:
                                wins += 1
                        else:
                            if model_means[model] > model_means[other_model]:
                                wins += 1
                
                win_rates[model] = (wins / total_comparisons) * 100 if total_comparisons > 0 else 0
            
            model_summary[metric] = {
                'means': model_means,
                'rankings': sorted_models,
                'win_rates': win_rates
            }
        
        # 2x2 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🏆 모델별 승률 (%)",
                "📊 평균 성능 순위",
                "💡 성능 개선율 (%)",
                "🎯 종합 평가"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 승률 차트
        all_win_rates = {}
        for metric, summary in model_summary.items():
            for model, rate in summary['win_rates'].items():
                if model not in all_win_rates:
                    all_win_rates[model] = []
                all_win_rates[model].append(rate)
        
        # 평균 승률 계산
        avg_win_rates = {model: np.mean(rates) for model, rates in all_win_rates.items()}
        models = list(avg_win_rates.keys())
        win_rates = list(avg_win_rates.values())
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=win_rates,
                name="승률",
                marker_color=['#e74c3c' if rate < 50 else '#2ecc71' for rate in win_rates],
                text=[f'{rate:.1f}%' for rate in win_rates],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>승률: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. 평균 성능 순위
        avg_rankings = {}
        for metric, summary in model_summary.items():
            for i, model in enumerate(summary['rankings'].index):
                if model not in avg_rankings:
                    avg_rankings[model] = []
                avg_rankings[model].append(i + 1)
        
        avg_ranks = {model: np.mean(ranks) for model, ranks in avg_rankings.items()}
        models_rank = list(avg_ranks.keys())
        ranks = list(avg_ranks.values())
        
        fig.add_trace(
            go.Bar(
                x=models_rank,
                y=ranks,
                name="평균 순위",
                marker_color=['#3498db' if rank <= 2 else '#f39c12' for rank in ranks],
                text=[f'{rank:.1f}위' for rank in ranks],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>평균 순위: %{y:.1f}위<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. 성능 개선율 (최고 성능 대비)
        improvement_rates = {}
        for metric, summary in model_summary.items():
            best_value = summary['means'].iloc[0] if metric in ['MAE', 'RMSE', 'MAPE'] else summary['means'].iloc[-1]
            
            for model, value in summary['means'].items():
                if model not in improvement_rates:
                    improvement_rates[model] = []
                
                if metric in ['MAE', 'RMSE', 'MAPE']:
                    # 낮을수록 좋은 지표: 최고 성능 대비 얼마나 나쁜지
                    improvement = ((value - best_value) / best_value) * 100
                else:
                    # 높을수록 좋은 지표: 최고 성능 대비 얼마나 나쁜지
                    improvement = ((best_value - value) / best_value) * 100
                
                improvement_rates[model].append(improvement)
        
        avg_improvements = {model: np.mean(rates) for model, rates in improvement_rates.items()}
        models_imp = list(avg_improvements.keys())
        improvements = list(avg_improvements.values())
        
        fig.add_trace(
            go.Bar(
                x=models_imp,
                y=improvements,
                name="성능 개선율",
                marker_color=['#e74c3c' if imp > 20 else '#f39c12' if imp > 10 else '#2ecc71' for imp in improvements],
                text=[f'{imp:.1f}%' for imp in improvements],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>개선 필요: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. 종합 평가 (점수화)
        scores = {}
        score_details = {}
        for model in models:
            # 승률 점수 (0-40점)
            win_score = avg_win_rates[model] * 0.4
            
            # 순위 점수 (0-30점) - 1위=30점, 2위=20점, 3위=10점
            rank_score = max(0, 30 - (avg_ranks[model] - 1) * 10)
            
            # 개선율 점수 (0-30점) - 개선율이 낮을수록 높은 점수
            imp_score = max(0, 30 - avg_improvements[model] * 1.5)
            
            total_score = win_score + rank_score + imp_score
            scores[model] = total_score
            
            # 점수 세부 내역 저장
            score_details[model] = {
                '승률점수': win_score,
                '순위점수': rank_score,
                '개선율점수': imp_score,
                '총점': total_score
            }
        
        models_score = list(scores.keys())
        score_values = list(scores.values())
        
        fig.add_trace(
            go.Bar(
                x=models_score,
                y=score_values,
                name="종합 점수",
                marker_color=['#2ecc71' if score > 70 else '#f39c12' if score > 50 else '#e74c3c' for score in score_values],
                text=[f'{score:.0f}점' for score in score_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>종합 점수: %{y:.0f}점<extra></extra>',
                customdata=[[
                    f"승률점수: {score_details[model]['승률점수']:.1f}점<br>순위점수: {score_details[model]['순위점수']:.1f}점<br>개선율점수: {score_details[model]['개선율점수']:.1f}점"
                    for model in models_score
                ]]
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>🎯 모델 성능 종합 분석</b><br><sub>승률, 순위, 개선율, 종합 점수로 모델 우수성 평가</sub>",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=800,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=120, b=80),
            showlegend=False
        )
        
        # 각 서브플롯 스타일링 (파이 차트 제외)
        for i in range(1, 3):
            for j in range(1, 3):
                # 파이 차트는 스킵
                if i == 1 and j == 2:
                    continue
                    
                fig.update_xaxes(
                    title_text="모델" if i == 2 and j == 1 else "날짜",
                    gridcolor='rgba(128,128,128,0.2)',
                    row=i, col=j
                )
                fig.update_yaxes(
                    title_text="성장률 (%)" if i == 2 and j == 1 else "금액 (원)",
                    gridcolor='rgba(128,128,128,0.2)',
                    row=i, col=j
                )
        
        return fig
    
    def create_feature_importance_plot(self, processed_data: pd.DataFrame,
                                     target_columns: List[str]) -> go.Figure:
        """특성 중요도 시각화 (상관관계 기반) - 개선된 디자인"""
        # 계정과목 컬럼만 선택
        account_cols = [col for col in processed_data.columns 
                       if col not in ['year', 'month', 'quarter', 'sin_month', 'cos_month', 
                                    'sin_quarter', 'cos_quarter', 'year_since_start']]
        
        # 상관관계 계산
        correlation_matrix = processed_data[account_cols].corr()
        
        # 중요 상관관계 식별 (절댓값 0.7 이상)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= 0.7:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'type': '강한 양의 상관관계' if corr_value > 0 else '강한 음의 상관관계'
                    })
        
        # 상관관계 강도별 색상 스케일
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale=[
                [0, '#e74c3c'],    # 빨간색 (강한 음의 상관관계)
                [0.3, '#f39c12'],  # 주황색 (약한 음의 상관관계)
                [0.5, '#ecf0f1'],  # 회색 (무상관)
                [0.7, '#3498db'],  # 파란색 (약한 양의 상관관계)
                [1, '#2ecc71']     # 초록색 (강한 양의 상관관계)
            ],
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="<b>%{text}</b>",
            textfont={"size": 10, "color": "#2c3e50"},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>상관계수: %{z:.3f}<br>해석: %{customdata}<extra></extra>',
            customdata=[[
                '강한 양의 상관관계' if val > 0.7 else
                '약한 양의 상관관계' if val > 0.3 else
                '약한 음의 상관관계' if val < -0.3 else
                '강한 음의 상관관계' if val < -0.7 else
                '무상관관계'
                for val in row
            ] for row in correlation_matrix.values]
        ))
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>🔗 계정과목 간 상관관계 분석</b><br><sub>강한 상관관계(±0.7 이상) 하이라이트</sub>",
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            width=700,  # 크기 축소
            height=600,  # 크기 축소
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=100, b=80),  # 여백 축소
            xaxis=dict(
                title="계정과목",
                tickangle=45,
                tickfont=dict(size=8)
            ),
            yaxis=dict(
                title="계정과목",
                tickfont=dict(size=8)
            )
        )
        
        # 중요 상관관계 정보 추가
        if strong_correlations:
            info_text = "<b>🔍 주요 발견사항:</b><br>"
            for i, corr in enumerate(strong_correlations[:5]):  # 상위 5개만
                info_text += f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.2f}<br>"
            
            fig.add_annotation(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=info_text,
                showarrow=False,
                font=dict(size=10, color='#2c3e50'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#3498db',
                borderwidth=1
            )
        
        return fig
    
    def create_seasonal_decomposition_plot(self, time_series_dict: Dict,
                                         target_columns: List[str]) -> go.Figure:
        """계절성 분해 시각화 - 개선된 디자인"""
        n_rows = len(target_columns)
        if n_rows <= 1:
            vertical_spacing = 0.1
        else:
            vertical_spacing = min(0.08, 1.0 / (n_rows + 1))
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=[f"<b>{col} - 계절성 분석</b>" for col in target_columns],
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
                
                # 개선된 계절성 분석
                if len(values) >= 12:
                    # 트렌드 (12개월 이동평균)
                    trend = pd.Series(values).rolling(window=12, center=True).mean()
                    
                    # 계절성 (월별 평균 편차)
                    df = pd.DataFrame({'date': dates, 'value': values})
                    df['month'] = pd.to_datetime(df['date']).dt.month
                    monthly_means = df.groupby('month')['value'].mean()
                    overall_mean = df['value'].mean()
                    seasonal_pattern = monthly_means - overall_mean
                    
                    # 계절성 성분 계산
                    seasonal_values = []
                    for date in dates:
                        month = pd.to_datetime(date).month
                        seasonal_values.append(seasonal_pattern.get(month, 0))
                    seasonal_values = pd.Series(seasonal_values, index=dates)
                    
                    # 잔차
                    residual = pd.Series(values) - trend - seasonal_values
                    
                    # 계절성 강도 계산
                    seasonal_strength = (seasonal_values.std() / pd.Series(values).std()) * 100
                    
                    # 원본 데이터
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=values,
                            mode='lines',
                            name=f'{col} (원본)',
                            line=dict(color='#3498db', width=2),
                            showlegend=(i == 0),
                            hovertemplate='<b>%{x}</b><br>원본값: %{y:,.0f}<extra></extra>'
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
                            line=dict(color='#e74c3c', width=3),
                            showlegend=(i == 0),
                            hovertemplate='<b>%{x}</b><br>트렌드: %{y:,.0f}<extra></extra>'
                        ),
                        row=i+1, col=1
                    )
                    
                    # 계절성 성분
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=seasonal_values + trend,  # 트렌드에 계절성 추가
                            mode='lines',
                            name=f'{col} (계절성)',
                            line=dict(color='#2ecc71', width=2, dash='dot'),
                            showlegend=(i == 0),
                            hovertemplate='<b>%{x}</b><br>계절성: %{y:,.0f}<extra></extra>'
                        ),
                        row=i+1, col=1
                    )
                    
                    # 계절성 강도 정보 추가
                    strength_color = '#e74c3c' if seasonal_strength > 30 else '#f39c12' if seasonal_strength > 15 else '#2ecc71'
                    fig.add_annotation(
                        x=0.02, y=0.95,
                        xref=f'x{i+1}', yref=f'y{i+1}',
                        text=f'📅 계절성 강도: {seasonal_strength:.1f}%<br>{"🔴 강함" if seasonal_strength > 30 else "🟡 보통" if seasonal_strength > 15 else "🟢 약함"}',
                        showarrow=False,
                        font=dict(size=10, color=strength_color),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor=strength_color,
                        borderwidth=1
                    )
                    
                    # 월별 패턴 정보 추가 (우상단)
                    peak_month = seasonal_pattern.idxmax()
                    trough_month = seasonal_pattern.idxmin()
                    fig.add_annotation(
                        x=0.98, y=0.95,
                        xref=f'x{i+1}', yref=f'y{i+1}',
                        text=f'📈 최고점: {peak_month}월<br>📉 최저점: {trough_month}월',
                        showarrow=False,
                        font=dict(size=9, color='#2c3e50'),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#bdc3c7',
                        borderwidth=1
                    )
        
        # 각 서브플롯의 X축 범위 설정
        for i in range(len(target_columns)):
            if target_columns[i] in time_series_dict:
                series = time_series_dict[target_columns[i]]
                dates = series.time_index
                if len(dates) > 0:
                    try:
                        fig.update_xaxes(
                            range=[dates.min(), dates.max()],
                            title_text="날짜",
                            gridcolor='rgba(128,128,128,0.2)',
                            row=i+1, col=1
                        )
                    except Exception as e:
                        logger.warning(f"계절성 차트 X축 범위 설정 실패 (차트 {i+1}): {e}")
                        fig.update_xaxes(
                            title_text="날짜",
                            gridcolor='rgba(128,128,128,0.2)',
                            row=i+1, col=1
                        )
                else:
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
        
        fig.update_layout(
            title=dict(
                text="<b>📅 계절성 분해 분석 - 월별 패턴 탐색</b><br><sub>원본, 트렌드, 계절성 성분 분리 및 계절성 강도 측정</sub>",
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            height=350 * len(target_columns),
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=120, b=80)
        )
        
        return fig
    

    
    def create_dashboard(self, results: Dict, processed_data: pd.DataFrame = None, target_columns: List[str] = None, data_processor=None) -> str:
        """대시보드 생성 - 완전한 차트 포함"""
        
        # 차트 생성
        charts_html = ""
        
        try:
            # 1. 예측 결과 차트 - 개별 파일과 동일한 방식으로 생성
            if 'ensemble_forecast' in results and processed_data is not None and target_columns is not None:
                try:
                    # 개별 파일과 동일한 방식으로 차트 생성
                    forecast_fig = self.create_forecast_plot(processed_data, results['ensemble_forecast'], target_columns, data_processor)
                    
                    # 디버깅을 위한 로그 추가
                    logger.info(f"대시보드 예측 차트 생성: {len(processed_data)} 행, {len(target_columns)} 컬럼")
                    logger.info(f"예측 데이터: {len(results['ensemble_forecast'])} 행, {len(results['ensemble_forecast'].columns)} 컬럼")
                    
                    charts_html += f"""
                    <div class="chart">
                        <h3>📊 예측 결과</h3>
                        <div id="forecast-chart">{forecast_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
                    </div>
                    """
                except Exception as e:
                    logger.error(f"대시보드 예측 차트 생성 실패: {e}")
                    charts_html += """
                    <div class="chart">
                        <h3>📊 예측 결과</h3>
                        <p>차트 생성 중 오류가 발생했습니다. forecast_plot.html을 확인해주세요.</p>
                    </div>
                    """
            
            # 2. 모델 성능 비교 차트
            if 'evaluation_results' in results:
                accuracy_fig = self.create_accuracy_plot(results['evaluation_results'])
                charts_html += f"""
                <div class="chart">
                    <h3>🎯 모델 성능 비교</h3>
                    <div id="accuracy-chart">{accuracy_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
                </div>
                """
            
            # 3. 상관관계 분석 차트
            if processed_data is not None and target_columns is not None:
                correlation_fig = self.create_feature_importance_plot(processed_data, target_columns)
                charts_html += f"""
                <div class="chart">
                    <h3>🔗 상관관계 분석</h3>
                    <div id="correlation-chart">{correlation_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
                </div>
                """
            
            # 4. 계절성 분석 차트
            if 'time_series_dict' in results and target_columns is not None:
                seasonal_fig = self.create_seasonal_decomposition_plot(results['time_series_dict'], target_columns)
                charts_html += f"""
                <div class="chart">
                    <h3>📅 계절성 분석</h3>
                    <div id="seasonal-chart">{seasonal_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
                </div>
                """
            

                
        except Exception as e:
            logger.error(f"대시보드 차트 생성 중 오류: {e}")
            charts_html = """
            <div class="chart">
                <h3>⚠️ 차트 로딩 오류</h3>
                <p>차트를 불러오는 중 오류가 발생했습니다. 개별 HTML 파일을 확인해주세요.</p>
            </div>
            """
        
        # 대시보드 HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>통신사 재무 예측 대시보드</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #2c3e50;
                }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 40px;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .header h1 {{
                    font-size: 2.2rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 10px;
                }}
                .header p {{ font-size: 1rem; color: #7f8c8d; font-weight: 500; }}
                .summary {{ 
                    background: rgba(255, 255, 255, 0.95);
                    padding: 25px; 
                    border-radius: 15px; 
                    margin-bottom: 25px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                }}
                .summary h2 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 15px; color: #2c3e50; }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .summary-item {{
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 18px;
                    border-radius: 12px;
                    text-align: center;
                }}
                .summary-item h3 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 5px; }}
                .summary-item p {{ font-size: 0.85rem; opacity: 0.9; }}
                .chart {{ 
                    margin-bottom: 25px;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .chart h3 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 20px; color: #2c3e50; text-align: center; }}
                @media (max-width: 768px) {{
                    .container {{ padding: 10px; }}
                    .header h1 {{ font-size: 2rem; }}
                    .summary-grid {{ grid-template-columns: 1fr; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 통신사 재무 예측 대시보드</h1>
                    <p>AI 기반 시계열 예측 분석 리포트</p>
                    <p style="margin-top: 10px; font-size: 0.9rem; color: #95a5a6;">생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="summary">
                    <h2>📈 예측 요약</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <h3>{len(results.get('ensemble_forecast', pd.DataFrame()))}</h3>
                            <p>예측 기간 (개월)</p>
                        </div>
                        <div class="summary-item">
                            <h3>{len(results.get('ensemble_forecast', pd.DataFrame()).columns)}</h3>
                            <p>예측 대상 계정과목</p>
                        </div>
                        <div class="summary-item">
                            <h3>{len(results.get('evaluation_results', {}))}</h3>
                            <p>사용 모델 수</p>
                        </div>
                    </div>
                </div>
                
                {charts_html}
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
        
        # 2-1. 모델 비교 요약 시각화 (새로 추가)
        comparison_fig = self.create_model_comparison_summary(
            results.get('evaluation_results', {})
        )
        comparison_fig.write_html(self.results_dir / "model_comparison_summary.html")
        
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
        

        
        # 6. 대시보드 생성
        dashboard_path = self.create_dashboard(results, processed_data, target_columns, data_processor)
        
        # 7. 예측 결과 CSV 저장
        if 'ensemble_forecast' in results:
            results['ensemble_forecast'].to_csv(self.results_dir / "forecast_results.csv")
        
        # 8. 평가 결과 CSV 저장
        if 'evaluation_results' in results:
            evaluation_data = []
            for model_name, model_results in results['evaluation_results'].items():
                for metric_name, metric_results in model_results.items():
                    if isinstance(metric_results, dict):
                        for col, value in metric_results.items():
                            evaluation_data.append({
                                'Model': model_name,
                                'Metric': metric_name,
                                'Account': col,
                                'Value': value
                            })
            
            evaluation_df = pd.DataFrame(evaluation_data)
            
            evaluation_df.to_csv(self.results_dir / "evaluation_results.csv", index=False)
        
        logger.info("=== 시각화 리포트 생성 완료 ===")
        return dashboard_path 