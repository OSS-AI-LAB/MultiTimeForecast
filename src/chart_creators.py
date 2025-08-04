"""
차트 생성 모듈
각종 시각화 차트를 생성하는 함수들을 포함
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class ChartCreators:
    """차트 생성 클래스"""
    
    @staticmethod
    def create_forecast_plot(actual_data: pd.DataFrame, 
                           forecast_data: pd.DataFrame,
                           target_columns: list,
                           data_processor=None) -> go.Figure:
        """예측 결과 시각화 - 개선된 디자인 (성장률, 신뢰구간 포함)"""
        # vertical_spacing을 동적으로 계산하여 오류 방지
        n_rows = len(target_columns)
        if n_rows <= 1:
            vertical_spacing = 0.1
        else:
            # matplotlib 제약: vertical_spacing <= 1.0 / (rows - 1)
            max_spacing = 1.0 / (n_rows - 1)
            vertical_spacing = min(0.08, max_spacing * 0.8)  # 안전 마진 적용
        
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
            if col in actual_display.columns and col in forecast_display.columns:
                # 성장률 계산
                actual_values = actual_display[col].dropna()
                forecast_values = forecast_display[col].dropna()
                
                if len(actual_values) > 0 and len(forecast_values) > 0:
                    # 최근 실제값과 예측값 비교
                    recent_actual = actual_values.iloc[-1]
                    first_forecast = forecast_values.iloc[0]
                    last_forecast = forecast_values.iloc[-1]
                    
                    # 단기 성장률 (최근 실제 → 첫 예측)
                    short_growth = ((first_forecast - recent_actual) / recent_actual) * 100 if recent_actual != 0 else 0
                    
                    # 장기 성장률 (최근 실제 → 마지막 예측)
                    long_growth = ((last_forecast - recent_actual) / recent_actual) * 100 if recent_actual != 0 else 0
                    
                    # 예측 신뢰구간 (더 현실적인 방법: 예측값의 ±15%)
                    upper_bound = forecast_values * 1.15
                    lower_bound = forecast_values * 0.85
                    
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
                    
                    # 예측 신뢰구간 (개선된 방법)
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_display.index,
                            y=upper_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            fill=None
                        ),
                        row=i+1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_display.index,
                            y=lower_bound,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(52, 152, 219, 0.1)',
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=i+1, col=1
                    )
                    
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
                    
                    # 성장률 정보 추가 (더 현실적인 표시)
                    growth_color = '#2ecc71' if long_growth > 0 else '#e74c3c'
                    growth_text = f'📈 단기: {short_growth:+.1f}%<br>📊 장기: {long_growth:+.1f}%'
                    
                    # 데이터 부족 경고 추가 (29개월치 데이터)
                    if len(actual_values) < 36:  # 3년 미만
                        growth_text += '<br>⚠️ 데이터 부족 (29개월)'
                        growth_color = '#f39c12'  # 주황색으로 경고
                    
                    fig.add_annotation(
                        x=0.02, y=0.95,
                        xref=f'x{i+1}', yref=f'y{i+1}',
                        text=growth_text,
                        showarrow=False,
                        font=dict(size=9, color=growth_color),
                        bgcolor='rgba(255,255,255,0.95)',
                        bordercolor=growth_color,
                        borderwidth=1
                    )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>📈 통신사 재무 예측 결과 - 성장률 분석</b><br><sub>실제값 vs 예측값, 성장률, 신뢰구간 포함</sub>",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=350 * len(target_columns),
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=120, b=80)
        )
        
        # 각 서브플롯 스타일링 - X축 범위 개선
        for i in range(len(target_columns)):
            try:
                # 전체 데이터 범위 계산
                all_data = pd.concat([actual_display, forecast_display])
                if not all_data.empty:
                    # X축 범위를 전체 데이터에 맞게 설정
                    fig.update_xaxes(
                        title_text="날짜",
                        gridcolor='rgba(128,128,128,0.2)',
                        range=[all_data.index.min(), all_data.index.max()],
                        row=i+1, col=1
                    )
                else:
                    fig.update_xaxes(
                        title_text="날짜",
                        gridcolor='rgba(128,128,128,0.2)',
                        row=i+1, col=1
                    )
            except Exception as e:
                logger.warning(f"X축 범위 설정 실패 (차트 {i+1}): {e}")
                # 기본 X축 설정
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
    
    @staticmethod
    def create_accuracy_plot(evaluation_results: dict) -> go.Figure:
        """모델 정확도 비교 시각화 - 완전히 리뉴얼된 깔끔한 디자인"""
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
        
        # 메트릭별로 서브플롯 생성 (단일 열 레이아웃으로 변경)
        metrics = df_accuracy['Metric'].unique()
        n_metrics = len(metrics)
        
        # 서브플롯 간격을 동적으로 계산
        if n_metrics <= 1:
            vertical_spacing = 0.1
        else:
            # matplotlib 제약: vertical_spacing <= 1.0 / (rows - 1)
            max_spacing = 1.0 / (n_metrics - 1)
            vertical_spacing = min(0.15, max_spacing * 0.8)  # 안전 마진 적용
        
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=[f"<b>{metric} - 모델 성능 비교</b>" for metric in metrics],
            vertical_spacing=vertical_spacing,
            specs=[[{"secondary_y": False}] for _ in range(n_metrics)]
        )
        
        # 현대적인 색상 팔레트
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
        
        for i, metric in enumerate(metrics):
            metric_data = df_accuracy[df_accuracy['Metric'] == metric]
            
            # 모델별 평균값 계산
            model_means = metric_data.groupby('Model')['Value'].mean().sort_values()
            models = model_means.index.tolist()
            means = model_means.values.tolist()
            
            # 성능 순위 계산 (낮을수록 좋은 지표: MAE, RMSE, MAPE)
            is_lower_better = metric in ['MAE', 'RMSE', 'MAPE']
            if is_lower_better:
                best_model = models[0]  # 가장 낮은 값
                worst_model = models[-1]  # 가장 높은 값
                performance_text = f"🏆 최고: {best_model}<br>❌ 최악: {worst_model}"
            else:
                best_model = models[-1]  # 가장 높은 값
                worst_model = models[0]  # 가장 낮은 값
                performance_text = f"🏆 최고: {best_model}<br>❌ 최악: {worst_model}"
            
            # 바 차트로 모델 성능 비교
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=means,
                    name=metric,
                    marker_color=[colors[j % len(colors)] for j in range(len(models))],
                    text=[f'{val:.3f}' for val in means],
                    textposition='outside',
                    textfont=dict(size=11, color='#2c3e50'),
                    hovertemplate='<b>%{x}</b><br>평균값: %{y:.3f}<br>순위: %{customdata}<extra></extra>',
                    customdata=[f"{j+1}위" for j in range(len(models))],
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # 성능 순위 텍스트 추가 (좌상단)
            fig.add_annotation(
                x=0.02, y=0.95,
                xref=f'x{i+1}', yref=f'y{i+1}',
                text=performance_text,
                showarrow=False,
                font=dict(size=11, color='#2c3e50'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#3498db',
                borderwidth=1
            )
            
            # 성능 차이 정보 추가 (우상단)
            if len(means) >= 2:
                best_value = means[0] if is_lower_better else means[-1]
                second_best = means[1] if is_lower_better else means[-2]
                improvement = ((second_best - best_value) / best_value) * 100 if best_value != 0 else 0
                
                improvement_text = f"💡 {best_model}<br>{abs(improvement):.1f}% 우수"
                
                fig.add_annotation(
                    x=0.98, y=0.95,
                    xref=f'x{i+1}', yref=f'y{i+1}',
                    text=improvement_text,
                    showarrow=False,
                    font=dict(size=10, color='#27ae60'),
                    bgcolor='rgba(39, 174, 96, 0.15)',
                    bordercolor='#27ae60',
                    borderwidth=1
                )
            
            # 모델별 표준편차 정보 추가 (바 위에)
            for j, model in enumerate(models):
                model_data = metric_data[metric_data['Model'] == model]
                std_val = model_data['Value'].std()
                
                # 표준편차 정보를 바 위에 표시
                fig.add_annotation(
                    x=j, y=means[j] + max(means) * 0.05,
                    xref=f'x{i+1}', yref=f'y{i+1}',
                    text=f'σ: {std_val:.3f}',
                    showarrow=False,
                    font=dict(size=9, color='#7f8c8d'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ecf0f1',
                    borderwidth=0.5
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="<b>🎯 모델 성능 비교 분석</b><br><sub>각 지표별 모델 순위와 성능 차이</sub>",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=300 * n_metrics,  # 각 차트당 충분한 높이
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, t=120, b=80),
            showlegend=False
        )
        
        # 각 서브플롯 스타일링
        for i in range(n_metrics):
            fig.update_xaxes(
                title_text="모델",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
            fig.update_yaxes(
                title_text="평균값",
                gridcolor='rgba(128,128,128,0.2)',
                row=i+1, col=1
            )
        
        return fig 