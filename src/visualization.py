"""
통신사 손익전망 시각화 모듈
TimesFM 모델 예측 결과를 다양한 차트로 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class TelecomVisualizer:
    """통신사 데이터 시각화 클래스"""
    
    def __init__(self, style: str = 'default'):
        """
        Args:
            style: 시각화 스타일 ('default', 'dark', 'light')
        """
        self.style = style
        self.set_style()
        
    def set_style(self):
        """시각화 스타일 설정"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            sns.set_style("darkgrid")
        elif self.style == 'light':
            plt.style.use('default')
            sns.set_style("whitegrid")
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")
    
    def plot_time_series(self, 
                        df: pd.DataFrame,
                        columns: List[str],
                        title: str = "시계열 데이터",
                        figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        시계열 데이터 플롯
        
        Args:
            df: 데이터프레임
            columns: 플롯할 컬럼들
            title: 차트 제목
            figsize: 차트 크기
            
        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
        if len(columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            if col in df.columns:
                axes[i].plot(df['date'], df[col], linewidth=2, marker='o', markersize=4)
                axes[i].set_title(f'{col} 추이', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('날짜')
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_forecast_comparison(self, 
                               historical_df: pd.DataFrame,
                               forecast_df: pd.DataFrame,
                               columns: List[str],
                               title: str = "예측 결과 비교") -> plt.Figure:
        """
        과거 데이터와 예측 결과 비교 플롯
        
        Args:
            historical_df: 과거 데이터
            forecast_df: 예측 데이터
            columns: 비교할 컬럼들
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(15, 4*n_cols), sharex=True)
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            if col in historical_df.columns and col in forecast_df.columns:
                # 과거 데이터
                axes[i].plot(historical_df['date'], historical_df[col], 
                           label='과거 데이터', linewidth=2, color='blue')
                
                # 예측 데이터
                axes[i].plot(forecast_df['date'], forecast_df[col], 
                           label='예측 데이터', linewidth=2, color='red', linestyle='--')
                
                axes[i].set_title(f'{col} 예측 결과', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(col)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('날짜')
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_profitability_analysis(self, 
                                  df: pd.DataFrame,
                                  title: str = "수익성 분석") -> plt.Figure:
        """
        수익성 분석 차트
        
        Args:
            df: 수익성 데이터가 포함된 데이터프레임
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 총 매출 vs 총 비용
        if 'total_revenue' in df.columns and 'total_cost' in df.columns:
            axes[0, 0].plot(df['date'], df['total_revenue'], label='총 매출', linewidth=2)
            axes[0, 0].plot(df['date'], df['total_cost'], label='총 비용', linewidth=2)
            axes[0, 0].set_title('총 매출 vs 총 비용', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 순이익
        if 'profit' in df.columns:
            axes[0, 1].plot(df['date'], df['profit'], label='순이익', linewidth=2, color='green')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('순이익 추이', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 수익률
        if 'profit_margin' in df.columns:
            axes[1, 0].plot(df['date'], df['profit_margin'] * 100, 
                           label='수익률 (%)', linewidth=2, color='purple')
            axes[1, 0].set_title('수익률 추이', fontweight='bold')
            axes[1, 0].set_ylabel('수익률 (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 기술별 사용자 점유율
        tech_cols = ['5g_users', 'lte_users', '3g_users']
        available_techs = [col for col in tech_cols if col in df.columns]
        
        if available_techs:
            tech_data = df[available_techs].values
            tech_labels = [col.replace('_users', '').upper() for col in available_techs]
            
            axes[1, 1].stackplot(df['date'], tech_data.T, labels=tech_labels, alpha=0.7)
            axes[1, 1].set_title('기술별 사용자 점유율', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        return fig
    
    def plot_technology_comparison(self, 
                                 df: pd.DataFrame,
                                 title: str = "기술별 비교 분석") -> plt.Figure:
        """
        기술별 비교 분석 차트
        
        Args:
            df: 기술별 데이터가 포함된 데이터프레임
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        technologies = ['5g', 'lte', '3g']
        available_techs = [tech for tech in technologies 
                          if f'{tech}_users' in df.columns]
        
        if not available_techs:
            raise ValueError("기술별 데이터가 없습니다.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 사용자 수 비교
        for tech in available_techs:
            if f'{tech}_users' in df.columns:
                axes[0, 0].plot(df['date'], df[f'{tech}_users'], 
                               label=tech.upper(), linewidth=2)
        axes[0, 0].set_title('기술별 사용자 수', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 매출 비교
        for tech in available_techs:
            if f'{tech}_revenue' in df.columns:
                axes[0, 1].plot(df['date'], df[f'{tech}_revenue'], 
                               label=f'{tech.upper()} 매출', linewidth=2)
        axes[0, 1].set_title('기술별 매출', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 비용 비교
        for tech in available_techs:
            if f'{tech}_cost' in df.columns:
                axes[1, 0].plot(df['date'], df[f'{tech}_cost'], 
                               label=f'{tech.upper()} 비용', linewidth=2)
        axes[1, 0].set_title('기술별 비용', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 수익성 비교 (매출 - 비용)
        for tech in available_techs:
            if f'{tech}_revenue' in df.columns and f'{tech}_cost' in df.columns:
                profit = df[f'{tech}_revenue'] - df[f'{tech}_cost']
                axes[1, 1].plot(df['date'], profit, 
                               label=f'{tech.upper()} 수익', linewidth=2)
        axes[1, 1].set_title('기술별 수익', fontweight='bold')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   historical_df: pd.DataFrame,
                                   forecast_df: pd.DataFrame) -> go.Figure:
        """
        대화형 대시보드 생성 (Plotly)
        
        Args:
            historical_df: 과거 데이터
            forecast_df: 예측 데이터
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('사용자 수 추이', '매출 추이', 
                          '비용 추이', '수익 추이',
                          '수익률 추이', '기술별 점유율'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 데이터 결합
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        # 1. 사용자 수 추이
        tech_cols = ['5g_users', 'lte_users', '3g_users']
        for tech in tech_cols:
            if tech in combined_df.columns:
                fig.add_trace(
                    go.Scatter(x=combined_df['date'], y=combined_df[tech],
                              name=tech.replace('_users', '').upper(),
                              mode='lines+markers'),
                    row=1, col=1
                )
        
        # 2. 매출 추이
        revenue_cols = ['5g_revenue', 'lte_revenue', '3g_revenue']
        for revenue in revenue_cols:
            if revenue in combined_df.columns:
                fig.add_trace(
                    go.Scatter(x=combined_df['date'], y=combined_df[revenue],
                              name=revenue.replace('_revenue', '').upper() + ' 매출',
                              mode='lines+markers'),
                    row=1, col=2
                )
        
        # 3. 비용 추이
        cost_cols = ['5g_cost', 'lte_cost', '3g_cost']
        for cost in cost_cols:
            if cost in combined_df.columns:
                fig.add_trace(
                    go.Scatter(x=combined_df['date'], y=combined_df[cost],
                              name=cost.replace('_cost', '').upper() + ' 비용',
                              mode='lines+markers'),
                    row=2, col=1
                )
        
        # 4. 수익 추이
        if 'profit' in combined_df.columns:
            fig.add_trace(
                go.Scatter(x=combined_df['date'], y=combined_df['profit'],
                          name='순이익', mode='lines+markers',
                          line=dict(color='green')),
                row=2, col=2
            )
        
        # 5. 수익률 추이
        if 'profit_margin' in combined_df.columns:
            fig.add_trace(
                go.Scatter(x=combined_df['date'], y=combined_df['profit_margin'] * 100,
                          name='수익률 (%)', mode='lines+markers',
                          line=dict(color='purple')),
                row=3, col=1
            )
        
        # 6. 기술별 점유율 (파이 차트)
        if tech_cols:
            latest_data = combined_df.iloc[-1]
            tech_values = [latest_data[tech] for tech in tech_cols if tech in latest_data]
            tech_labels = [tech.replace('_users', '').upper() for tech in tech_cols if tech in latest_data]
            
            if tech_values:
                fig.add_trace(
                    go.Pie(labels=tech_labels, values=tech_values,
                           name="최신 점유율"),
                    row=3, col=2
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="통신사 손익전망 대시보드",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_plots(self, 
                  figures: List[plt.Figure],
                  filenames: List[str],
                  output_dir: str = 'results'):
        """
        차트들을 파일로 저장
        
        Args:
            figures: 저장할 Figure 객체들
            filenames: 파일명들
            output_dir: 출력 디렉토리
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for fig, filename in zip(figures, filenames):
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"차트가 저장되었습니다: {filepath}")
    
    def generate_report(self, 
                       historical_df: pd.DataFrame,
                       forecast_df: pd.DataFrame,
                       output_dir: str = 'results') -> str:
        """
        종합 리포트 생성
        
        Args:
            historical_df: 과거 데이터
            forecast_df: 예측 데이터
            output_dir: 출력 디렉토리
            
        Returns:
            리포트 파일 경로
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 차트 생성
        figures = []
        filenames = []
        
        # 1. 시계열 데이터
        time_series_cols = ['5g_users', 'lte_users', '3g_users', 
                           'total_revenue', 'total_cost', 'profit']
        available_cols = [col for col in time_series_cols if col in historical_df.columns]
        
        if available_cols:
            fig1 = self.plot_time_series(historical_df, available_cols, "과거 데이터 추이")
            figures.append(fig1)
            filenames.append('historical_trends.png')
        
        # 2. 예측 비교
        if available_cols:
            fig2 = self.plot_forecast_comparison(historical_df, forecast_df, available_cols)
            figures.append(fig2)
            filenames.append('forecast_comparison.png')
        
        # 3. 수익성 분석
        fig3 = self.plot_profitability_analysis(historical_df)
        figures.append(fig3)
        filenames.append('profitability_analysis.png')
        
        # 4. 기술별 비교
        fig4 = self.plot_technology_comparison(historical_df)
        figures.append(fig4)
        filenames.append('technology_comparison.png')
        
        # 차트 저장
        self.save_plots(figures, filenames, output_dir)
        
        # 대화형 대시보드 생성
        dashboard = self.create_interactive_dashboard(historical_df, forecast_df)
        dashboard.write_html(os.path.join(output_dir, 'interactive_dashboard.html'))
        
        # 리포트 파일 생성
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 통신사 손익전망 분석 리포트 ===\n\n")
            
            # 기본 통계
            f.write("1. 데이터 개요\n")
            f.write(f"   - 과거 데이터 기간: {len(historical_df)}개월\n")
            f.write(f"   - 예측 기간: {len(forecast_df)}개월\n")
            f.write(f"   - 분석 컬럼 수: {len(available_cols)}\n\n")
            
            # 최신 데이터 요약
            f.write("2. 최신 데이터 요약 (과거 데이터 마지막)\n")
            latest = historical_df.iloc[-1]
            for col in available_cols:
                if col in latest:
                    f.write(f"   - {col}: {latest[col]:.2f}\n")
            f.write("\n")
            
            # 예측 결과 요약
            f.write("3. 예측 결과 요약 (12개월 후)\n")
            final_forecast = forecast_df.iloc[-1]
            for col in available_cols:
                if col in final_forecast:
                    f.write(f"   - {col}: {final_forecast[col]:.2f}\n")
            f.write("\n")
            
            # 성장률 분석
            f.write("4. 성장률 분석 (12개월)\n")
            for col in available_cols:
                if col in latest and col in final_forecast:
                    growth_rate = ((final_forecast[col] - latest[col]) / latest[col]) * 100
                    f.write(f"   - {col}: {growth_rate:.2f}%\n")
        
        print(f"리포트가 생성되었습니다: {report_path}")
        return report_path

if __name__ == "__main__":
    # 테스트 코드
    print("시각화 모듈 테스트")
    
    # 샘플 데이터 생성
    from data_processing import generate_sample_data
    
    sample_data = generate_sample_data()
    
    # 시각화 테스트
    visualizer = TelecomVisualizer()
    
    # 시계열 플롯
    fig = visualizer.plot_time_series(sample_data, ['5g_users', 'lte_users', '3g_users'])
    plt.show() 