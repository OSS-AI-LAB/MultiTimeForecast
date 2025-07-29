#!/usr/bin/env python3
"""
통신사 손익전망 예측 메인 스크립트
TFT (Temporal Fusion Transformer) 모델을 활용한 다변량 시계열 예측
"""

import sys
import os
sys.path.append('src')

from src.data_processing import generate_sample_data
from src.forecasting import TelecomForecaster
from src.visualization import TelecomVisualizer

def main():
    """메인 실행 함수"""
    print("=== 통신사 손익전망 예측 시스템 ===")
    print("TFT (Temporal Fusion Transformer) 모델 활용\n")
    
    # 1. 샘플 데이터 생성
    print("1. 샘플 데이터 생성 중...")
    sample_data = generate_sample_data(start_date='2020-01-01', periods=48)
    sample_data.to_csv('data/raw/sample_telecom_data.csv', index=False)
    print(f"   샘플 데이터 생성 완료: {sample_data.shape}")
    
    # 2. 예측기 초기화
    print("\n2. 예측기 초기화 중...")
    forecaster = TelecomForecaster()
    
    # 3. 타겟 컬럼 정의
    target_columns = [
        '5g_users', 'lte_users', '3g_users',
        '5g_revenue', 'lte_revenue', '3g_revenue',
        '5g_cost', 'lte_cost', '3g_cost'
    ]
    
    covariate_columns = [
        'month', 'quarter', 'year',
        'total_users', 'total_revenue', 'total_cost'
    ]
    
    print(f"   타겟 컬럼: {len(target_columns)}개")
    print(f"   공변량 컬럼: {len(covariate_columns)}개")
    
    # 4. 전체 파이프라인 실행
    print("\n3. 예측 파이프라인 실행 중...")
    results = forecaster.run_full_pipeline(
        file_path='data/raw/sample_telecom_data.csv',
        target_columns=target_columns,
        forecast_steps=12
    )
    
    # 5. 시각화 리포트 생성
    print("\n4. 시각화 리포트 생성 중...")
    visualizer = TelecomVisualizer()
    report_path = visualizer.generate_report(
        results['processed_data'], 
        results['forecast'], 
        'results'
    )
    
    # 6. 결과 요약
    print("\n=== 예측 완료 ===")
    print(f"예측 기간: {len(results['forecast'])}개월")
    print(f"예측된 변수: {len(target_columns)}개")
    
    # 최종 예측값 출력
    print("\n최종 예측값 (12개월 후):")
    final_forecast = results['forecast'].iloc[-1]
    for col in target_columns:
        if col in final_forecast:
            print(f"  {col}: {final_forecast[col]:.2f}")
    
    # 성장률 분석
    print("\n성장률 분석 (12개월):")
    latest_actual = results['processed_data'].iloc[-1]
    for col in target_columns:
        if col in latest_actual and col in final_forecast:
            growth_rate = ((final_forecast[col] - latest_actual[col]) / latest_actual[col]) * 100
            print(f"  {col}: {growth_rate:.2f}%")
    
    print(f"\n결과 파일 위치:")
    print(f"  - 예측 결과: results/forecast_results.csv")
    print(f"  - 수익성 분석: results/profitability_analysis.csv")
    print(f"  - 대시보드: results/interactive_dashboard.html")
    print(f"  - 분석 리포트: {report_path}")
    print(f"  - TFT 모델: results/tft_model.pth")
    
    print("\n=== 시스템 실행 완료 ===")

if __name__ == "__main__":
    main() 