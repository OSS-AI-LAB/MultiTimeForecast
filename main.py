#!/usr/bin/env python3
"""
통신사 재무 예측 메인 스크립트
Darts TFTModel을 활용한 계정과목별 매출 예측
"""

import sys
import os
import logging
from pathlib import Path
sys.path.append('src')

from src.data_processor import TelecomDataProcessor
from src.models import TelecomForecaster
from src.visualizer import TelecomVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    print("=== 통신사 재무 예측 시스템 ===")
    print("Darts TFTModel을 활용한 계정과목별 매출 예측\n")
    
    try:
        # 1. 데이터 처리기 초기화
        print("1. 데이터 처리기 초기화 중...")
        data_processor = TelecomDataProcessor()
        
        # 2. 원본 데이터 처리
        print("\n2. 원본 데이터 처리 중...")
        processed_data = data_processor.process_data()
        
        # 특성 정보 출력
        feature_info = data_processor.get_feature_info()
        print(f"   처리된 계정과목: {len(feature_info['account_columns'])}개")
        print(f"   처리된 제품: {len(feature_info['product_columns'])}개")
        print(f"   총 특성 수: {feature_info['total_features']}개")
        
        # 3. 예측기 초기화
        print("\n3. 예측기 초기화 중...")
        forecaster = TelecomForecaster()
        
        # 모델 설정 정보 출력
        model_config = data_processor.config['model']
        use_ensemble = model_config['use_ensemble']
        if use_ensemble:
            print(f"   모델: TFT + Prophet 앙상블")
            weights = model_config['ensemble']['weights']
            print(f"   가중치: TFT({weights[0]:.1%}), Prophet({weights[1]:.1%})")
        else:
            print(f"   모델: TFT만 사용")
        
        # 4. 타겟 컬럼 정의 (주요 계정과목들)
        target_columns = feature_info['account_columns'][:10]  # 상위 10개 계정과목
        print(f"   예측 대상 계정과목: {len(target_columns)}개")
        for i, col in enumerate(target_columns[:5]):
            print(f"     {i+1}. {col}")
        if len(target_columns) > 5:
            print(f"     ... 외 {len(target_columns)-5}개")
        
        # 5. 예측 파이프라인 실행
        print("\n4. 예측 파이프라인 실행 중...")
        forecast_config = data_processor.config['forecasting']
        forecast_horizon = forecast_config['forecast_horizon']
        
        results = forecaster.run_forecast_pipeline(
            processed_data=processed_data,
            target_columns=target_columns,
            forecast_horizon=forecast_horizon
        )
        
        # 6. 시각화 리포트 생성
        print("\n5. 시각화 리포트 생성 중...")
        visualizer = TelecomVisualizer()
        report_path = visualizer.generate_report(
            processed_data=processed_data,
            results=results,
            target_columns=target_columns,
            data_processor=data_processor
        )
        
        # 7. 결과 요약
        print("\n=== 예측 완료 ===")
        print(f"예측 기간: {forecast_horizon}개월")
        print(f"예측된 계정과목: {len(target_columns)}개")
        
        # 최종 예측값 출력
        if 'ensemble_forecast' in results and not results['ensemble_forecast'].empty:
            print("\n최종 예측값 (12개월 후):")
            final_forecast = results['ensemble_forecast'].iloc[-1]
            for col in target_columns[:5]:  # 상위 5개만 출력
                if col in final_forecast:
                    print(f"  {col}: {final_forecast[col]:,.0f}원")
            if len(target_columns) > 5:
                print(f"  ... 외 {len(target_columns)-5}개 계정과목")
        
        # 성장률 분석
        if 'ensemble_forecast' in results and not results['ensemble_forecast'].empty:
            print("\n성장률 분석 (12개월):")
            latest_actual = processed_data[target_columns].iloc[-1]
            final_forecast = results['ensemble_forecast'].iloc[-1]
            
            for col in target_columns[:5]:
                if col in latest_actual and col in final_forecast:
                    if latest_actual[col] != 0:
                        growth_rate = ((final_forecast[col] - latest_actual[col]) / abs(latest_actual[col])) * 100
                        print(f"  {col}: {growth_rate:+.2f}%")
        
        # 모델 성능 요약
        if 'evaluation_results' in results:
            print("\n모델 성능 요약:")
            for model_name, model_results in results['evaluation_results'].items():
                if isinstance(model_results, dict):
                    # 평균 MAE 계산
                    mae_values = []
                    for metric_name, metric_results in model_results.items():
                        if metric_name == 'mae' and isinstance(metric_results, dict):
                            mae_values.extend(metric_results.values())
                    
                    if mae_values:
                        avg_mae = sum(mae_values) / len(mae_values)
                        print(f"  {model_name.upper()}: 평균 MAE = {avg_mae:,.0f}원")
        
        # 결과 파일 위치
        print(f"\n결과 파일 위치:")
        print(f"  - 예측 결과: results/forecast_results.csv")
        print(f"  - 평가 결과: results/evaluation_results.csv")
        print(f"  - 예측 차트: results/forecast_plot.html")
        print(f"  - 정확도 차트: results/accuracy_plot.html")
        print(f"  - 상관관계 차트: results/correlation_plot.html")
        print(f"  - 계절성 차트: results/seasonal_plot.html")

        print(f"  - 대시보드: {report_path}")
        
        print("\n=== 시스템 실행 완료 ===")
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {str(e)}")
        print(f"\n오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 