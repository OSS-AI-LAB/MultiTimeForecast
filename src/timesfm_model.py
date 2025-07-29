"""
TimesFM 모델 래퍼
Google의 TimesFM 모델을 통신사 시계열 예측에 맞게 래핑
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class TimesFMWrapper:
    """TimesFM 모델 래퍼 클래스"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 device: str = 'auto'):
        """
        Args:
            input_size: 입력 특성 수
            hidden_size: 은닉층 크기
            num_layers: 레이어 수
            dropout: 드롭아웃 비율
            learning_rate: 학습률
            device: 사용할 디바이스
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 모델 초기화
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        print(f"TimesFM 모델이 {self.device}에서 초기화되었습니다.")
    
    def _build_model(self) -> nn.Module:
        """TimesFM 모델 아키텍처 구축"""
        
        class TimesFM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(TimesFM, self).__init__()
                
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # 인코더 (LSTM)
                self.encoder = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=True
                )
                
                # 디코더 (LSTM)
                self.decoder = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                
                # 어텐션 메커니즘
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True
                )
                
                # 출력 레이어
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, input_size)
                )
                
                # 드롭아웃
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, target_length=1):
                batch_size, seq_len, features = x.size()
                
                # 인코더
                encoder_output, (hidden, cell) = self.encoder(x)
                
                # 어텐션 적용
                attn_output, _ = self.attention(
                    encoder_output, encoder_output, encoder_output
                )
                
                # 디코더 입력 준비
                decoder_input = x[:, -1:, :]  # 마지막 시점의 입력
                
                # 디코더
                decoder_output, _ = self.decoder(
                    decoder_input, (hidden, cell)
                )
                
                # 어텐션 출력과 디코더 출력 결합
                combined = torch.cat([attn_output[:, -1, :], decoder_output[:, -1, :]], dim=1)
                
                # 출력 생성
                output = self.output_layer(combined)
                
                return output.unsqueeze(1)  # [batch_size, 1, features]
        
        return TimesFM(self.input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def prepare_data(self, 
                    input_data: np.ndarray, 
                    target_data: np.ndarray,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1) -> Tuple[torch.Tensor, ...]:
        """
        데이터 준비 및 분할
        
        Args:
            input_data: 입력 데이터
            target_data: 타겟 데이터
            train_ratio: 훈련 데이터 비율
            val_ratio: 검증 데이터 비율
            
        Returns:
            (train_input, train_target, val_input, val_target, test_input, test_target)
        """
        # 데이터 정규화
        input_scaled = self.scaler.fit_transform(
            input_data.reshape(-1, input_data.shape[-1])
        ).reshape(input_data.shape)
        
        target_scaled = self.target_scaler.fit_transform(target_data)
        
        # 데이터 분할
        total_samples = len(input_scaled)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # 훈련 데이터
        train_input = torch.FloatTensor(input_scaled[:train_size]).to(self.device)
        train_target = torch.FloatTensor(target_scaled[:train_size]).to(self.device)
        
        # 검증 데이터
        val_input = torch.FloatTensor(input_scaled[train_size:train_size+val_size]).to(self.device)
        val_target = torch.FloatTensor(target_scaled[train_size:train_size+val_size]).to(self.device)
        
        # 테스트 데이터
        test_input = torch.FloatTensor(input_scaled[train_size+val_size:]).to(self.device)
        test_target = torch.FloatTensor(target_scaled[train_size+val_size:]).to(self.device)
        
        return train_input, train_target, val_input, val_target, test_input, test_target
    
    def train(self, 
              train_input: torch.Tensor,
              train_target: torch.Tensor,
              val_input: torch.Tensor,
              val_target: torch.Tensor,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 10) -> Dict[str, List[float]]:
        """
        모델 훈련
        
        Args:
            train_input: 훈련 입력 데이터
            train_target: 훈련 타겟 데이터
            val_input: 검증 입력 데이터
            val_target: 검증 타겟 데이터
            epochs: 훈련 에포크 수
            batch_size: 배치 크기
            patience: 조기 종료 인내심
            
        Returns:
            훈련 히스토리
        """
        # 모델 초기화
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 훈련 히스토리
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("모델 훈련 시작...")
        
        for epoch in range(epochs):
            # 훈련 모드
            self.model.train()
            train_loss = 0.0
            
            # 배치별 훈련
            for i in range(0, len(train_input), batch_size):
                batch_input = train_input[i:i+batch_size]
                batch_target = train_target[i:i+batch_size]
                
                # 순전파
                self.optimizer.zero_grad()
                outputs = self.model(batch_input)
                loss = self.criterion(outputs, batch_target)
                
                # 역전파
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 검증
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_input)
                val_loss = self.criterion(val_outputs, val_target).item()
            
            # 히스토리 저장
            avg_train_loss = train_loss / (len(train_input) // batch_size + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"조기 종료: {epoch+1} 에포크에서 훈련 중단")
                    break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("훈련 완료!")
        
        return history
    
    def predict(self, input_data: torch.Tensor) -> np.ndarray:
        """
        예측 수행
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            예측 결과
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_data)
            # 역정규화
            predictions_np = predictions.cpu().numpy()
            predictions_denorm = self.target_scaler.inverse_transform(predictions_np.reshape(-1, predictions_np.shape[-1]))
            return predictions_denorm.reshape(predictions_np.shape)
    
    def evaluate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            test_input: 테스트 입력 데이터
            test_target: 테스트 타겟 데이터
            
        Returns:
            평가 지표들
        """
        predictions = self.predict(test_input)
        test_target_np = test_target.cpu().numpy()
        test_target_denorm = self.target_scaler.inverse_transform(test_target_np)
        
        # MSE
        mse = np.mean((predictions - test_target_denorm) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - test_target_denorm))
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((test_target_denorm - predictions) / test_target_denorm)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def forecast_future(self, 
                       last_sequence: np.ndarray,
                       forecast_steps: int,
                       target_columns: List[str]) -> pd.DataFrame:
        """
        미래 예측
        
        Args:
            last_sequence: 마지막 시퀀스 데이터
            forecast_steps: 예측할 스텝 수
            target_columns: 타겟 컬럼명들
            
        Returns:
            예측 결과 데이터프레임
        """
        self.model.eval()
        
        # 입력 데이터 준비
        input_scaled = self.scaler.transform(
            last_sequence.reshape(-1, last_sequence.shape[-1])
        ).reshape(last_sequence.shape)
        
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(forecast_steps):
                # 예측
                output = self.model(input_tensor)
                predictions.append(output.cpu().numpy())
                
                # 다음 입력을 위해 시퀀스 업데이트
                new_input = input_tensor.clone()
                new_input[0, :-1, :] = new_input[0, 1:, :]
                new_input[0, -1, :] = output[0, 0, :]
                input_tensor = new_input
        
        # 예측 결과 역정규화
        predictions_array = np.concatenate(predictions, axis=1)
        predictions_denorm = self.target_scaler.inverse_transform(
            predictions_array.reshape(-1, predictions_array.shape[-1])
        )
        
        # 데이터프레임으로 변환
        forecast_df = pd.DataFrame(predictions_denorm, columns=target_columns)
        
        return forecast_df
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler = checkpoint['scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        print(f"모델이 {filepath}에서 로드되었습니다.")

if __name__ == "__main__":
    # 테스트 코드
    print("TimesFM 모델 래퍼 테스트")
    
    # 샘플 데이터 생성
    input_size = 10
    seq_length = 12
    num_samples = 100
    
    # 랜덤 데이터 생성
    np.random.seed(42)
    input_data = np.random.randn(num_samples, seq_length, input_size)
    target_data = np.random.randn(num_samples, input_size)
    
    # 모델 초기화
    model = TimesFMWrapper(input_size=input_size)
    
    # 데이터 준비
    train_input, train_target, val_input, val_target, test_input, test_target = \
        model.prepare_data(input_data, target_data)
    
    # 모델 훈련
    history = model.train(train_input, train_target, val_input, val_target, epochs=50)
    
    # 모델 평가
    metrics = model.evaluate(test_input, test_target)
    print("평가 결과:", metrics) 