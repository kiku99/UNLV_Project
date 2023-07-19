import numpy as np
from sklearn.linear_model import LinearRegression
import time

# 데이터 생성
X = np.random.rand(10000, 11)  # 10000개의 샘플과 11개의 특성으로 구성된 입력 데이터
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(10000)  # 랜덤 노이즈를 추가한 선형 관계

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 훈련
start_time = time.time()
model.fit(X, y)
end_time = time.time()

# 테스트 데이터 생성
X_test = np.random.rand(10, 11)  # 새로운 테스트 데이터 생성

# 예측
y_pred = model.predict(X_test)  # 테스트 데이터에 대한 예측값 계산
print(f"Multiprocessing Time: {(end_time - start_time) / 60:.3f}min {(end_time - start_time) % 60:.3f}sec")
print("Predictions:", y_pred)
