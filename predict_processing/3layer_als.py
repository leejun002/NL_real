import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import dct, idct

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing을 이용한 baseline 추정 함수.
    
    Parameters:
      y     : 1D numpy array (스펙트럼 intensity)
      lam   : 매끄러움(smoothness) 제어 파라미터 (크면 baseline이 더 부드러워짐)
      p     : 비대칭 가중치 (보통 0.001~0.1 사이, 여기서는 0.01 사용)
      niter : 반복 횟수
    
    Returns:
      z     : 추정된 baseline (1D array)
    """
    L = len(y)
    # 2차 차분 행렬 계산 (np.diff를 두 번 적용하는 것과 동일)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + D)
        z = Z.dot(w * y)
        # y > z 인 경우 가중치 p, y < z 인 경우 (1-p)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def main():
    # 파일 경로 설정
    file_path = "/home/jyounglee/NL/data/predict/3siwafer_#1_0.5M_0.txt"
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다:", file_path)
        return

    # 텍스트 파일 로드 (탭 혹은 공백 구분자로 되어 있다고 가정)
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print("파일 로드 중 오류:", e)
        return

    # 데이터는 최소 2열이어야 함 (첫 번째 열: x축, 두 번째 열: intensity)
    if data.ndim != 2 or data.shape[1] < 2:
        print("데이터 형식이 올바르지 않습니다. 데이터 shape:", data.shape)
        return

    # x: 파장 혹은 x축, y: intensity
    x = data[:, 0]
    y = data[:, 1]

    # 1. 보간: x의 최소 ~ 최대 범위에서 1600 포인트로 보간 (예: 스펙트럼 해상도 맞춤)
    row_new_length = 1600
    x_new = np.linspace(x.min(), x.max(), row_new_length)
    f_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    y_interp = f_interp(x_new)

    # 2. ALS 기반 baseline 추정 및 제거
    baseline = baseline_als(y_interp, lam=1e5, p=0.01, niter=10)
    residual = y_interp - baseline

    # 3. residual 신호에 대해 DCT 변환 (type-II, 'ortho')
    residual_dct = dct(residual, norm='ortho')

    # 4. 각 단계별 결과 시각화 (subplots)
    plt.figure(figsize=(12, 10))

    # 원본 보간 스펙트럼
    plt.subplot(2, 2, 1)
    plt.plot(x_new, y_interp, 'b-', label="Interpolated Spectrum")
    plt.title("Original (Interpolated) Spectrum")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    # 추정된 baseline
    plt.subplot(2, 2, 2)
    plt.plot(x_new, baseline, 'g-', label="Estimated Baseline (ALS)")
    plt.title("Estimated Baseline")
    plt.xlabel("x")
    plt.ylabel("Baseline")
    plt.legend()
    plt.grid(True)

    # Residual 신호 (원본 - baseline)
    plt.subplot(2, 2, 3)
    plt.plot(x_new, residual, 'r-', label="Residual Spectrum")
    plt.title("Residual (Spectrum - Baseline)")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True)

    # DCT 변환 결과
    plt.subplot(2, 2, 4)
    plt.plot(residual_dct, 'm.-', label="DCT Coefficients")
    plt.title("DCT of Residual Signal")
    plt.xlabel("Index")
    plt.ylabel("DCT Coefficient")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
