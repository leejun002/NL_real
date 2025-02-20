import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd  # (참고: 표준 SVD도 사용할 수 있으나, 아래에서는 randomized_svd를 사용)
from scipy.fftpack import dct, idct
from sklearn.utils.extmath import randomized_svd

def load_and_interpolate(file_path, num_points=1600):
    """
    지정된 텍스트 파일을 로드하여, 첫 번째 열(x)과 두 번째 열(y)를 추출한 후,
    x의 최소~최대 범위에서 num_points 개의 포인트로 cubic 보간한 결과(x_new, y_interp)를 반환합니다.
    """
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"파일 로드 중 오류: {e}")
        return None, None
    
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"데이터 형식이 올바르지 않습니다. 데이터 shape: {data.shape}")
        return None, None

    x = data[:, 0]
    y = data[:, 1]
    x_new = np.linspace(x.min(), x.max(), num_points)
    f_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    y_interp = f_interp(x_new)
    return x_new, y_interp

def hankel_matrix(x, L):
    """
    1차원 배열 x (길이 N)로부터 윈도우 길이 L인 Hankel 행렬 생성.
    (행 수 L, 열 수 K = N - L + 1)
    """
    N = len(x)
    K = N - L + 1
    H = np.empty((L, K))
    for i in range(L):
        H[i, :] = x[i:i+K]
    return H

def diagonal_averaging(H):
    """
    Hankel 행렬 H로부터 각 대각선의 평균을 계산하여 1차원 시퀀스로 복원합니다.
    복원된 시퀀스 길이는 (행 수 + 열 수 - 1)입니다.
    """
    L, K = H.shape
    N = L + K - 1
    recon = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            recon[i+j] += H[i, j]
            counts[i+j] += 1
    return recon / counts

def randomized_svd_decompose(signal, L=None, n_components=5):
    """
    1차원 신호(signal)에 대해 Hankel 행렬을 구성한 후 randomized SVD를 수행합니다.
    
    Parameters:
      signal      : 1차원 입력 신호 (예: 보간된 스펙트럼)
      L           : Hankel 행렬의 행 수 (None이면 기본값: N//2)
      n_components: 추출할 singular component 개수
      
    Returns:
      H, U, s, Vt, L
    """
    N = len(signal)
    if L is None:
        L = N // 2
    H = hankel_matrix(signal, L)
    # randomized SVD를 이용하여 n_components 만큼의 singular component를 추출합니다.
    U, s, Vt = randomized_svd(H, n_components=n_components, n_iter=5, random_state=42)
    return H, U, s, Vt, L

def reconstruct_component(U, s, Vt, idx, N, L):
    """
    randomized SVD 결과에서 idx번째 singular component를 추출하여,
    대각 평균(diagonal averaging)을 통해 1차원 신호로 복원합니다.
    
    Parameters:
      U, s, Vt : SVD 결과
      idx      : 추출할 singular component의 인덱스 (0부터 시작)
      N        : 원래 신호의 길이
      L        : Hankel 행렬의 행 수
      
    Returns:
      comp     : 복원된 1차원 성분 (길이 N)
    """
    H_comp = s[idx] * np.outer(U[:, idx], Vt[idx, :])
    comp = diagonal_averaging(H_comp)
    # 원래 신호 길이와 다르면 선형 보간하여 길이를 맞춤
    if len(comp) != N:
        f_interp = interp1d(np.linspace(0, 1, len(comp)), comp, kind='linear', fill_value="extrapolate")
        comp = f_interp(np.linspace(0, 1, N))
    return comp

def plot_randomized_svd(file_path, num_points=1600, n_components=5, L=None):
    """
    파일을 로드하고 보간한 후, Hankel 행렬 기반 randomized SVD 분해를 수행하여
    각 singular component와 누적 재구성을 시각화합니다.
    
    결과는 1열의 subplot으로 표시됩니다.
    """
    x_new, y_interp = load_and_interpolate(file_path, num_points=num_points)
    if x_new is None or y_interp is None:
        return
    N = len(y_interp)
    
    # randomized SVD 분해
    H, U, s, Vt, L = randomized_svd_decompose(y_interp, L=L, n_components=n_components)
    
    total_plots = n_components + 2  # 원본 스펙트럼, singular 값 플롯, 그리고 각 구성요소 플롯
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
    # 원본 보간 스펙트럼
    axes[0].plot(x_new, y_interp, 'k-', label="Original")
    axes[0].set_title("Original (Interpolated) Spectrum")
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    axes[0].grid(True)
    
    # Singular 값 플롯
    axes[1].plot(s, 'bo-', label="Singular Values (Randomized SVD)")
    axes[1].set_title("Singular Values")
    axes[1].set_ylabel("Singular Value")
    axes[1].legend()
    axes[1].grid(True)
    
    cumulative = np.zeros(N)
    for i in range(n_components):
        comp = reconstruct_component(U, s, Vt, i, N, L)
        cumulative += comp
        axes[i+2].plot(x_new, comp, label=f"Component {i+1}")
        axes[i+2].plot(x_new, cumulative, 'r--', label=f"Cumulative 1-{i+1}")
        axes[i+2].set_title(f"Component {i+1} and Cumulative Reconstruction")
        axes[i+2].set_ylabel("Amplitude")
        axes[i+2].legend()
        axes[i+2].grid(True)
    
    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.show()

def main():
    file_path = "/home/jyounglee/NL_real/data/predict/3siwafer_#1_0.5M_0.txt"
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다:", file_path)
        return
    plot_randomized_svd(file_path, num_points=1600, n_components=5, L=None)

if __name__ == "__main__":
    main()
