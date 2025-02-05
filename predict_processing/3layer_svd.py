import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd

def load_and_interpolate(file_path, num_points=1600):
    """
    파일을 로드한 후, 첫 번째 열은 x, 두 번째 열은 y로 가정하고,
    x의 최소~최대 범위에서 num_points 개의 포인트로 cubic 보간한 스펙트럼을 반환합니다.
    """
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"파일 로드 중 오류: {e}")
        return None, None

    if data.ndim != 2 or data.shape[1] < 2:
        print(f"데이터 형식이 올바르지 않습니다. shape: {data.shape}")
        return None, None

    x = data[:, 0]
    y = data[:, 1]

    # 보간: x의 최소~최대 범위에서 num_points 개의 포인트로 보간
    x_new = np.linspace(x.min(), x.max(), num_points)
    f_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    y_interp = f_interp(x_new)
    return x_new, y_interp

def hankel_matrix(x, L):
    """
    1차원 배열 x (길이 N)로부터 윈도우 길이 L인 Hankel 행렬 생성
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
    Hankel 행렬 H로부터 대각 평균(diagonal averaging)하여 1차원 시퀀스를 복원합니다.
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

def svd_decomposition(signal, L=None, n_components=5):
    """
    1차원 신호에 대해 Hankel 행렬을 구성한 후 SVD를 수행하고,
    상위 n_components 개의 rank-1 성분을 대각 평균을 통해 복원합니다.
    
    Parameters:
      signal      : 1차원 입력 신호 (예: 보간된 스펙트럼)
      L           : Hankel 행렬의 행 수 (None이면 기본값: N//2)
      n_components: 복원할 성분 개수
      
    Returns:
      components  : 각 rank-1 성분을 1차원 신호로 복원한 리스트 (길이 n_components)
      singular_vals: 전체 singular value 배열
    """
    N = len(signal)
    if L is None:
        L = N // 2  # 기본값
    # Hankel 행렬 생성
    H = hankel_matrix(signal, L)
    # SVD 수행
    U, s, Vt = svd(H, full_matrices=False)
    components = []
    # 상위 n_components 개의 rank-1 근사 성분 추출
    for i in range(n_components):
        H_i = s[i] * np.outer(U[:, i], Vt[i, :])
        comp_i = diagonal_averaging(H_i)
        # 원래 신호 길이와 다르면 보간으로 맞춤 (보통 H에서 나온 길이는 L + K - 1 = N)
        if len(comp_i) != N:
            f_interp = interp1d(np.linspace(0, 1, len(comp_i)), comp_i, kind='linear', fill_value="extrapolate")
            comp_i = f_interp(np.linspace(0, 1, N))
        components.append(comp_i)
    return components, s

def main():
    # 파일 경로 설정
    file_path = "/home/jyounglee/NL/data/predict/3siwafer_#1_0.5M_0.txt"
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다:", file_path)
        return

    # 1. 파일 로드 및 보간
    x_new, y_interp = load_and_interpolate(file_path, num_points=1600)
    if x_new is None or y_interp is None:
        return

    # 2. SVD 분해: Hankel 행렬로부터 상위 n_components 성분을 추출
    n_components = 5  # 원하는 성분 수
    components, singular_vals = svd_decomposition(y_interp, L=None, n_components=n_components)

    # 3. 원본 스펙트럼과 각 성분, singular values 시각화
    plt.figure(figsize=(14, 10))

    # 원본 보간 스펙트럼
    plt.subplot(3, 1, 1)
    plt.plot(x_new, y_interp, 'k-', label="Original Interpolated Spectrum")
    plt.title("Original (Interpolated) Spectrum")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    # 각 rank-1 성분
    plt.subplot(3, 1, 2)
    for i, comp in enumerate(components):
        plt.plot(x_new, comp, label=f"Rank-{i+1} Component")
    plt.title("Extracted Rank-1 Components via SVD")
    plt.xlabel("x")
    plt.ylabel("Component Amplitude")
    plt.legend()
    plt.grid(True)

    # singular value plot
    plt.subplot(3, 1, 3)
    plt.plot(singular_vals, 'bo-')
    plt.title("Singular Values")
    plt.xlabel("Component Index")
    plt.ylabel("Singular Value")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
