import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd
from scipy.fftpack import dct, idct

# Hankel 행렬 생성 함수
def hankel_matrix(x, L):
    """
    1차원 배열 x (길이 N)로부터 윈도우 길이 L인 Hankel 행렬 생성
    행의 개수 L, 열의 개수 K = N - L + 1
    """
    N = len(x)
    K = N - L + 1
    H = np.empty((L, K))
    for i in range(L):
        H[i, :] = x[i:i+K]
    return H

# 대각 평균 (diagonal averaging)으로 1차원 시퀀스 복원 함수
def diagonal_averaging(H):
    """
    Hankel 행렬 H (예: rank-1 근사 행렬 H1)로부터
    대각 평균(diagonal averaging)하여 원래 1차원 시퀀스를 복원.
    복원된 시퀀스 길이 = (행 수 + 열 수 - 1)
    """
    L, K = H.shape
    N = L + K - 1
    recon = np.zeros(N)
    counts = np.zeros(N)
    # H의 (i,j) 원소는 복원된 시퀀스의 인덱스 i+j에 기여함
    for i in range(L):
        for j in range(K):
            recon[i+j] += H[i, j]
            counts[i+j] += 1
    return recon / counts

# SVD 기반 배경 제거 (SSA)
def remove_background_ssa(y, L=None):
    """
    1차원 스펙트럼 y에 대해 Hankel 행렬을 구성하고,
    SVD를 수행하여 rank-1 근사를 배경으로 복원한 후,
    residual = y - background 를 반환합니다.
    L: Hankel 행렬의 행의 수 (생략하면 N/2 사용)
    """
    N = len(y)
    if L is None:
        L = N // 2  # 기본값: N/2
    # Hankel 행렬 생성
    H = hankel_matrix(y, L)
    # SVD 수행
    U, s, Vt = svd(H, full_matrices=False)
    # rank-1 근사: H1 = s[0] * outer(U[:,0], Vt[0,:])
    H1 = s[0] * np.outer(U[:, 0], Vt[0, :])
    # 대각 평균으로 배경 복원
    background = diagonal_averaging(H1)
    # 만약 복원된 background 길이가 원래 y와 다르면 보간
    if len(background) != N:
        f_interp = interp1d(np.linspace(0, 1, len(background)), background, kind='linear', fill_value="extrapolate")
        background = f_interp(np.linspace(0, 1, N))
    # residual 계산
    residual = y - background
    return background, residual

def main():
    # 파일 경로 설정
    file_path = "/home/jyounglee/NL_real/data/predict/3siwafer_#1_0.5M_0.txt"
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다:", file_path)
        return

    # txt 파일 불러오기 (delimiter가 탭이나 공백일 수 있으므로 필요에 따라 조정)
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print("파일 로드 중 오류:", e)
        return

    # data가 2열 이상이어야 함 (첫 번째 열: wave, 두 번째 열: intensity)
    if data.ndim != 2 or data.shape[1] < 2:
        print("데이터 형식이 올바르지 않습니다. 데이터 shape:", data.shape)
        return

    # x: 파장 혹은 x축, y: intensity
    x = data[:, 0]
    y = data[:, 1]

    # (선택사항) 만약 보간이 필요하면, 예를 들어 x축이 일정하지 않은 경우
    # 여기서는 x의 최소~최대 범위에서 1600 포인트로 보간
    row_new_length = 1600
    x_new = np.linspace(x.min(), x.max(), row_new_length)
    f_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    y_interp = f_interp(x_new)

    # 1. 원본 (보간된) 스펙트럼 시각화
    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.plot(x_new, y_interp, 'b-')
    plt.title("Original (Interpolated) Spectrum")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.grid(True)

    # 2. SVD(SSA) 배경 제거
    # Hankel 행렬 윈도우 길이 L은 데이터 길이에 따라 조정 (여기서는 기본값 사용)
    background, residual = remove_background_ssa(y_interp)
    
    plt.subplot(2,2,2)
    plt.plot(x_new, background, 'g-')
    plt.title("Extracted Background (Rank-1 SSA)")
    plt.xlabel("x")
    plt.ylabel("Background")
    plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(x_new, residual, 'r-')
    plt.title("Residual (Original - Background)")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.grid(True)

    # 3. DCT 변환: residual에 대해 DCT 수행 (type-II, norm='ortho')
    residual_dct = dct(residual, norm='ortho')
    plt.subplot(2,2,4)
    plt.plot(residual_dct, 'm.-')
    plt.title("DCT of Residual Signal")
    plt.xlabel("Index")
    plt.ylabel("DCT Coefficient")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
