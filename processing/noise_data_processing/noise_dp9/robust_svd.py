import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# 데이터 로드 및 보간 함수
# -----------------------------
def load_and_interpolate_all(base_dir, num_points=1600):
    """
    base_dir 및 그 하위 디렉토리 내의 모든 txt 파일을 읽어,
    각 파일의 두 번째 열(스펙트럼 intensity)을 x의 최소~최대 범위에서 
    num_points 포인트로 cubic 보간한 후, 
    (n_files x num_points) 크기의 2D 배열과 보간된 x축(x_new)을 반환합니다.
    """
    # 재귀적으로 모든 txt 파일 검색 (하위 디렉토리 포함)
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"'{base_dir}' 내에 txt 파일이 없습니다.")
        return None, None, None

    spectra = []
    file_paths = []  # 유효하게 로드한 파일 경로 저장
    x_new_global = None

    for f in file_list:
        try:
            data = np.loadtxt(f)
        except Exception as e:
            print(f"파일 {f} 로드 중 오류: {e}")
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"[경고] {f} 파일의 데이터 형식이 올바르지 않습니다. (shape: {data.shape})")
            continue
        x = data[:, 0]
        y = data[:, 1]
        if x_new_global is None:
            x_new_global = np.linspace(x.min(), x.max(), num_points)
        # cubic 보간
        f_interp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        y_interp = f_interp(x_new_global)
        spectra.append(y_interp)
        file_paths.append(f)
    if len(spectra) == 0:
        print("유효한 스펙트럼 데이터가 없습니다.")
        return None, None, None

    M = np.array(spectra)  # shape: (n_files, num_points)
    return M, x_new_global, file_paths

# -----------------------------
# Robust PCA (IALM 방식)
# -----------------------------
def rpca(M, lambda_val=None, tol=1e-7, max_iter=1000):
    """
    Inexact Augmented Lagrange Multiplier (IALM) 방식을 이용하여
    Robust PCA를 수행합니다.
    
    Parameters
    ----------
    M : numpy.ndarray
        입력 행렬 (n_files x num_points)
    lambda_val : float, optional
        희소성 제어 파라미터 (기본값: 1 / sqrt(max(m, n)))
    tol : float, optional
        수렴 종료 tolerance (기본값 1e-7)
    max_iter : int, optional
        최대 반복 횟수 (기본값 1000)
        
    Returns
    -------
    L : numpy.ndarray
        저순위 (low-rank) 행렬 (배경)
    S : numpy.ndarray
        희소 (sparse) 행렬 (잔차)
    """
    m, n = M.shape
    norm_M = np.linalg.norm(M, ord='fro')
    if lambda_val is None:
        lambda_val = 1 / np.sqrt(max(m, n))
    # 초기화
    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf)/lambda_val)
    mu = 1.25 / np.linalg.norm(M, 2)
    mu_bar = mu * 1e7
    rho = 1.5
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    iter_num = 0
    while iter_num < max_iter:
        iter_num += 1
        # L 업데이트: SVD 후 soft-thresholding
        U, sigma, Vt = np.linalg.svd(M - S + (1/mu)*Y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        rank = (sigma_thresh > 0).sum()
        L = (U[:, :rank] * sigma_thresh[:rank]) @ Vt[:rank, :]
        # S 업데이트: soft-thresholding
        temp = M - L + (1/mu)*Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lambda_val/mu, 0)
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu*rho, mu_bar)
        err = np.linalg.norm(Z, ord='fro') / norm_M
        if err < tol:
            break
    print(f"RPCA converged in {iter_num} iterations with error {err:.2e}")
    return L, S

# -----------------------------
# Robust PCA 배경제거 및 시각화
# -----------------------------
def robust_pca_background_removal_and_visualize(base_dir, num_points=1600):
    """
    base_dir 폴더(및 하위 폴더) 내의 모든 txt 파일을 보간하여 행렬 M으로 구성한 후,
    Robust PCA (IALM)를 통해 저순위 성분(L, 배경)과 희소 성분(S, 잔차)를 분리합니다.
    각 파일별로 원본, 배경, 잔차를 시각화합니다.
    """
    M, x_new, file_paths = load_and_interpolate_all(base_dir, num_points=num_points)
    if M is None or x_new is None:
        return
    n_files, spec_len = M.shape
    print(f"총 {n_files}개의 스펙트럼 로드 (보간 후 길이: {spec_len})")
    
    # RPCA 수행
    L, S = rpca(M, tol=1e-7, max_iter=1000)
    
    # 각 파일별 시각화
    for i in range(n_files):
        original = M[i, :]
        background = L[i, :]
        residual = S[i, :]
        fname = os.path.basename(file_paths[i])
        
        plt.figure(figsize=(10, 6))
        plt.suptitle(f"File: {fname}", fontsize=14)
        
        # 원본과 배경 비교
        plt.subplot(2, 1, 1)
        plt.plot(x_new, original, 'b-', label="Original")
        plt.plot(x_new, background, 'g--', label="Background (L)")
        plt.title("Original Spectrum & Extracted Background")
        plt.xlabel("x")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        
        # 잔차(Original - Background)와 희소 성분 S 비교
        plt.subplot(2, 1, 2)
        plt.plot(x_new, original - background, 'r-', label="Residual (Original - Background)")
        plt.plot(x_new, residual, 'k--', label="Sparse Component (S)")
        plt.title("Residual vs. Sparse Component")
        plt.xlabel("x")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
def main():
    base_dir = "/home/jyounglee/NL_real/data/noise_3siwafer"
    # base_dir = "/home/jyounglee/NL_real/data/3layer_on_siwafer"
    if not os.path.exists(base_dir):
        print("경로가 존재하지 않습니다:", base_dir)
        return
    robust_pca_background_removal_and_visualize(base_dir, num_points=1600)

if __name__ == "__main__":
    main()
