import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat
from scipy.fftpack import dct
import math

def load_and_interpolate_all(base_dir, num_points=1600):
    """
    base_dir 및 그 하위 디렉토리 내의 모든 txt 파일을 읽어,
    각 파일의 두 번째 열(스펙트럼 intensity)을 x의 최소~최대 범위에서 
    num_points 포인트로 cubic 보간한 후, 
    (n_files x num_points) 크기의 2D 배열(M)과 보간된 x축(x_new), 
    그리고 파일 경로 리스트를 반환합니다.
    """
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"'{base_dir}' 내에 txt 파일이 없습니다.")
        return None, None, None

    spectra = []
    file_paths = []
    x_new_global = None

    for f in file_list:
        try:
            data = np.loadtxt(f)
        except Exception as e:
            print(f"[로드 오류] {f} : {e}")
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"[경고] {f} : (N,2) 형태가 아님 (shape={data.shape}). 스킵.")
            continue

        x = data[:, 0]
        y = data[:, 1]
        if x_new_global is None:
            x_min, x_max = x.min(), x.max()
            x_new_global = np.linspace(x_min, x_max, num_points)
        # cubic 보간
        f_intp = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        y_interp = f_intp(x_new_global)

        spectra.append(y_interp)
        file_paths.append(f)

    if len(spectra) == 0:
        print("유효한 스펙트럼 데이터가 없습니다.")
        return None, None, None

    M = np.array(spectra)  # (n_files, 1600)
    return M, x_new_global, file_paths

def rpca(M, lambda_val=None, tol=1e-7, max_iter=1000):
    """
    Inexact Augmented Lagrange Multiplier (IALM) 방식을 이용하여 Robust PCA 수행.
    M = L + S
    - L: 저순위 (low-rank)
    - S: 희소 (sparse)
    """
    m, n = M.shape
    norm_M = np.linalg.norm(M, ord='fro')
    if lambda_val is None:
        lambda_val = 1 / math.sqrt(max(m, n))

    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf)/lambda_val)
    mu = 1.25 / np.linalg.norm(M, 2)
    mu_bar = mu * 1e7
    rho = 1.5

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    iter_num = 0

    while iter_num < max_iter:
        iter_num += 1
        # L 업데이트 (SVD 후 soft-thresholding)
        U, sigma, Vt = np.linalg.svd(M - S + (1/mu)*Y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        rank = np.sum(sigma_thresh > 0)
        L = (U[:, :rank] * sigma_thresh[:rank]) @ Vt[:rank, :]
        # S 업데이트 (soft-thresholding)
        temp = M - L + (1/mu)*Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - (lambda_val/mu), 0)
        # 잔차
        Z = M - L - S
        Y = Y + mu*Z
        mu = min(mu*rho, mu_bar)
        err = np.linalg.norm(Z, 'fro') / norm_M
        if err < tol:
            print(f"[RPCA] {iter_num} iters, err={err:.2e}")
            break

    print(f"[RPCA] 최종 iter={iter_num}, err={err:.2e}")
    return L, S

def robust_pca_to_mat(base_dir, row_points=1600, col_points=640, out_mat="noise_3siwafer_640x1600.mat"):
    """
    1) base_dir 하위 모든 txt 파일(하위 폴더 포함) -> (n_files, row_points) 형식으로 로드 & 보간
    2) RPCA -> M = L + S에서 S(잔차)를 추출
    3) 열 방향 보간 -> col_points(기본 640)까지 맞춤 -> (row_points, col_points)
    4) DCT 변환 (행 방향 기준) -> (row_points, col_points)
    5) 필요하다면 전치 -> (col_points, row_points) => (640, 1600)
    6) .mat 저장 (key="noise")
    """
    M, x_new, file_paths = load_and_interpolate_all(base_dir, num_points=row_points)
    if M is None:
        print("로딩 실패. 종료.")
        return

    # Robust PCA
    print("[정보] RPCA 수행 중 ...")
    L, S = rpca(M, tol=1e-7, max_iter=1000)
    # S가 희소 성분 = 잡음 + 신호 잔차

    # 열 보간 (현재 S.shape = (n_files, row_points))
    # n_files -> col_points 로 보간
    n_files, row_len = S.shape  # (n_files, row_points) = (?, 1600)
    # n_files 축 = "column index"로 볼 수 있음
    col_idx = np.arange(n_files)  # 0..(n_files-1)
    col_new = np.linspace(0, n_files-1, col_points)
    # 보간 후 (row_points, col_points)
    S2 = np.zeros((row_len, col_points), dtype=S.dtype)

    for i in range(row_len):
        f_col = interp1d(col_idx, S[:, i], kind='linear', fill_value="extrapolate")
        S2[i, :] = f_col(col_new)

    # DCT 변환 (행 방향)
    # shape: (row_len=1600, col_points=640)
    from scipy.fftpack import dct
    S2_dct = np.zeros_like(S2)
    for i in range(S2.shape[0]):
        S2_dct[i, :] = dct(S2[i, :], norm='ortho')

    # 최종 전치 -> (640, 1600)
    noise_data = S2_dct.T

    # .mat 저장
    import scipy.io as sio
    mat_dict = {"noise": noise_data}
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}' 저장 (shape={noise_data.shape})")

def main():
    base_dir = "/home/jyounglee/NL_real/data/given_data/noise_3siwafer"
    out_mat = "noise_3siwafer_640x1600.mat"
    robust_pca_to_mat(base_dir, row_points=1600, col_points=640, out_mat=out_mat)

if __name__ == "__main__":
    main()
