import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd
from scipy.fftpack import dct
from scipy.io import savemat

def load_training_data(
    base_folder="/home/jyounglee/NL_real/data/given_data/noise_3siwafer",
    subfolders=("1", "2", "3"),
    zero_cut=0.0,
    row_new_length=1600,   # 행 보간 후 길이
    col_new_length=640     # 열 보간 후 최종 스펙트럼 개수
):
    """
    base_folder의 하위 subfolders 내의 모든 txt 파일을 재귀적으로 읽어서,
    각 파일의 (x, y) 데이터를 로드한 후, x 값이 zero_cut 이상인 부분만 사용합니다.
    각 파일마다 x축 최소~최대 범위에서 row_new_length 포인트로 cubic 보간하고,
    모든 파일의 결과를 모아 (row_new_length, num_files) 배열을 생성합니다.
    이후 각 행(보간된 스펙트럼)에 대해 선형 보간을 적용하여 최종 데이터 shape를
    (row_new_length, col_new_length)로 만듭니다.
    """
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)
    
    if not txt_files:
        print("No txt files found in subfolders:", subfolders)
        return None
    
    txt_files = sorted(txt_files)
    spectra_list = []
    
    for fpath in txt_files:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue
        
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"Skipping {fpath}, shape={arr.shape}")
            continue
        
        x_raw, y_raw = arr[:, 0], arr[:, 1]
        # x 값이 zero_cut 이상인 부분만 선택
        mask = (x_raw >= zero_cut)
        x_cut = x_raw[mask]
        y_cut = y_raw[mask]
        if x_cut.size < 2:
            print(f"Skipping {fpath}: not enough data after zero_cut.")
            continue
        
        x_min, x_max = x_cut.min(), x_cut.max()
        if x_min == x_max:
            print(f"Skipping {fpath}: x range is zero.")
            continue
        
        # row_new_length 포인트로 cubic 보간
        x_new = np.linspace(x_min, x_max, row_new_length)
        try:
            f_intp = interp1d(x_cut, y_cut, kind='cubic')
            y_new = f_intp(x_new)
        except Exception as e:
            print(f"Interpolation error in {fpath}: {e}")
            continue
        
        spectra_list.append(y_new)
    
    if not spectra_list:
        print("No valid spectra after processing.")
        return None
    
    # 각 파일의 스펙트럼을 모아 (row_new_length, num_files) 배열 생성
    all_spectra = np.array(spectra_list).T
    print(f"After row interpolation: shape = {all_spectra.shape}")
    
    # 각 행(보간된 스펙트럼)에 대해 열 보간: 최종 shape = (row_new_length, col_new_length)
    num_rows = all_spectra.shape[0]
    num_cols = all_spectra.shape[1]
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_col = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_col(col_new)
    
    print(f"After column interpolation: shape = {all_spectra_2d.shape}")
    return all_spectra_2d

def pipeline_and_save_noise(
    base_folder="/home/jyounglee/NL_real/data/given_data/noise_3siwafer",
    output_mat="/home/jyounglee/NL_real/noise_data_processing/result/noise_data.mat",
    subfolders=("1", "2", "3"),
    zero_cut=0.0,
    row_new_length=1600,
    col_new_length=640,
    remove_svs=1
):
    # 1) Interpolation
    data_interp = load_training_data(
        base_folder=base_folder,
        subfolders=subfolders,
        zero_cut=zero_cut,
        row_new_length=row_new_length,
        col_new_length=col_new_length
    )
    if data_interp is None:
        print("데이터 로드 실패")
        return
    print("After interpolation: shape =", data_interp.shape)

    # 2) SVD BG Removal
    U, s, Vt = np.linalg.svd(data_interp, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] = 0.0
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD background removal:", data_bg_removed.shape)

    # 3) DCT per column
    dct_result = np.zeros_like(data_bg_removed)
    for j in range(data_bg_removed.shape[1]):
        dct_result[:, j] = dct(data_bg_removed[:, j], type=2, norm='ortho')
    print("After DCT:", dct_result.shape)

    # 4) Transpose
    noise_data = dct_result.T
    print("After transpose:", noise_data.shape)

    # 5) z-score
    train_mean = np.mean(noise_data)
    train_std  = np.std(noise_data)
    print("train_mean =", train_mean, ", train_std =", train_std)

    if train_std < 1e-6:
        noise_norm = noise_data - train_mean
    else:
        noise_norm = (noise_data - train_mean) / train_std

    print("Normalized: mean =", np.mean(noise_norm), ", std =", np.std(noise_norm))

    # **중요**: 결과 폴더 생성
    out_dir = os.path.dirname(output_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 6) Save .mat
    mat_dict = {
        "noise": noise_norm,
        "train_mean": np.array([train_mean]),
        "train_std": np.array([train_std])
    }
    savemat(output_mat, mat_dict)
    print("Saved .mat file:", output_mat)


if __name__ == "__main__":
    pipeline_and_save_noise(
        base_folder="/home/jyounglee/NL_real/data/given_data/noise_3siwafer",
        output_mat="/home/jyounglee/NL_real/noise_data_processing/result/noise_data.mat",
        subfolders=("1", "2", "3"),
        zero_cut=0.0,
        row_new_length=1600,
        col_new_length=640,
        remove_svs=1
    )