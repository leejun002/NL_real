import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd
from scipy.signal import butter, filtfilt
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
    텍스트 파일들을 읽어서,
      - x 값이 zero_cut 이상인 데이터만 사용하고,
      - 해당 영역을 row_new_length로 cubic 보간한 후,
      - 여러 파일의 결과를 모아서 (row_new_length, num_spectra) 배열을 만들고,
      - 각 행에 대해 선형 보간하여 최종 데이터 shape를 (row_new_length, col_new_length)로 만듭니다.
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
        
        if arr.shape[0] < 2 or arr.shape[1] < 2:
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
        
        # row_new_length로 cubic 보간
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
    
    # 각 스펙트럼을 열로 두어 (row_new_length, num_spectra) 배열 생성
    all_spectra = np.array(spectra_list).T
    print(f"After row interpolation: shape = {all_spectra.shape}")
    
    # 각 행에 대해 열 보간: 최종적으로 (row_new_length, col_new_length) 배열 생성
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

def high_pass_filter(data, cutoff=0.1, fs=1.0, order=4):
    """
    Butterworth high-pass 필터를 사용하여 2D 데이터의 각 열(스펙트럼)에 대해
    저주파 성분(배경)을 제거합니다.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    filtered_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        filtered_data[:, col] = filtfilt(b, a, data[:, col])
    return filtered_data

def remove_background_svd(data, remove_rank=1):
    """
    SVD를 통해 데이터의 배경(주로 rank-1 성분)을 제거합니다.
    """
    U, s, Vt = svd(data, full_matrices=False)
    # 첫 번째 remove_rank 성분을 배경으로 간주하고 제거
    background = np.zeros_like(data)
    for i in range(remove_rank):
        background += np.outer(U[:, i], s[i] * Vt[i, :])
    residual = data - background
    return residual, U, s, Vt

def apply_dct(data):
    """
    각 열(스펙트럼)에 대해 DCT(type-II, 'ortho' 정규화)를 적용하여
    주파수 도메인 표현을 반환합니다.
    """
    dct_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        dct_data[:, col] = dct(data[:, col], type=2, norm='ortho')
    return dct_data

def visualize_sample(data, sample_col, stage_name, output_dir):
    """
    주어진 2D 데이터에서 특정 sample (열)을 추출하여 시각화하고,
    output_dir에 이미지 파일로 저장합니다.
    """
    sample = data[:, sample_col]
    plt.figure(figsize=(8, 4))
    plt.plot(sample, 'b.-')
    plt.title(f"{stage_name} - Sample (Col {sample_col})")
    plt.xlabel("Row index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    filename = os.path.join(output_dir, f"sample_{stage_name.replace(' ', '_').lower()}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved visualization: {filename}")

def pipeline_and_save_mat(
    base_folder="/home/jyounglee/NL_real/data/noise_3siwafer",
    output_mat_path="/home/jyounglee/NL_real/noise_data_processing/noise_dp8/result_ihsdt/noise_data.mat",
    subfolders=("1", "2", "3"),
    zero_cut=0.0,
    row_new_length=1600,
    col_new_length=640,
    hp_cutoff=0.1,  # 고주파 필터 컷오프 주파수 (Hz)
    fs=1.0,         # 보간 후 샘플링 주파수 (Hz)
    filter_order=4,
    remove_rank=1,  # 제거할 rank 수 (배경 제거)
    sample_col=0    # 시각화할 샘플의 열 인덱스
):
    """
    1. 텍스트 파일로부터 데이터를 로드하고 보간 수행  
    2. 고주파 필터링 적용  
    3. SVD를 통해 배경(주로 rank-1 성분) 제거  
    4. residual 신호에 대해 각 열별 DCT 적용  
    5. 최종 DCT 결과를 .mat 파일로 저장 (transpose하여 640x1600 shape)  
    6. 각 단계에서 선택한 sample을 시각화
    """
    output_dir = os.path.dirname(output_mat_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # (1) 데이터 로드 및 보간
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
    visualize_sample(data_interp, sample_col, "Interpolated Data", output_dir)
    
    # (2) 고주파 필터링 적용
    data_hp = high_pass_filter(data_interp, cutoff=hp_cutoff, fs=fs, order=filter_order)
    print("고주파 필터링 완료.")
    visualize_sample(data_hp, sample_col, "High-Pass Filtered Data", output_dir)
    
    # (3) SVD를 통한 배경 제거 (예: rank-1 제거)
    residual, U, s, Vt = remove_background_svd(data_hp, remove_rank=remove_rank)
    print("SVD를 통한 배경 제거 완료.")
    visualize_sample(residual, sample_col, "Residual after SVD Background Removal", output_dir)
    
    # (4) residual 신호에 대해 각 열별 DCT 적용
    dct_result = apply_dct(residual)
    print("DCT 적용 완료.")
    visualize_sample(dct_result, sample_col, "DCT of Residual Signal", output_dir)
    
    # (5) 결과 저장 (.mat 파일)
    # dct_result의 원래 shape는 (row_new_length, col_new_length) = (1600, 640)
    # 원하는 저장 shape는 (640, 1600)이므로 transpose합니다.
    mat_dict = {
        "noise": dct_result.T
    }
    savemat(output_mat_path, mat_dict)
    print(f".mat 파일 저장 완료: {output_mat_path}")

if __name__ == "__main__":
    pipeline_and_save_mat(
        base_folder="/home/jyounglee/NL_real/data/noise_3siwafer",
        output_mat_path="/home/jyounglee/NL_real/noise_data_processing/noise_dp8/result_ihsdt/noise_data.mat",
        subfolders=("1", "2", "3"),
        zero_cut=0.0,
        row_new_length=1600,
        col_new_length=640,
        hp_cutoff=0.1,  # 필요에 따라 조정
        fs=1.0,         # 보간 후 데이터의 샘플링 주파수
        filter_order=4,
        remove_rank=1,
        sample_col=0   # 시각화할 sample의 열 인덱스
    )
