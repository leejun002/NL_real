import numpy as np
from scipy.interpolate import interp1d

def process_prediction_txt(txt_file, train_mean, train_std, output_txt):
    """
    예측할 txt 파일을 보간(1600개) + 정규화한 후 새 txt 파일로 저장하는 함수.

    Parameters:
        txt_file (str): 예측할 Raman 스펙트럼 데이터(txt 파일).
        train_mean (float): 학습 데이터의 평균값.
        train_std (float): 학습 데이터의 표준편차.
        output_txt (str): 저장할 정규화된 txt 파일 경로.
    """

    # 1. txt 파일 로드 (Raman shift 포함)
    data = np.loadtxt(txt_file)
    if data.shape[1] != 2:
        raise ValueError(f"txt 파일 형식이 맞지 않습니다. shape={data.shape}")

    raman_shift = data[:, 0]  # Raman shift 값 (X축)
    spectrum = data[:, 1]  # Intensity 값 (Y축)

    # 2. 1039개 데이터를 1600개로 보간 (1D interpolation)
    x_new = np.linspace(raman_shift.min(), raman_shift.max(), 1600)
    f_interp = interp1d(raman_shift, spectrum, kind='cubic')
    interpolated_spectrum = f_interp(x_new)

    # 3. 정규화 수행
    normalized_spectrum = (interpolated_spectrum - train_mean) / train_std

    # 4. 새로운 txt 파일로 저장
    processed_data = np.column_stack((x_new, normalized_spectrum))  # (1600,2) 형태로 저장
    np.savetxt(output_txt, processed_data, fmt="%.30f", delimiter="\t", comments="")

    print(f"Saved processed txt => {output_txt}")
    print(f"Normalized Data: Mean={np.mean(normalized_spectrum):.6f}, Std={np.std(normalized_spectrum):.6f}")

# === 실행 예시 ===
txt_file = "/home/jyounglee/NL_real/data/predict_save/3siwafer_#1_0.5M_0.txt" # 예측에 사용할 txt 파일 경로
train_mean = 9.42063274094999
train_std = 7900.101090550069
output_txt = "/home/jyounglee/NL_real/predict_processing/z_3siwafer_#1_0.5M_0.txt"

process_prediction_txt(txt_file, train_mean, train_std, output_txt)
