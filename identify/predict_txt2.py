import scipy.io as sio
import numpy as np
import os

def save_spectrum_to_txt(mat_file, output_dir):
    # .mat 파일 로드
    data = sio.loadmat(mat_file)
    
    # 'spectrum' 데이터 가져오기
    if 'spectrum' not in data:
        print(f"'spectrum' key not found in {mat_file}")
        return

    spectrum = data['spectrum'].flatten()  # 1D 배열로 변환
    wave = np.arange(1, len(spectrum) + 1)  # wave 생성 (1부터 시작)

    # wave와 spectrum 합치기
    combined = np.column_stack((wave, spectrum))

    # 저장 파일 경로 생성
    base_name = os.path.splitext(os.path.basename(mat_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # .txt 파일 저장
    np.savetxt(output_file, combined, delimiter='\t', fmt='%.6f', header='Wave\tSpectrum', comments='')
    print(f"Saved spectrum to {output_file}")

# .mat 파일 경로와 저장할 디렉토리 설정
mat_file_path = "/home/jyounglee/NL/data/predict/190607_FTC_Nthy_no1.mat"
output_directory = "/home/jyounglee/NL/data/predict"
save_spectrum_to_txt(mat_file_path, output_directory)