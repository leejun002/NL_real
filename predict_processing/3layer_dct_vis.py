import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def visualize_dct_all_files(directory):
    """
    지정된 디렉토리 내의 모든 텍스트 파일을 읽어,
    각 파일의 y 데이터(두 번째 열)에 대해 DCT(type-II, norm='ortho')를 수행하고,
    DCT 계수를 시각화합니다.
    """
    # 디렉토리 내 모든 txt 파일 검색
    file_pattern = os.path.join(directory, "*.txt")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print("디렉토리 내에 txt 파일이 없습니다:", directory)
        return
    
    print(f"{len(file_list)}개의 파일을 찾았습니다.")
    num_files = len(file_list)
    
    # 파일 개수에 따라 subplot 구성 (세로로 배치)
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files), squeeze=False)
    
    for idx, file_path in enumerate(file_list):
        try:
            data = np.loadtxt(file_path)
        except Exception as e:
            print(f"{file_path} 파일 로드 실패: {e}")
            continue

        # 데이터가 2열 이상이어야 함 (첫 번째: x, 두 번째: y)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"{file_path} 파일의 데이터 형식이 올바르지 않습니다 (shape={data.shape}).")
            continue

        # x: 첫 번째 열, y: 두 번째 열
        x = data[:, 0]
        y = data[:, 1]
        
        # DCT 변환 (type-II, 정규화: 'ortho')
        y_dct = dct(y, norm='ortho')
        
        # 시각화: DCT 계수 플롯
        ax = axes[idx, 0]
        ax.plot(y_dct, 'r.-')
        ax.set_title(os.path.basename(file_path) + " - DCT")
        ax.set_xlabel("Index")
        ax.set_ylabel("DCT Coefficient")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = "/home/jyounglee/NL/data/3layer/0.5M"
    visualize_dct_all_files(directory)
