import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def visualize_all_files(directory):
    # 지정한 디렉토리 내의 모든 txt 파일 검색
    file_pattern = os.path.join(directory, "*.txt")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print("디렉토리 내에 txt 파일이 없습니다:", directory)
        return

    print(f"{len(file_list)}개의 파일을 찾았습니다.")

    # 파일 개수에 따라 subplot 구성 (파일 개수가 1개 이상일 때)
    num_files = len(file_list)
    # 한 Figure에 모두 그리기 (세로로 배치)
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files), squeeze=False)
    
    for idx, file_path in enumerate(file_list):
        try:
            data = np.loadtxt(file_path)
        except Exception as e:
            print(f"{file_path} 파일 로드 실패: {e}")
            continue

        # 데이터가 2열 미만이면 건너뜁니다.
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"{file_path} 파일의 데이터 형식이 올바르지 않습니다 (shape={data.shape}).")
            continue

        x = data[:, 0]
        y = data[:, 1]
        ax = axes[idx, 0]
        ax.plot(x, y, 'b.-')
        ax.set_title(os.path.basename(file_path))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = "/home/jyounglee/NL_real/data/3layer/0.5M"
    visualize_all_files(directory)
