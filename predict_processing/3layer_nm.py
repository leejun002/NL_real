import os
import numpy as np
import matplotlib.pyplot as plt

def load_spectrum(file_path):
    """
    지정된 텍스트 파일을 로드하여, 첫 번째 열(x축)과 두 번째 열(y축)을 반환합니다.
    """
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"파일 로드 중 오류: {e}")
        return None, None
    
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"데이터 형식이 올바르지 않습니다. 데이터 shape: {data.shape}")
        return None, None
    
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def normalize_signal(y):
    """
    y 신호에 대해 z‑score 정규화를 수행합니다.
    만약 표준편차가 너무 작으면 평균만 제거합니다.
    """
    mean_val = np.mean(y)
    std_val = np.std(y)
    if std_val < 1e-6:
        normalized = y - mean_val
    else:
        normalized = (y - mean_val) / std_val
    return normalized

def main():
    # 파일 경로 설정 (필요에 따라 경로 수정)
    file_path = "/home/jyounglee/NL/data/predict/3siwafer_#1_0.5M_0.txt"
    if not os.path.exists(file_path):
        print("파일이 존재하지 않습니다:", file_path)
        return

    # 스펙트럼 데이터 로드
    x, y = load_spectrum(file_path)
    if x is None or y is None:
        return

    # 정규화 수행
    y_norm = normalize_signal(y)

    # 원본과 정규화된 결과 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', label="Original")
    plt.title("Original Spectrum")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, y_norm, 'r-', label="Normalized")
    plt.title("Normalized Spectrum (z-score)")
    plt.xlabel("x")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
