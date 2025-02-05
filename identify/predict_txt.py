import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from Spectra_generator import spectra_generator  # Spectra_generator.py에서 가져옴

def save_to_txt(filename, wave, data):
    """
    데이터를 .txt 파일로 저장합니다.

    Args:
        filename (str): 저장할 파일 이름.
        wave (numpy.ndarray): 1열에 저장될 wave 데이터.
        data (numpy.ndarray): 2열에 저장될 intensity 데이터.
    """
    combined_data = np.column_stack((wave, data))
    np.savetxt(filename, combined_data, fmt='%.6f', delimiter='\t', header='wave\tx', comments='')

def generate_noisy_data(output_dir, shape=(1, 1600), wave_range=(100, 2000), noise_range=(-50, 50)):
    """
    spectra_generator를 이용해 noisy 데이터를 생성하고 .txt 파일로 저장합니다.

    Args:
        output_dir (str): 저장 디렉토리 경로.
        shape (tuple): 데이터 배열의 형태 (e.g., (1, 1600)).
        wave_range (tuple): Raman shift 범위 (최소값, 최대값).
        noise_range (tuple): 노이즈 범위 (최소값, 최대값).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # wave 생성
    wave = np.linspace(wave_range[0], wave_range[1], shape[1])

    # spectra_generator 초기화
    gen = spectra_generator()

    # Ground Truth 생성
    gt = gen.generator(L=shape[1], batch_size=shape[0]).T

    # 노이즈 생성
    noise = np.random.uniform(noise_range[0], noise_range[1], size=shape)

    # 노이즈 추가 데이터 생성
    noisy = gt + noise

    # 저장 파일 경로 설정
    noisy_filename = os.path.join(output_dir, "predict1.txt")

    # Noisy 데이터 저장
    save_to_txt(noisy_filename, wave, noisy[0])

    print(f"{noisy_filename} 파일이 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    output_directory = "predict_txt"  # 저장할 디렉토리 경로
    generate_noisy_data(output_directory)
