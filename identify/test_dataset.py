import numpy as np
import scipy.io as sio
import sys
import os

# 상위 폴더(NL)를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Spectra_generator.py에서 spectra_generator 클래스 불러오기
from Spectra_generator import spectra_generator

# 테스트 데이터셋 생성 함수 정의
def create_test_dataset(input_mat_file, output_mat_file):
    """
    입력 노이즈 데이터를 사용하여 테스트 데이터셋을 생성합니다.

    Args:
        input_mat_file (str): 입력 .mat 파일 경로 (노이즈 데이터 포함, key='noise').
        output_mat_file (str): 출력 .mat 파일 경로.
    """
    # 입력 노이즈 데이터 로드
    input_data = sio.loadmat(input_mat_file)
    noise = input_data['noise']  # (640, 1600) 형태의 노이즈 데이터

    # Ground Truth (lcube) 데이터 생성
    gen = spectra_generator()
    L = noise.shape[1]  # 스펙트럼 길이 (1600)
    batch_size = noise.shape[0]  # 배치 크기 (640)
    lcube = gen.generator(L=L, batch_size=batch_size).T  # (640, 1600)

    # 노이즈 추가하여 cube 생성
    cube = lcube + noise

    # .mat 파일로 저장
    dataset = {'lcube': lcube, 'cube': cube}
    sio.savemat(output_mat_file, dataset)

    print(f"테스트 데이터셋이 생성되어 '{output_mat_file}'에 저장되었습니다.")
    print(f"lcube 크기: {lcube.shape}, cube 크기: {cube.shape}")

# 실행 예시
if __name__ == "__main__":
    input_mat_file = r'/home/jyounglee/NL/data/train/noise_dataset_2.mat'  # 입력 .mat 파일 경로
    output_mat_file = r'/home/jyounglee/NL/data/test/test_dataset_2.mat'  # 출력 .mat 파일 경로
    create_test_dataset(input_mat_file, output_mat_file)
