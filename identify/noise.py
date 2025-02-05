import numpy as np
import scipy.io as sio

# 노이즈 생성 함수
def generate_noise(shape, noise_range=(-50, 50)):
    """
    랜덤 노이즈를 생성합니다.

    Args:
        shape (tuple): 노이즈 배열의 형태 (e.g., (640, 1600)).
        noise_range (tuple): 노이즈 범위 (최소값, 최대값).

    Returns:
        numpy.ndarray: 생성된 노이즈 배열.
    """
    return np.random.uniform(noise_range[0], noise_range[1], size=shape)

# 노이즈 배열 생성
shape = (640, 1600)  # 원하는 크기
noise_range = (-50, 50)  # 노이즈 범위 (-50 to 50 intensity)
noise_data = generate_noise(shape, noise_range)

# .mat 파일 저장
def save_to_mat(filename, data, key='noise'):
    """
    .mat 파일로 데이터를 저장합니다.

    Args:
        filename (str): 저장할 파일 이름.
        data (numpy.ndarray): 저장할 데이터.
        key (str): .mat 파일 내의 데이터 키.
    """
    sio.savemat(filename, {key: data})

# 파일 이름과 저장
output_filename = "noise_dataset_2.mat"
save_to_mat(output_filename, noise_data, key='noise')

print(f"{output_filename} 파일이 생성되었습니다. {shape} 형태의 노이즈가 저장되었습니다.")
