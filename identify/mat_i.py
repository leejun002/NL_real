import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom

def crop_center(data, crop_width, crop_height):
    """
    3D 데이터에서 중심 부분을 crop합니다.
    Args:
        data (numpy.ndarray): 3D 데이터 (Width x Wavenumbers x Height).
        crop_width (int): 자를 가로 크기.
        crop_height (int): 자를 세로 크기.
    Returns:
        numpy.ndarray: crop된 데이터.
    """
    width, wavenumbers, height = data.shape
    start_w = (width - crop_width) // 2
    start_h = (height - crop_height) // 2
    return data[start_w:start_w + crop_width, :, start_h:start_h + crop_height]

def resize_data(data, target_shape):
    """
    데이터를 리샘플링합니다.
    Args:
        data (numpy.ndarray): 3D 데이터.
        target_shape (tuple): 목표 크기 (Width, Wavenumbers, Height).
    Returns:
        numpy.ndarray: 리샘플링된 데이터.
    """
    factors = [
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2],
    ]
    return zoom(data, factors, order=1)  # 선형 보간 사용

# 데이터 로드
file_path = r'/home/jyounglee/NL/identify/190607_FTC_Nthy_no1.mat'
data = sio.loadmat(file_path)

# 원본 데이터
orig_data = data['origimagedata']  # 3D 데이터 (Width x Wavenumbers x Height)
wavenumbers = data['wavenumber'].flatten()  # 1D array

# 중심부 추출
cropped_data = crop_center(orig_data, crop_width=320, crop_height=320)

# 리샘플링하여 목표 크기로 조정 (640x1600로 축소)
resized_data = resize_data(cropped_data, target_shape=(640, 1600, 640))

# 최종 데이터 저장
sio.savemat("FTC_processed_resized.mat", {
    'cube': resized_data,
    'wavenumber': wavenumbers
})