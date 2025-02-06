import numpy as np

# 예: txt 파일 읽기 (원본 1038행)
data = np.loadtxt(r'/home/jyounglee/NL_real/data/predict/11.txt')  # shape: (1038, 2)

num_original = data.shape[0]  # 1038
target_length = 1600
if num_original < target_length:
    # (target_length - num_original) 만큼 0행 추가
    pad_length = target_length - num_original
    
    # zeros 형태: (pad_length, 2)
    pad_data = np.zeros((pad_length, 2), dtype=data.dtype)
    
    # 붙이기
    data_padded = np.vstack([data, pad_data])
    
    # 최종적으로 (1600, 2) 크기
    np.savetxt('padded.txt', data_padded, fmt='%.6f', delimiter='\t')
