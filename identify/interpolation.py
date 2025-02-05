import numpy as np

# 원본
data = np.loadtxt(r'/home/jyounglee/NL/data/noise/1/pico + glass 1_RamanShift__0__16-23-01-143.txt')
wave = data[:, 0]
intensity = data[:, 1]

# 원하는 x축을 1600개 균등 분할
new_wave = np.linspace(wave.min(), wave.max(), 1600)

# intensity를 new_wave에 맞춰 선형보간
new_intensity = np.interp(new_wave, wave, intensity)

resampled_data = np.column_stack([new_wave, new_intensity])
np.savetxt('ip_noise_1_0.txt', resampled_data, fmt='%.6f', delimiter='\t')
