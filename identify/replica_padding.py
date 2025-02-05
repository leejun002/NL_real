import numpy as np

data = np.loadtxt(r'/home/jyounglee/NL/SI-wafer + #1 AuNPs (44nm)_1구역 +0.5M glycine_RamanShift__0__17-23-16-392.txt')
last_row = data[-1]  # e.g. [wave_end, intensity_end]
pad_length = 1600 - data.shape[0]

pad_data = np.tile(last_row, (pad_length, 1))
data_padded = np.vstack([data, pad_data])
np.savetxt('r_p_11.txt', data_padded, fmt='%.6f', delimiter='\t')