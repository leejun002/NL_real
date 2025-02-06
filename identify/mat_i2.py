import scipy.io as sio

file_path = r'/home/jyounglee/NL_real/data/predict/190607_FTC_Nthy_no1.mat'
mat_data = sio.loadmat(file_path)

print(mat_data.keys())