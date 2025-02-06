import os
import pandas as pd
import matplotlib.pyplot as plt

# 데이터가 저장된 폴더 경로
data_folder = '/home/jyounglee/NL_real/data/noise'

# 폴더 내 모든 .txt 파일 검색
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

# 서브플롯 설정
num_files = len(txt_files)
cols = 3  # 한 행에 3개의 그래프
rows = (num_files // cols) + (1 if num_files % cols != 0 else 0)  # 필요한 행 수 계산

# 서브플롯 생성
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()  # 축 배열을 평탄화하여 쉽게 접근 가능

# 각 파일 시각화
for i, file_name in enumerate(txt_files):
    file_path = os.path.join(data_folder, file_name)  # 전체 파일 경로
    
    try:
        # 데이터 읽기
        data = pd.read_csv(file_path, sep="\t", header=None, names=["RamanShift", "Signal"])
        
        # 데이터 플롯팅
        axes[i].plot(data["RamanShift"], data["Signal"], color='blue')
        axes[i].set_title(file_name, fontsize=10)
        axes[i].set_xlabel("Raman Shift")
        axes[i].set_ylabel("Signal Intensity")
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    except Exception as e:
        axes[i].set_title(f"Error: {file_name}", fontsize=10)
        print(f"Error processing file {file_name}: {e}")

# 빈 서브플롯 숨기기
for j in range(len(txt_files), len(axes)):
    fig.delaxes(axes[j])

# 레이아웃 조정 및 출력
plt.tight_layout()
plt.show()
