import os
import glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

def merge_txt_into_mat_interpolate(
    base_dir = "/home/jyounglee/NL/data/given_data/3layer_on_siwafer",
    out_mat  = "raman_dataset.mat",
    row_points = 1600
):
    """
    base_dir 하위 모든 txt 파일을 재귀적으로 찾고,
    각 파일의 (x, y) 데이터를 로드한 뒤,
    전체 파일의 x축 최소~최대 범위를 구해, 이를 [global_min, global_max]로 설정.
    x_new = np.linspace(global_min, global_max, row_points) 로 보간(cubic).
    
    최종 (n_files, row_points) 크기의 data_matrix, 
    공통 x_new (row_points 길이),
    file_list (n_files 길이) 를 하나의 .mat 파일에 저장.
    
    .mat 구조:
    - data_matrix: shape (n_files, row_points)
    - x_new      : shape (row_points,)
    - file_list  : shape (n_files,) list of strings
    """

    # (1) 재귀적으로 base_dir 아래의 모든 .txt 파일 찾기
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 디렉토리에 txt 파일이 없습니다.")
        return

    # x_min, x_max를 전역으로 추적
    global_min = None
    global_max = None

    # 임시 저장용 (파일별 x, y)
    xy_data = []

    print("[1] 모든 txt 파일 로드 및 (x_min, x_max) 추적 중...")
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath} : {e}")
            continue

        # (N,2) 형식 확인
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath} : shape={arr.shape}, (N,2) 요구.")
            continue

        x = arr[:,0]
        y = arr[:,1]

        # 전역 min, max 갱신
        local_min = np.min(x)
        local_max = np.max(x)
        if global_min is None or local_min < global_min:
            global_min = local_min
        if global_max is None or local_max > global_max:
            global_max = local_max

        xy_data.append( (fpath, x, y) )

    n_files = len(xy_data)
    if n_files == 0:
        print("[결과] 유효한 (N,2) 스펙트럼이 없습니다. 종료.")
        return

    # 보간용 x_new
    x_new = np.linspace(global_min, global_max, row_points)
    print(f"[2] 전역 x 범위=({global_min}, {global_max}), row_points={row_points}")
    print("[3] 각 스펙트럼 보간 및 data_matrix 구성...")

    data_matrix = np.zeros((n_files, row_points), dtype=np.float32)
    path_list   = []

    for i, (fpath, x_arr, y_arr) in enumerate(xy_data):
        # cubic 보간
        f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value="extrapolate")
        y_interp = f_intp(x_new)  # shape=(row_points,)

        data_matrix[i, :] = y_interp.astype(np.float32)
        path_list.append(fpath)

    # (4) .mat 파일 저장
    mat_dict = {
        "data_matrix": data_matrix,  # (n_files, row_points)
        "x_new"      : x_new,        # (row_points,)
        "file_list"  : path_list     # (n_files,) list of strings
    }

    sio.savemat(out_mat, mat_dict)
    print(f"[완료] 총 {n_files}개의 txt 데이터를 보간({row_points}포인트) 후 '{out_mat}'에 저장했습니다.")

def main():
    base_dir = "/home/jyounglee/NL/data/3layer_on_siwafer"
    out_mat  = "raman_dataset_2d.mat"
    row_points = 1600
    merge_txt_into_mat_interpolate(base_dir, out_mat, row_points)

if __name__ == "__main__":
    main()
