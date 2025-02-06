import os
import glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

def merge_txt_to_single_key_mat(
    base_dir = "/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer",
    out_mat  = "merged_spectra.mat",
    row_points = 1600
):
    """
    base_dir 내(하위 폴더 포함)의 모든 .txt 파일을 찾아,
    각 파일의 (x, y) 데이터를 로드.
    모든 파일에서 x축의 전역 min~max를 구하여,
    x_new = np.linspace(global_min, global_max, row_points) 에 대해
    cubic 보간을 수행.
    
    최종 결과 (n_files, row_points) 크기의 data_matrix를
    하나의 key='data_matrix'로 .mat 파일에 저장합니다.
    """
    # 1) base_dir 및 모든 하위 디렉토리에서 .txt 파일 검색
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.")
        return

    # 전역 x 범위 추적용 변수
    global_min = None
    global_max = None

    # 임시로 (파일경로, x, y) 저장
    data_entries = []

    print("[1] 모든 txt 파일 로드 및 (x_min, x_max) 추적 중...")
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath} : {e}")
            continue

        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath} : (N,2) 형태가 아님. shape={arr.shape}")
            continue

        x = arr[:, 0]
        y = arr[:, 1]

        # 최소 2개 이상의 점이 있어야 보간 가능
        if len(x) < 2:
            print(f"[스킵] {fpath} : 샘플 수가 너무 적음.")
            continue

        # local min/max
        lmin, lmax = x.min(), x.max()
        if global_min is None or lmin < global_min:
            global_min = lmin
        if global_max is None or lmax > global_max:
            global_max = lmax

        data_entries.append((fpath, x, y))

    n_files = len(data_entries)
    if n_files == 0:
        print("[결과] 유효한 (x,y) 스펙트럼이 없습니다. 종료.")
        return

    print(f"[2] 총 {n_files}개의 txt 파일을 확인했습니다.")
    print(f"    전역 x 범위=({global_min:.4f}, {global_max:.4f}), row_points={row_points}")

    # 2) 보간용 x_new = 전역 min~max 구간에서 row_points개
    x_new = np.linspace(global_min, global_max, row_points)

    # 3) 보간 결과를 (n_files, row_points) 크기로 저장
    data_matrix = np.zeros((n_files, row_points), dtype=np.float32)

    print("[3] 각 파일에 대해 cubic 보간 실시...")
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        # cubic 보간 (fill_value='extrapolate')
        interp_fn = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
        y_interp = interp_fn(x_new)
        data_matrix[i, :] = y_interp.astype(np.float32)

    # 4) .mat 파일로 저장
    # 하나의 key='data_matrix' 만 저장
    mat_dict = {
        "data_matrix": data_matrix
    }
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] 보간 완료. shape={data_matrix.shape}")
    print(f"         '{out_mat}' 에 data_matrix만 저장했습니다.")

def main():
    base_dir = "/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer"
    out_mat = "merged_spectra.mat"
    row_points = 1600

    merge_txt_to_single_key_mat(base_dir, out_mat, row_points)

if __name__ == "__main__":
    main()
