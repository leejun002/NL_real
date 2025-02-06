import os
import glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

def merge_txt_to_single_key_mat_1280(
    base_dir = "/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer",
    out_mat  = "/home/jyounglee/NL_real/data/batch_predict/merged_spectra_1280.mat",
    row_points = 1600,
    target_files = 1280
):
    """
    base_dir 내(하위 폴더 포함)의 모든 .txt 파일을 찾아,
    각 파일의 (x, y) 데이터를 로드 후,
    전역 x_min~x_max 범위에서 row_points(1600)개로 cubic 보간.
    -> (n_files, row_points) 결과.

    이후 n_files가 target_files(1280)보다 작으면 부족분만큼 마지막 스펙트럼을 복제,
    n_files가 1280보다 크면 앞 1280개만 사용,
    최종 (1280, 1600) 크기의 data_matrix를
    하나의 key 'data_matrix'로 .mat 파일에 저장.
    """

    # 1) .txt 파일 검색 (재귀)
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.")
        return

    # 전역 x 범위 추적용
    global_min = None
    global_max = None
    data_entries = []

    print("[1] txt 파일 로드 & global_min~max 탐색...")
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath}: {e}")
            continue

        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath}: (N,2) 형태가 아님. shape={arr.shape}")
            continue

        x = arr[:,0]
        y = arr[:,1]
        if x.size < 2:
            print(f"[스킵] {fpath}: 점이 너무 적어 보간 불가.")
            continue

        # local min / max
        lmin, lmax = x.min(), x.max()
        if global_min is None or lmin < global_min:
            global_min = lmin
        if global_max is None or lmax > global_max:
            global_max = lmax

        data_entries.append((fpath, x, y))

    n_files = len(data_entries)
    if n_files == 0:
        print("[결과] 유효한 스펙트럼 없음. 종료.")
        return

    print(f"총 {n_files}개 txt 파일 발견.")
    print(f"전역 x 범위=({global_min}, {global_max}), row_points={row_points}")

    # 2) 보간용 x_new
    x_new = np.linspace(global_min, global_max, row_points)

    # 3) 보간 -> (n_files, row_points)
    data_matrix_all = np.zeros((n_files, row_points), dtype=np.float32)

    print("[2] cubic 보간 중...")
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        interp_fn = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
        y_interp = interp_fn(x_new)
        data_matrix_all[i, :] = y_interp.astype(np.float32)

    # 4) n_files를 target_files(=1280)에 맞춤
    if n_files == target_files:
        data_matrix = data_matrix_all
        print(f"n_files={n_files} 정확히 {target_files}와 일치.")
    elif n_files > target_files:
        # 앞쪽 1280개만 사용
        print(f"n_files={n_files} > {target_files}. 앞 {target_files}개만 사용.")
        data_matrix = data_matrix_all[:target_files, :]
    else:
        # n_files < 1280인 경우
        diff = target_files - n_files
        print(f"n_files={n_files} < {target_files}. 마지막 스펙트럼 {diff}번 복제.")
        data_matrix = np.zeros((target_files, row_points), dtype=np.float32)
        data_matrix[:n_files, :] = data_matrix_all
        # 마지막 하나를 반복 복제
        last_spec = data_matrix_all[-1, :]
        for i in range(diff):
            data_matrix[n_files + i, :] = last_spec

    print(f"[결과] 최종 data_matrix shape={data_matrix.shape} (1280, 1600)")

    # 5) .mat 저장 (단일 key='data_matrix')
    mat_dict = {"data_matrix": data_matrix}
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}' 에 (1280 x {row_points}) 스펙트럼 저장.")

def main():
    base_dir = "/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer"
    out_mat  = "merged_spectra_1280.mat"
    row_points = 1600
    merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points)

if __name__ == "__main__":
    main()
