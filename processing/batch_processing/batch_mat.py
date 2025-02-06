import os
import glob
import numpy as np
import scipy.io as sio

def merge_txt_into_mat(
    base_dir = "/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer",
    out_mat  = "raman_dataset.mat"
):
    """
    base_dir 하위 모든 폴더(재귀)에서 *.txt 파일을 찾아
    각 파일의 (x, y) 데이터를 로드한 뒤,
    하나의 .mat 파일로 저장하는 스크립트.

    - data_list: 파일별 (N_i, 2) numpy array를 저장한 리스트
    - file_list: 각 데이터에 대응하는 파일 경로 리스트
    """
    # (1) 재귀적으로 base_dir 아래의 모든 .txt 파일 찾기
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 에 txt 파일이 없음.")
        return

    data_list = []
    path_list = []

    # (2) 모든 txt 파일 로드
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath} : {e}")
            continue

        # 최소 2열(첫 번째 열: x, 두 번째 열: y)
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath} : shape={arr.shape}, (N,2) 이상만 처리.")
            continue

        data_list.append(arr)    # shape (N,2)
        path_list.append(fpath)

    if not data_list:
        print("[결과] 유효한 스펙트럼 데이터 없음. 종료.")
        return

    # (3) Python 리스트 구조를 .mat 파일에 cell array 형태로 저장
    # 'data_list' : 각 txt 파일의 (N,2) 배열을 담은 list
    # 'file_list' : 각 배열에 대응하는 파일 경로
    mat_dict = {
        "data_list": data_list,  # 이건 MATLAB에서 cell array로 읽힘
        "file_list": path_list   # 파일 경로 문자열
    }

    sio.savemat(out_mat, mat_dict)
    print(f"[완료] 총 {len(data_list)}개의 txt 데이터를 '{out_mat}'에 저장했습니다.")

def main():
    base_dir = "/home/jyounglee/NL_real/data/3layer_on_siwafer"
    out_mat  = "raman_dataset.mat"
    merge_txt_into_mat(base_dir, out_mat)

if __name__ == "__main__":
    main()
