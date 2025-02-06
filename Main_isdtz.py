import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
import numpy as np
import os
from scipy.fftpack import dct, idct
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'


# 체크포인트 디렉토리 확인 및 생성
def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)


# 테스트 결과 저장 경로 정의
def test_result_dir(config):
    result_dir = os.path.join(config.batch_save_root, config.Instrument,
                              config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


# 모델 저장 경로 정의
def save_model_dir(config):
    save_model_dir = os.path.join(config.checkpoint, config.Instrument,
                                  config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    return save_model_dir


# 로그 경로 정의
def save_log_dir(config):
    save_log_dir = os.path.join(config.logs, config.Instrument,
                                config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    return save_log_dir

# 학습 함수
def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    schedual = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)

    save_model_path = save_model_dir(config)
    save_log = save_log_dir(config)
    global_step = 0

    # 사전 학습된 모델 로드
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model at global step =', global_step)

    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)

    gen_train = spectra_generator()
    gen_valid = spectra_generator()

    writer = SummaryWriter(save_log)

    for epoch in range(config.max_epoch):
        # ========== Training Loop ==========
        for idx, noise in enumerate(train_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_train.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise

            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)

            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')

            input_coef = input_coef.reshape(-1, 1, spec)
            output_coef = output_coef.reshape(-1, 1, spec)

            input_coef = torch.from_numpy(input_coef).float()
            output_coef = torch.from_numpy(output_coef).float()
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)

            global_step += 1
            model.train()
            optimizer.zero_grad()

            preds = model(input_coef)
            train_loss = criterion(preds, output_coef)
            train_loss.backward()
            optimizer.step()

            writer.add_scalar('train loss', train_loss.item(), global_step)

            if idx % config.print_freq == 0:
                print(f'epoch {epoch}, batch {idx}, global_step {global_step}, train loss = {train_loss.item()}')

        # ========== Validation Loop ==========
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_valid.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise

            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)

            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')

            input_coef = input_coef.reshape(-1, 1, spec)
            output_coef = output_coef.reshape(-1, 1, spec)

            input_coef = torch.from_numpy(input_coef).float()
            output_coef = torch.from_numpy(output_coef).float()
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)

            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()

        valid_loss /= len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step)

        # 학습률 스케줄러 업데이트
        schedual.step()

        # ========== 100 에포크마다 모델 저장 ==========
        if (epoch + 1) % 100 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'loss': train_loss.item()
            }
            # epoch 정보를 파일명에 포함
            model_file = os.path.join(save_model_path, f'{global_step}.pt')
            torch.save(state, model_file)
            print(f"Model saved at {model_file} (epoch {epoch+1})")


def batch_predict(config):
    """
    1) 모델 로드
    2) predict_root 경로에서 .mat 파일 로드 (예: merged_spectra_1280.mat, key='data_matrix')
    3) data_matrix(1280,1600) 각 행 i에 대해:
        (a) SVD -> BG 제거 (rank=1)
        (b) DCT(행 단위)
        (c) z-score 정규화(학습 시 mean/std)
        (d) 모델 예측 => 노이즈 DCT 계수
        (e) 역정규화
        (f) IDCT => 노이즈
        (g) 최종 = (원본 행 - BG) - 노이즈 + BG  또는 (원본 행 - predicted_noise)
           (BG 재결합)
    4) 결과를 background_list, noise_pred_list, denoise_list 등으로 저장
    """
    import torch
    import numpy as np
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd

    print("batch predicting...")

    # -----------------------------
    # (1) 모델 로드
    # -----------------------------
    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    if not os.path.exists(model_file):
        print(f"[오류] 모델 파일이 없음: {model_file}")
        return

    state = torch.load(model_file, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print(f"Successfully loaded model at global step={state['global_step']}")
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")

    # -----------------------------
    # (1-2) 학습 시 사용한 mean, std 로드 (noise_data.mat 등)
    # -----------------------------
    # 가정: config.noise_mat_path 에 학습시 쓴 noise_data.mat 경로가 있음
    #       그 안에 'train_mean', 'train_std'가 저장되어 있다고 가정
    if not hasattr(config, 'noise_mat_path'):
        print("[경고] config에 noise_mat_path가 정의되지 않음. (train_mean, train_std 못 불러옴)")
        train_mean, train_std = 0.0, 1.0
    else:
        if not os.path.exists(config.noise_mat_path):
            print(f"[오류] noise_mat_path 파일이 없음: {config.noise_mat_path}")
            return
        tmp_meanstd = sio.loadmat(config.noise_mat_path)
        train_mean = float(tmp_meanstd['train_mean'].ravel()[0])
        train_std  = float(tmp_meanstd['train_std'].ravel()[0])
    print(f"[info] Loaded train_mean={train_mean:.4f}, train_std={train_std:.4f}")

    # -----------------------------
    # (2) 사물 스펙트럼 .mat 로드
    # -----------------------------
    mat_filename = os.path.join(config.batch_predict_root, "merged_spectra_1280.mat")
    if not os.path.exists(mat_filename):
        print(f"[오류] 파일이 존재하지 않습니다: {mat_filename}")
        return

    tmp = sio.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)

    if 'data_matrix' not in tmp:
        print("[오류] 'data_matrix' 키가 없음.")
        return

    data_matrix = tmp['data_matrix']  # shape=(1280,1600) float32
    data_matrix = data_matrix.astype(np.float64)  # 필요시 double 변환
    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape=({n_files},{spec_len}) (예:1280,1600)")

    # -----------------------------
    # (3) 각 행 i에 대해 BG 제거(SVD), DCT, 정규화, 모델 예측, 역정규화, IDCT, BG 결합
    # -----------------------------
    background_list = []
    noise_pred_list = []
    denoise_list    = []

    rank_bg = 2  # 제거할 배경 랭크 (1)
    for i in range(n_files):
        # (A) 한 줄 (1,1600) 추출
        row_data = data_matrix[i, :]  # shape=(1600,)

        # (B) SVD -> rank=1 배경 제거
        #   row_data(1,1600) => reshape(1600,1)로 SVD할 수도 있지만, 여기선 바로 1D SVD
        #   편의상 (1,1600) 형태로 SVD를 하려면:
        row_2d = row_data.reshape((1, -1))  # (1,1600)
        U, s, Vt = svd(row_2d, full_matrices=False)
        # rank=1 배경
        #  => outer(U[:,0], s[0]*Vt[0,:]) = U[:,0]*s[0]*Vt[0,:]
        # 여기서 shape=(1,1) * scalar * (1,1600)
        bg_2d = np.outer(U[:,0], s[0] * Vt[0,:])  # (1,1600)
        # residual
        residual_2d = row_2d - bg_2d
        bg_1d = bg_2d.flatten()
        residual_1d = residual_2d.flatten()

        # (C) DCT(행 단위): residual은 (1600,) 1D이므로 dct로
        resid_dct = dct(residual_1d, norm='ortho')  # shape=(1600,)

        # (D) z-score 정규화: (resid_dct - train_mean)/train_std
        resid_dct_norm = (resid_dct - train_mean) / train_std

        # (E) 모델 예측
        #   shape => (1,1,1600) [batch=1, channel=1, length=1600]
        inp_t = resid_dct_norm.reshape(1,1,-1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()

        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()  # shape=(1,1,1600)
        pred_out = pred_out.reshape(-1)  # (1600,)

        # (F) 역정규화 => pred_dct
        pred_dct = pred_out * train_std + train_mean

        # (G) IDCT => noise_pred
        noise_pred_1d = idct(pred_dct, norm='ortho')  # shape=(1600,)

        # (H) BG 재결합 => denoised = (raw - noise_pred)
        #   여기서 raw = row_data, BG+residual
        #   row_data = bg_1d + residual_1d
        #   residual_1d - noise_pred_1d = clean residual
        #   최종 => bg_1d + (residual_1d - noise_pred_1d)
        #   또는 row_data - noise_pred_1d
        denoised_1d = row_data - noise_pred_1d

        # 저장
        background_list.append(bg_1d)
        noise_pred_list.append(noise_pred_1d)
        denoise_list.append(denoised_1d)

    # -----------------------------
    # (4) 결과 저장
    # -----------------------------
    background_arr = np.array(background_list, dtype=object)
    noise_arr      = np.array(noise_pred_list, dtype=object)
    denoise_arr    = np.array(denoise_list,  dtype=object)

    # 기존 tmp dict에 결과 삽입 (optionally)
    tmp['background_list'] = background_arr
    tmp['noise_pred_list'] = noise_arr
    tmp['denoise_list']    = denoise_arr

    out_dir = test_result_dir(config)
    out_name = os.path.join(out_dir, "isdtz_bg2_result.mat")
    sio.savemat(out_name, tmp, do_compression=True)
    print(f"[완료] 최종 결과 저장 -> {out_name}")



def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
