# import torch
# state_dict = torch.load("./Noise-learning/562800.pt", weights_only=True)
# print(state_dict.keys())


# import torch

# # .pt 파일 로드
# state_dict = torch.load("./Noise-learning/562800.pt")

# # 모델 가중치 키 확인
# if 'model' in state_dict:
#     model_weights = state_dict['model']
#     print("모델 가중치 키:")
#     print(model_weights.keys())
# else:
#     print("모델 가중치가 포함되지 않았습니다.")


# import torch
# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         # Conv2 ~ Conv5, Up-sampling 계층 등 추가 구현
#         self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         x = self.Conv1(x)
#         # Forward 단계 추가
#         x = self.Conv_1x1(x)
#         return x

# # 모델 초기화
# model = MyModel()

# # 저장된 가중치 로드
# state_dict = torch.load(r'/home/jyounglee/NL/Noise-learning/562800.pt', weights_only=True)
# model.load_state_dict(state_dict['model'], strict=False)  # strict=False로 일부 누락된 경우 무시

import torch

# PT 파일 경로 설정
pt_file_path = "./Noise-learning/562800.pt"

# 파일 로드
state_dict = torch.load(pt_file_path)

# 파일의 키 확인
print("PT 파일의 키:")
print(state_dict.keys())

# 모델 파라미터 확인
if "model" in state_dict:
    print("\n모델의 가중치 키:")
    print(state_dict["model"].keys())

# 옵티마이저 상태 확인
if "optimizer" in state_dict:
    print("\n옵티마이저 상태 키:")
    print(state_dict["optimizer"].keys())

# 기타 데이터 확인
for key in state_dict:
    if key not in ["model", "optimizer"]:
        print(f"\n키 '{key}'의 데이터:")
        print(state_dict[key])

model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})
optimizer.load_state_dict(state_dict['optimizer'])
