import torch

# 하이퍼파라미터
BATCH_SIZE = 128
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 