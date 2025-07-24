import torch

# 하이퍼파라미터
BATCH_SIZE = 256
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

# 모델 버전 번호 (에피소드별 저장용)
MODEL_VERSION = 10

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")