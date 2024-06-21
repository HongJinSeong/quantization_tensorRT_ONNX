import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn

from torchvision import models

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torch.onnx
import time

model = models.resnet50(pretrained=True)
model.cuda()
model.eval()

batch_size = 1    # 임의의 수

x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
x = x.cuda()

start = time.time()
output = model(x)
print(time.time()-start)

map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

### torch -> onnx로 저장진행
torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "test.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})