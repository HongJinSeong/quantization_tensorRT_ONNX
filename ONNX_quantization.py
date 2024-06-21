import torch

import onnx
import time
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static

from onnxruntime import quantization

import torchvision as tv
### torch -> onnx가 정상적으로 이루어져서 모델이 정상저장됬는지 확인
onnx_model = onnx.load("preprocess_test.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device())

### ONNX를 통한 quantization은 8bit quantization이 지원됨
### preprocess 이전에 onnxruntime의 preprocess를 통해서 처리필요
### dynamic quantization은  동적으로 scailing 진행해서 calibration이 필요없음ㄴ
### static quantization은 calibration이 필요함

## dynamic으로 진행 
## 현재 예시로 돌리는 모델이 resnet50 이라서 dynamic으로하면 Conv layer에서 문제가 있기 때문에 아래의 예시에서는 operation type을 제한함
# quantize_dynamic('preprocess_test.onnx', 'test_quantized_dynamic.onnx', weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul', 'Attention', 'LSTM', 'Gather', 'Transpose', 'EmbedLayerNormalization'] )
## dynamic으로 진행


## stastic으로 진행
## CNN은 static으로 진행이 일반적임
## static으로 진행하기 위해서는 quantization datareader 미리 선언 필요함
preprocess = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageNetValDataset(torch.utils.data.Dataset):
  def __init__(self, transform=None):
      self.transform = transform

  def __len__(self):
      return 100

  def __getitem__(self, idx):
      return torch.randn(3, 256, 256), 1

ds = ImageNetValDataset(transform=preprocess)

class QuntizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)

ort_provider = ['CUDAExecutionProvider']

qdr = QuntizationDataReader(ds, batch_size=1, input_name='input')
q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}
if torch.cuda.is_available():
  q_static_opts = {"ActivationSymmetric":True,
                  "WeightSymmetric":True}

quantize_static('preprocess_test.onnx', 'test_quantized_static.onnx', calibration_data_reader = qdr, extra_options = q_static_opts)
## stastic으로 진행

ort_provider = ['CUDAExecutionProvider']
ort_session = onnxruntime.InferenceSession("test_quantized_static.onnx", providers=ort_provider)

#numpy로 input 넣어야함
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.randn(1, 3, 256, 256, requires_grad=True)
x = x.cuda()

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

start = time.time()
ort_outs = ort_session.run(None, ort_inputs)

print(time.time()-start)



print('gg')