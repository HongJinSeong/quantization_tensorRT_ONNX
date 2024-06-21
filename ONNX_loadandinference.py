import torch

import onnx
import time
import numpy as np


### torch -> onnx가 정상적으로 이루어져서 모델이 정상저장됬는지 확인
onnx_model = onnx.load("test.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device())
ort_provider = ['CUDAExecutionProvider']
ort_session = onnxruntime.InferenceSession("test.onnx", providers=ort_provider)

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