### tensorRT로 최적화시킨 모델 다시 불러서 inference 진행하기
### GPU 메모리 영역으로 데이터 복사하는 행위해야되서 pycuda도 필요함
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit   ### cuda.pagelocked_empty 함수 호출시 에러나서 추가함
from time import time

tensorrt_file_name = 'model_32.plan'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

with open(tensorrt_file_name,'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


inputs, outputs, bindings, stream, allocations = [], [], [], [], []

for binding in engine:
    ### max-size 기준으로 set
    ### tensorRT endgine 저장시에 profile에 input size 를 min opt max로 넣기 때문에 여기서는 max로 set함
    size = trt.volume(engine.get_tensor_profile_shape(binding, 0)[2]) * 1
    ### profile 시에 data type 정의해둔 data type 불러오기
    dtype = trt.nptype(engine.get_tensor_dtype(binding))
    ### host(cpu쪽) 메모리 set numpy로 return 되는거보니 그냥 위에서 size dtype 받아오면 np.zeros로 해도 될듯
    host_mem = cuda.pagelocked_empty(size, dtype)
    ### device(gpu쪽) 메모리로 set
    ### 이전에 cuda 기반 GPU 병렬 프로그래밍에서 선언하던 것과 유사함
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    allocations.append(device_mem)

    bindings.append(int(device_mem))
    if binding == 'input':
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))
context = engine.create_execution_context()

# input set
### input data 불러올때 꼭 위에서 binding한  input data type 확인해서 똑같이 넣어두기
input = np.ones(shape=(1, 3, 256, 256), dtype="float32")
# ascontiguousarray 함수는 메모리에 연속적으로 저장되지 않는 배열을 연속적으로 저장되는 배열로 변환
input = np.ascontiguousarray(input)
# input image device(gpu쪽)메모리에 set
cuda.memcpy_htod(inputs[0].device, input)
## inference 속도 확인하기 ( 메모리 넣고 빼고 말고 순수 inference만 체크하기
start = time()
context.execute_v2(allocations)
print(time()-start)
## inference 속도 확인하기 ( 메모리 넣고 빼고 말고 순수 inference만 체크하기



print('ccvc')