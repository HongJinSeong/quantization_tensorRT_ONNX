## tensorRT의 경우에는 단순 quantization을 통한 scailing 뿐만 아니라
## NVIDIA GPU상에서 추론속도를 크게 향상시킬수 있는 모델 최적화 엔진임
## 아래의 예시는 ONNX 를 통해 만든 model을 16bit 로 quantization 과 최적화 진행

import tensorrt as trt

onnx_file_name = 'test.onnx'
tensorrt_file_name = 'model_32.plan'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
## input을 고정아니라 dynamic하게 사용하도록 set
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

## logger
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
builder = trt.Builder(TRT_LOGGER)

network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)

config = builder.create_builder_config()


# config.set_flag(trt.BuilderFlag.FP16)
# config.set_flag(trt.BuilderFlag.FP32)

############ 안댐 ###############
# config.set_flag(trt.BuilderFlag.INT8)
############ 안댐 ###############

with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

inputTensor = network.get_input(0)
profile = builder.create_optimization_profile()
### shape 지정시에 min opt max 에 대한 shape를 모두 줘야 해서 3개가 들어가야함
### 예시로는 그냥 단일 input 들어간다 가정하고 3가지에 모두 같은 값을 부여함
profile.set_shape(inputTensor.name, (1, 3, 256, 256), \
    (1, 3, 256, 256), \
    (1, 3, 256, 256))
config.add_optimization_profile(profile)


## 위에 정의한 data type FP16 기준으로 최적화 진행
# engine = builder.build_cuda_engine(network)
plan = builder.build_serialized_network(network, config)
with trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(plan)

## tensorRT engine으로 저장
buf = engine.serialize()
with open(tensorrt_file_name, 'wb') as f:
    f.write(buf)