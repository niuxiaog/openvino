import openvino as ov
import torch

# input: (128, 16). weight: (16, 64). bias: (64). output: (128, 64)
batch_size = 128
in_features = 16
out_features = 64
saved_path = './toy-net.xml'

batch_size = 128
in_features = 1024
out_features = 1024
saved_path = './MLIR_MLP_BENCH_f32_128_1024.xml'

# compile the model for CPU device
core = ov.Core()
device_name = 'CPU'

model_from_read = core.read_model(saved_path)
# model_from_read.reshape([batch_size, in_features])
print(model_from_read)
print(model_from_read.get_ordered_ops())
compiled_model_from_read = core.compile_model(model=model_from_read, device_name=device_name)
print("Read model compiled:\n", compiled_model_from_read)

example = torch.randn(batch_size, in_features)
print("===== #1 run start =====")
output = compiled_model_from_read({0: example.numpy()})
print("===== #1 run finish =====")
