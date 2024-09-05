import openvino as ov
import torch
import torch.nn as nn
from ml_dtypes import bfloat16

# input: (128, 1024). weight: (1024, 512). bias: (512). output: (128, 512)
batch_size = 128
in_features = 1024
out_features = 512

use_bf16 = True
dtype = torch.bfloat16 if use_bf16 else torch.float

# class ToyNet(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ToyNet, self).__init__()
#         # input: (N, in_feature), weight: (out_feature, in_feature), bias: (out_feature).
#         # out = x * W^T + b
#         self.fc = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         out = self.fc(x)
#         return out

# model = ToyNet(in_features, out_features)



class ToyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ToyNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype), requires_grad=False)
    def forward(self, x):
        out = x @ self.weight + self.bias
        out = torch.relu(out)
        return out

model = ToyNet(in_features, out_features)




#class ToyNet(nn.Module):
#    def __init__(self, in_features, out_features):
#        super(ToyNet, self).__init__()
#        # input: (N, in_feature), weight: (out_feature, in_feature), bias: (out_feature).
#        # out = x * W^T + b
#        self.fc1 = nn.Linear(in_features, in_features)
#        self.fc2 = nn.Linear(in_features, out_features)
#        self.fc3 = nn.Linear(out_features, out_features)
#
#    def forward(self, x):
#        out = torch.relu(self.fc1(x))
#        out = torch.relu(self.fc2(out))
#        out = torch.relu(self.fc3(out))
#        return out

# # (128x64, 64x64) -> (128x64). (128x64, 64x64) -> (128x64). Two same matmuls.
#model = ToyNet(in_features, out_features)

# in_features = 16
# out_features = 64
# # (128x16, 16x16) -> (128x16). (128x16, 16x64) -> (128x64). Two different matmuls.
# model = ToyNet(in_features, out_features)


# load PyTorch model into memory
# model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", weights="DEFAULT")

# convert the model into OpenVINO model
example = torch.randn(batch_size, in_features, dtype=dtype)
print("===== Convert model start =====")
ov_model = ov.convert_model(model, example_input=(example,))
ov_model.reshape([batch_size, in_features])
print(ov_model)
print("===== Convert model finish =====")

# print(ov_model.is_dynamic())
# print(ov_model.get_ordered_ops())
# weight = ov_model.get_ordered_ops()[1]
# print(weight.get_output_tensor(0))
# print(weight.get_output_tensor(0).shape)

# print("Compress weights start")
# ov_model = compress_weights(ov_model)
# print(ov_model)
# print("Compress weights finish")

# openvino.save_model(model: ov::Model, output_model: object, compress_to_fp16: bool = True)
print("save model start")
saved_path = './toy-net.xml'
ov.save_model(ov_model, saved_path, compress_to_fp16=False)
print("save model finish")

# compile the model for CPU device
core = ov.Core()
device_name = 'CPU'

# Find 'EXPORT_IMPORT' capability in supported capabilities
#caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, 'OPTIMIZATION_CAPABILITIES')
#print("===== On device: ", device_name, ", model caching supported: ", caching_supported, " =====")


print("===== Compile model start =====")
compiled_model = core.compile_model(ov_model, device_name)
print(compiled_model)
print("===== Compile model finish =====")

if use_bf16:
    example_np = example.view(dtype=torch.uint16).numpy().view(bfloat16)
else:
    example_np = example.numpy()

# infer the model on random data
print("===== #1 run start =====")

output = compiled_model({0: example_np})
print("===== #1 run finish =====")

print("===== #2 run start =====")
output = compiled_model({0: example_np})
print("===== #2 run finish =====")


#model_from_read = core.read_model(saved_path)
#print("Read model:\n", model_from_read)
#compiled_model_from_read = core.compile_model(model=model_from_read, device_name=device_name)
#print("Read model compiled:\n", compiled_model_from_read)

#print("===== #1 run start =====")
#output = compiled_model_from_read({0: example.numpy()})
#print("===== #1 run finish =====")

# /home/xiaoguang/ov-llm/openvino-slyalin-mlir/bin/intel64/Release/benchmark_app -m toy-net.xml -d CPU -ip f32 -infer_precision f32  -niter 10 -hint none -nstreams 1 -nthreads 16
