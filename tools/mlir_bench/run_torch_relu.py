import openvino as ov
import torch
import torch.nn as nn
from ml_dtypes import bfloat16

# input: (128, 1024, 512)
A = 128
B = 1
C = 512

use_bf16 = True
dtype = torch.bfloat16 if use_bf16 else torch.float

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()

    def forward(self, x):
        out = torch.relu(x)
        return out

model = ToyNet()


# convert the model into OpenVINO model
example = torch.randn(A, B, C, dtype=dtype)
print("===== Convert model start =====")
ov_model = ov.convert_model(model, example_input=(example,))
ov_model.reshape([A, B, C])
print(ov_model)
print("===== Convert model finish =====")

print("save model start")
saved_path = './toy-net.xml'
ov.save_model(ov_model, saved_path, compress_to_fp16=False)
print("save model finish")

# compile the model for CPU device
core = ov.Core()
device_name = 'CPU'

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
