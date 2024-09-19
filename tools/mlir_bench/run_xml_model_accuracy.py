import openvino as ov
import numpy as np
import ml_dtypes
import torch
import os

xml_file_name = "MLIR_MLP_BENCH_f32_128_1024.xml"
# xml_file_name = "MLIR_MLP_BENCH_bf16_128_1024.xml"

# compile the model for CPU device
core = ov.Core()
device_name = 'CPU'

model = core.read_model(xml_file_name)
# model.reshape([batch_size, in_features])
print("Ops in the model: ", model.get_ordered_ops())

# only support single-input model: https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.__call__
for param in model.get_parameters():
    shape = param.get_output_shape(0)
    type = param.get_output_element_type(0).get_type_name()
    if type == "f32":
        input = torch.randn(*shape, dtype=torch.float32).numpy()
    elif type == "bf16":
        input = torch.randn(*shape, dtype=torch.bfloat16).view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
    else:
        print("Unsupported data type: ", type)
        exit()


os.environ['OV_MLIR'] = '0'
print("MLIR Enabled: ", os.getenv('OV_MLIR'))
compiled_model = core.compile_model(model=model, device_name=device_name)
print(compiled_model.get_runtime_model().get_ordered_ops())

print("===== #1 run start =====")
output_ov = compiled_model({0: input})
print("===== #1 run finish =====")


os.environ['OV_MLIR'] = '1'
os.environ['OV_MLIR_DEBUG'] = '1'
print("MLIR Enabled: ", os.getenv('OV_MLIR'))
compiled_model = core.compile_model(model=model, device_name=device_name)
print(compiled_model.get_runtime_model().get_ordered_ops())

print("===== #1 run start =====")
output_gc_1 = compiled_model({0: input})
print("===== #1 run finish =====")

print("===== #2 run start =====")
output_gc_2 = compiled_model({0: input})
print("===== #2 run finish =====")

# Default 'rtol=1e-05, atol=1e-08' will report error.
if type == "f32":
    tol = 1e-5
elif type == "bf16":
    tol = 1e-3
print("OV output vs GC #1 output close: ", np.allclose(output_ov[0].astype(np.float32), output_gc_1[0].astype(np.float32), rtol=tol, atol=tol))
print("OV output vs GC #2 output close: ", np.allclose(output_ov[0].astype(np.float32), output_gc_2[0].astype(np.float32), rtol=tol, atol=tol))
