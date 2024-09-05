module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>, #dlti.dl_entry<"num_threads", 4 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : i32>, #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : i32>, #dlti.dl_entry<"L3_cache_size_in_bytes", 1966080 : i32>, #dlti.dl_entry<"max_vector_width", 512 : i32>>>} {
  func.func @entry(%arg0: memref<128x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<128x1024xbf16>, %arg3: memref<128x1024xbf16>) attributes {compiletime_const_args_index = [1 : i32, 2 : i32]} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<128x1024xbf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xbf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<128x1024xbf16>
    %3 = tensor.empty() : tensor<1024x1024xbf16>
    %transposed = linalg.transpose ins(%1 : tensor<1024x1024xbf16>) outs(%3 : tensor<1024x1024xbf16>) permutation = [1, 0]
    %4 = tensor.empty() : tensor<128x1024xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill ins(%cst : bf16) outs(%4 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
    %6 = linalg.matmul ins(%0, %transposed : tensor<128x1024xbf16>, tensor<1024x1024xbf16>) outs(%5 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
    %7 = tensor.empty() : tensor<128x1024xbf16>
    %8 = linalg.add ins(%6, %2 : tensor<128x1024xbf16>, tensor<128x1024xbf16>) outs(%7 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
    %9 = tensor.empty() : tensor<128x1024xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %10 = linalg.fill ins(%cst_0 : bf16) outs(%9 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
    %11 = linalg.max ins(%8, %10 : tensor<128x1024xbf16>, tensor<128x1024xbf16>) outs(%9 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
    bufferization.materialize_in_destination %11 in restrict writable %arg3 : (tensor<128x1024xbf16>, memref<128x1024xbf16>) -> ()
    return
  }
}
