module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>, #dlti.dl_entry<"num_threads", 4 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : i32>, #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : i32>, #dlti.dl_entry<"L3_cache_size_in_bytes", 1966080 : i32>, #dlti.dl_entry<"max_vector_width", 512 : i32>>>} {
  func.func @entry(%arg0: memref<512x4096xbf16>, %arg1: memref<4096x4096xbf16>, %arg2: memref<512x4096xbf16>, %arg3: memref<512x4096xbf16>) attributes {onednn_graph.const_args_index = [0 : i32, 2 : i32]} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<512x4096xbf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xbf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<512x4096xbf16>
    %3 = tensor.empty() : tensor<512x4096xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<512x4096xbf16>) -> tensor<512x4096xbf16>
    %5 = linalg.matmul ins(%0, %1 : tensor<512x4096xbf16>, tensor<4096x4096xbf16>) outs(%4 : tensor<512x4096xbf16>) -> tensor<512x4096xbf16>
    %6 = tensor.empty() : tensor<512x4096xbf16>
    %7 = linalg.add ins(%5, %2 : tensor<512x4096xbf16>, tensor<512x4096xbf16>) outs(%6 : tensor<512x4096xbf16>) -> tensor<512x4096xbf16>
    %8 = tensor.empty() : tensor<512x4096xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %9 = linalg.fill ins(%cst_0 : bf16) outs(%8 : tensor<512x4096xbf16>) -> tensor<512x4096xbf16>
    %10 = linalg.max ins(%7, %9 : tensor<512x4096xbf16>, tensor<512x4096xbf16>) outs(%8 : tensor<512x4096xbf16>) -> tensor<512x4096xbf16>
    bufferization.materialize_in_destination %10 in restrict writable %arg3 : (tensor<512x4096xbf16>, memref<512x4096xbf16>) -> ()
    return
  }
}
