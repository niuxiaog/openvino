// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "openvino/op/op.hpp"

#include "convert_common.hpp"


namespace ov {
namespace mlir {

using ::mlir::OwningOpRef;
using ::mlir::ModuleOp;
using ::mlir::ExecutionEngine;
using ::mlir::ModuleOp;

enum MlirMode {
    MLIR_MODE_TPP,
    MLIR_MODE_GC,
    MLIR_MODE_DEFAULT,
};

using JitModuleFuncT = void (*)(void**);
static const char defaultFoldName[] = "runtime_fold";

struct FoldingInfo {
    int num_orig_args;
    llvm::ArrayRef<int> fold_args;
    llvm::ArrayRef<int> compute_args;
    llvm::ArrayRef<int64_t> fold_buffer_ids;
    JitModuleFuncT fold_func = nullptr;
};

struct CachedBuffer {
    void* buffer;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

class MLIREvaluate {
    OwningOpRef<ModuleOp> module;  // FIXME: needs to be kept?
    std::unique_ptr<ExecutionEngine> engine;

    void set_folding_info();
public:

    MLIREvaluate(OwningOpRef<ModuleOp> _module, MlirMode mode);
    ~MLIREvaluate();
    bool invoke_packed(std::vector<void*>& args);
    FoldingInfo folding_info;
    std::unordered_map<int64_t, CachedBuffer> cached_const_buffers;
};


// Maps [output index][dimension index] -> [input index][dimension index] to infer shapes for entire subgraph
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;


class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluate> engine;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;

public:

    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluate> engine, const OVOutputTypes& output_types, const DimensionsMap& dimensions_map);
    void validate_and_infer_types() override;
    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

} // namespace mlir
} // namespace ov