// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_op.hpp"

#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

// TODO: Prune unused headers -- it's hard to understand needed ones
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#ifdef TPP_MLIR // If TPP is available
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"
#endif

#ifdef GRAPH_COMPILER
#include "gc/Transforms/Passes.h"
#endif

namespace {

using namespace mlir;

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;

void prepareMLIRKernelWithoutWrapper(mlir::OwningOpRef<mlir::ModuleOp>& module, ov::mlir::MlirMode mode) {
    PassManager pm(module->getContext());

    switch (mode) {
#ifdef TPP_MLIR
        case ov::mlir::MLIR_MODE_TPP: {
            tpp::DefaultPipelineOptions defPipelineOpts;
            pm.addPass(tpp::createDefaultPipeline(defPipelineOpts));
            break;
        }
#endif
#ifdef GRAPH_COMPILER
        case ov::mlir::MLIR_MODE_GC: {
            gc::populateCPUPipeline(pm);
            break;
        }
#endif
        default: {
            assert(ov::mlir::MLIR_MODE_DEFAULT);
            // Cleanup before bufferization.
            // Simplifies IR to allow better bufferization.
            pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
            pm.addNestedPass<func::FuncOp>(createCSEPass());

            // Remove empty tensors to avoid converting them into temporary buffers.
            pm.addPass(bufferization::createEmptyTensorEliminationPass());

            pm.addPass(bufferization::createOneShotBufferizePass());
            pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());

            // Cleanup after bufferization - possibly remove redundant copies.
            pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
            pm.addNestedPass<func::FuncOp>(createCSEPass());

            // Deallocation pipeline to avoid memory leaks from created temporary buffers.
            pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
            pm.addPass(createCanonicalizerPass());
            bufferization::DeallocationOptions deallocOpts;
            deallocOpts.privateFuncDynamicOwnership = false;
            pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass(deallocOpts));
            pm.addPass(createCanonicalizerPass());
            pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
            pm.addPass(bufferization::createLowerDeallocationsPass());
            pm.addPass(createCSEPass());
            pm.addPass(createCanonicalizerPass());

            // Blanket-convert any remaining high-level vector ops to loops if any remain.
            pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
            // pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
            //  Blanket-convert any remaining linalg ops to loops if any remain.
            pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
            // Blanket-convert any remaining affine ops if any remain.
            pm.addPass(createLowerAffinePass());
            // Convert SCF to CF (always needed).
            pm.addPass(createConvertSCFToCFPass());
            // Sprinkle some cleanups.
            pm.addPass(createCanonicalizerPass());
            pm.addPass(createCSEPass());
            // Blanket-convert any remaining linalg ops to LLVM if any remain.
            // pm.addPass(createConvertLinalgToLLVMPass());  // no such pass
            // Convert vector to LLVM (always needed).
            pm.addPass(createConvertVectorToLLVMPass());
            // Convert Math to LLVM (always needed).
            pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
            // Expand complicated MemRef operations before lowering them.
            pm.addPass(memref::createExpandStridedMetadataPass());
            // The expansion may create affine expressions. Get rid of them.
            pm.addPass(createLowerAffinePass());
            // Convert MemRef to LLVM (always needed).
            // pm.addPass(memref::createExpandOpsPass());
            pm.addPass(createFinalizeMemRefToLLVMConversionPass());
            // Convert Func to LLVM (always needed).
            pm.addPass(createConvertFuncToLLVMPass());
            // Convert Index to LLVM (always needed).
            pm.addPass(createConvertIndexToLLVMPass());
            // Convert remaining unrealized_casts (always needed).
            pm.addPass(createReconcileUnrealizedCastsPass());
        }
    }

    auto result = pm.run(module.get());
    if (failed(result)) {
        llvm::errs() << "ERROR: Failed to lower IR to LLVM dialect\n";
        module->print(llvm::errs());
    }
}

std::unique_ptr<llvm::Module> lowerToLLVMIR(Operation* module, llvm::LLVMContext& llvmContext) {
    // Default lowering for mlir-cpu-runner
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    assert(llvmModule);

    // Target machine, null if not specified
    std::unique_ptr<llvm::TargetMachine> targetMachine;

    std::string triple = "x86_64-linux-gnu";
    std::string cpuName = "alderlake";  // sapphirerapids, nehalem, etc.
    std::string fpuName = "avx2";       //  sse4.2, avx, avx2, avx512bf16, etc.
    bool printLLVM = false;
    auto codeGenOpt = 2;

    // Specify target machine
    if (!triple.empty() && !cpuName.empty()) {
        std::string error;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target) {
            llvm::errs() << "Error while looking up target triple: ";
            llvm::errs() << error << "\n";
            return nullptr;
        }

        // These options should force fused MLA, but they don't. :/
        // Adding unsafe math attribute to functions below do the trick.
        llvm::TargetOptions targetOptions;
        targetOptions.UnsafeFPMath = true;
        targetOptions.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
        targetMachine.reset(target->createTargetMachine(triple,
                                                        cpuName,
                                                        "+" + fpuName,
                                                        targetOptions,
                                                        /* reloc model */ std::nullopt,
                                                        /* code model */ std::nullopt,
                                                        llvm::CodeGenOptLevel(codeGenOpt)));
        if (!targetMachine) {
            llvm::errs() << "Error while looking up target CPU: ";
            llvm::errs() << cpuName << "\n";
            return nullptr;
        }
    }

    // Run the optimized pipeline
    int sizeLevel = 0;
    auto optPipeline = makeOptimizingTransformer(codeGenOpt, sizeLevel, targetMachine.get());
    if (auto err = optPipeline(llvmModule.get())) {
        llvmModule->print(llvm::errs(), nullptr);
        llvm::errs() << "Error while passing through the LLVM pipeline: ";
        llvm::errs() << err << "\n";
        return nullptr;
    }

    // MLIR doesn't lower LLVM with fast-math flags, but we need that, so we
    // add for each function, to get FMAs and other goodies.
    for (auto& func : llvmModule->functions()) {
        func.addFnAttr("unsafe-fp-math", "true");
    }

    if (printLLVM)
        llvmModule->print(llvm::outs(), nullptr);

    return llvmModule;
}

// TODO: u4/i4 types are not supported
struct MemRefDescriptor {
    MemRefDescriptor() = default;

    MemRefDescriptor    (ov::Tensor tensor)
        : allocated(tensor.data()),
          aligned(tensor.data()),
          offset(0),
          shape(tensor.get_shape().begin(), tensor.get_shape().end()) {
        strides.resize(tensor.get_shape().size());
        const auto& byte_strides = tensor.get_strides();
        auto element_size = tensor.get_element_type().size();
        for (size_t i = 0; i < strides.size(); ++i) {
            assert(byte_strides[i] % element_size == 0);
            // TODO: handle case when stride is not aligned (restrict at OV API level)
            strides[i] = byte_strides[i] / element_size;
            //std::cerr << "stride [" << i << "] = " << strides[i] << "\n";
        }
    }

    MemRefDescriptor(ov::mlir::CachedBuffer buffer)
        : allocated(buffer.buffer),
          aligned(buffer.buffer),
          offset(0),
          shape(buffer.shape),
          strides(buffer.strides) {}

    void* allocated;
    void* aligned;
    int64_t offset;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    void append_to_packed_args(std::vector<void*>& args) {
        args.push_back(&allocated);
        args.push_back(&aligned);
        args.push_back(&offset);
        for (size_t i = 0; i < shape.size(); ++i) {
            args.push_back(&shape[i]);
        }
        for (size_t i = 0; i < strides.size(); ++i) {
            args.push_back(&strides[i]);
        }
    }
};

} // namespace

namespace ov {
namespace mlir {

using namespace ::mlir;

static std::unordered_set<const MLIROp *> executed_ops;

void MLIREvaluate::set_folding_info() {
    {
        auto expectArgs = engine->lookup("__num_orig_args");
        if (!expectArgs) {
            llvm::consumeError(expectArgs.takeError());
            return;
        }
        folding_info.num_orig_args = *reinterpret_cast<int32_t*>(*expectArgs);
    }

    {
        auto expectFold = engine->lookupPacked(defaultFoldName);
        if (!expectFold) {
            llvm::consumeError(expectFold.takeError());
            return;
        }
        folding_info.fold_func = *expectFold;
    }

    {
        auto expectBufferIds = engine->lookup("__runtime_fold_buffer_ids");
        if (!expectBufferIds) {
            llvm::consumeError(expectBufferIds.takeError());
            return;
        }
        auto raw = reinterpret_cast<int64_t*>(*expectBufferIds);
        folding_info.fold_buffer_ids = llvm::ArrayRef<int64_t>{raw + 1, raw[0]};
    }

    {
        auto expectFold = engine->lookup("__fold_args");
        if (!expectFold) {
            llvm::consumeError(expectFold.takeError());
            return;
        }
        auto raw = reinterpret_cast<int32_t*>(*expectFold);
        folding_info.fold_args = llvm::ArrayRef<int32_t>{raw + 1, raw[0]};
    }

    {
        auto expect = engine->lookup("__compute_args");
        if (!expect) {
            llvm::consumeError(expect.takeError());
            return;
        }
        auto raw = reinterpret_cast<int32_t*>(*expect);
        folding_info.compute_args = llvm::ArrayRef<int32_t>{raw + 1, raw[0]};
    }

    {
        auto expect = engine->lookup("__folded_ranks");
        if (!expect) {
            llvm::consumeError(expect.takeError());
            return;
        }
        auto raw = reinterpret_cast<int32_t*>(*expect);
        folding_info.folded_ranks = llvm::ArrayRef<int32_t>{raw, folding_info.fold_buffer_ids.size()};
    }

    {
        auto expect = engine->lookup("__folded_shapes");
        if (!expect) {
            llvm::consumeError(expect.takeError());
            return;
        }
        int32_t size = folding_info.fold_buffer_ids.size();  // element bytes of each buffer
        for (auto r : folding_info.folded_ranks) {
            size += r;
        }
        auto raw = reinterpret_cast<int64_t*>(*expect);
        llvm::ArrayRef<int64_t> folded_shapes = llvm::ArrayRef<int64_t>{raw, size};
        int pos = 0;
        for (int i = 0; i < folding_info.folded_ranks.size(); ++i) {
            std::vector<int64_t> shape(folded_shapes.begin() + pos,
                                       folded_shapes.begin() + pos + folding_info.folded_ranks[i] + 1);
            pos += folding_info.folded_ranks[i] + 1;
            folding_info.folded_shapes.push_back(shape);
        }
    }

    for (auto id : folding_info.fold_buffer_ids) {
        std::vector<int64_t> shape = folding_info.folded_shapes[id];
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        shape.pop_back();  // delete the last which is bytes of element
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = strides.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        void* buffer = std::aligned_alloc(/*alignment*/ 64, size);
        cached_const_buffers[id] = CachedBuffer{buffer, shape, strides};
    }
}

MLIREvaluate::MLIREvaluate(OwningOpRef<mlir::ModuleOp> _module, MlirMode mode) :
    module(std::move(_module)) {

    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Source MLIR:\n"
        "-----------------------------------------\n");
    OPENVINO_MLIR_DEBUG(module->dump());
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");

    prepareMLIRKernelWithoutWrapper(module, mode);

    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Target LLVM:\n"
        "-----------------------------------------\n");
    OPENVINO_MLIR_DEBUG(module->dump());
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");

    auto optPipeline = mlir::makeOptimizingTransformer(2,
                                                        /*sizeLevel=*/0,  // FIXME: HARDCODED
                                                        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;  // opt level looks to be overriden in lowerToLLVMIR, but is still used
                                                // in `create` independently
    engineOptions.llvmModuleBuilder = lowerToLLVMIR;
    auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (maybeEngine) {
        engine = std::move(maybeEngine.get());
    } else {
        llvm::errs() << "failed to construct an execution engine\n";
        abort();
    }

    set_folding_info();
}

MLIREvaluate::~MLIREvaluate() {
    for (auto pair : cached_const_buffers) {
        std::free(pair.second.buffer);
    }
}

bool MLIREvaluate::invoke_packed(std::vector<void*>& args) {
    auto invocationResult = engine->invokePacked("entry", args);
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return false;
    }
    return true;
}

MLIROp::MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluate> engine, const OVOutputTypes& output_types, const DimensionsMap& dimensions_map)
    : Op(args),
        engine(engine),
        output_types(output_types),
        dimensions_map(dimensions_map) {
    constructor_validate_and_infer_types();
}

void MLIROp::validate_and_infer_types() {
    set_output_size(output_types.size());
    for (size_t i = 0; i < output_types.size(); ++i) {
        set_output_type(i, std::get<0>(output_types[i]), std::get<1>(output_types[i]));
    }
}

NodePtr MLIROp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<MLIROp>(new_args, engine, output_types, dimensions_map);
}

bool MLIROp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] input size: " << inputs.size() << ", output size: " << outputs.size() << "\n");
    if (engine->folding_info.fold_func == nullptr) {  // No folding, call entry() directly
        std::vector<MemRefDescriptor> memref_args;
        for (size_t i = 0; i < inputs.size(); ++i) {
            memref_args.push_back(MemRefDescriptor(inputs[i]));
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            // TODO: Optimize by adding all dimensions to dimensions_map, not only dynamic
            Shape target;
            PartialShape expected = get_output_partial_shape(i);
            for (size_t j = 0; j < expected.size(); ++j) {
                auto dim = expected[j];
                if (dim.is_dynamic()) {
                    int input_index, dim_index;
                    std::tie(input_index, dim_index) = dimensions_map[i][j];
                    target.push_back(inputs[input_index].get_shape()[dim_index]);
                } else {
                    target.push_back(dim.get_length());
                }
            }
            // std::cerr << "[ DEBUG ] Set outputs[" << i << "].shape(" << target << ")\n";
            outputs[i].set_shape(target);
            memref_args.push_back(MemRefDescriptor(outputs[i]));
        }

        std::vector<void*> args;
        std::for_each(memref_args.begin(), memref_args.end(), [&args](MemRefDescriptor& x) {
            x.append_to_packed_args(args);
        });

        OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] Call entry func directly\n");
        return engine->invoke_packed(args);
    } else {                                  // call fold() first, then call entry()
        if (executed_ops.count(this) == 0) {  // Call fold()
            std::vector<MemRefDescriptor> memref_args;
            // Args of fold(): {constant inputs, folded inputs}.
            for (auto id : engine->folding_info.fold_args) {
                if (id < engine->folding_info.num_orig_args) {
                    memref_args.push_back(MemRefDescriptor(inputs[id]));
                } else {
                    int64_t buffer_id = id - engine->folding_info.num_orig_args;
                    assert(engine->cached_const_buffers.find(buffer_id) != engine->cached_const_buffers.end());
                    memref_args.push_back(MemRefDescriptor(engine->cached_const_buffers[buffer_id]));
                }
            }
            std::vector<void*> args;
            std::for_each(memref_args.begin(), memref_args.end(), [&args](MemRefDescriptor& x) {
                x.append_to_packed_args(args);
            });
            OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] First executon, call fold func\n");
            engine->folding_info.fold_func(args.data());

            // TODO: Find a better way to check if the op has executed.
            // This is a const function and can not modify member attributes directly.
            executed_ops.insert(this);
        }
        // call entry()
        std::vector<MemRefDescriptor> memref_args;
        // Args of entry(): {non-constant inputs, outputs, folded inputs}.
        for (auto id : engine->folding_info.compute_args) {
            // num_orig_args = inputs.size() + outputs.size()
            // if (id < engine->folding_info.num_orig_args) {
            if (id < inputs.size()) {  // non-constant input
                memref_args.push_back(MemRefDescriptor(inputs[id]));
            } else if (id < engine->folding_info.num_orig_args) {  // output
                int i = id - inputs.size();                        // output id
                Shape target;
                PartialShape expected = get_output_partial_shape(i);
                for (size_t j = 0; j < expected.size(); ++j) {
                    auto dim = expected[j];
                    if (dim.is_dynamic()) {
                        int input_index, dim_index;
                        std::tie(input_index, dim_index) = dimensions_map[i][j];
                        target.push_back(inputs[input_index].get_shape()[dim_index]);
                    } else {
                        target.push_back(dim.get_length());
                    }
                }
                // std::cerr << "[ DEBUG ] Set outputs[" << i << "].shape(" << target << ")\n";
                outputs[i].set_shape(target);
                memref_args.push_back(MemRefDescriptor(outputs[i]));
            } else {  // folded input
                int64_t buffer_id = id - engine->folding_info.num_orig_args;
                assert(engine->cached_const_buffers.find(buffer_id) != engine->cached_const_buffers.end());
                memref_args.push_back(MemRefDescriptor(engine->cached_const_buffers[buffer_id]));
            }
        }

        std::vector<void*> args;
        std::for_each(memref_args.begin(), memref_args.end(), [&args](MemRefDescriptor& x) {
            x.append_to_packed_args(args);
        });
        OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] Call entry func\n");
        // std::cerr << "[ INFO ] Running kernel in MLIROp::evaluate\n";
        return engine->invoke_packed(args);
    }
}

bool MLIROp::has_evaluate() const {
    return true;
}

} // namespace mlir
} // namespace ov