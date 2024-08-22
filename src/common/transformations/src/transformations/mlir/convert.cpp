// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mlir/convert.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/util/env_util.hpp>
#include <unordered_map>

// TODO: Prune unused headers -- it's hard to understand needed ones
#include "conversion_context.hpp"
#include "convert_common.hpp"
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
#include "mlir/Dialect/DLTI/DLTI.h"
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

#ifdef GRAPH_COMPILER
#include "gc/ExecutionEngine/Driver/Driver.h"
#endif

#ifdef TPP_MLIR // If TPP is available
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/GPU/Utils.h"
#endif

#include "mlir_op.hpp"
#include "op/matmul.hpp"
#include "op/relu.hpp"
#include "op/binary_eltwise.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "subgraph_tracker.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations_visibility.hpp"

namespace {

using namespace mlir;
using namespace ov::mlir;

MemRefType convertTensorToMemRef(TensorType tensorType) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    return MemRefType::get(shape, elementType);
}


SmallVector<mlir::Type> tensorsToMemRefs(SmallVector<mlir::Type> tensors) {
    SmallVector<mlir::Type> out;
    out.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        out.push_back(convertTensorToMemRef(dyn_cast<TensorType>(tensor)));
    }
    return out;
}


SmallVector<mlir::Type> get_types_for_values(mlir::MLIRContext* context, const ov::OutputVector& values) {
    SmallVector<mlir::Type> types;
    types.reserve(values.size());
    for (const auto& output : values) {
        types.push_back(importTensor(context, output.get_partial_shape(), output.get_element_type()));
    }
    return types;
}


mlir::OwningOpRef<mlir::ModuleOp> ngraph_to_mlir(MLIRContext* context,
                                                 const ov::OutputVector& inputs,
                                                 const ov::NodeVector& nodes,
                                                 const ov::OutputVector& outputs) {
    auto inputTypes = tensorsToMemRefs(get_types_for_values(context, inputs));
    auto outputTypes = tensorsToMemRefs(get_types_for_values(context, outputs));

    const auto moduleLoc = createLayerLocation(context, "module", "Module");
    auto module = mlir::ModuleOp::create(moduleLoc, StringRef("fragment_name"));

    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody() /* TODO: building log here */);

    const auto funcLoc = createLayerLocation(context, "entry", "Func");
    auto memref_args = inputTypes;
    memref_args.append(outputTypes);
    const auto funcType = mlir::FunctionType::get(context, ArrayRef(memref_args), ArrayRef(SmallVector<mlir::Type>()));
    auto func = moduleBuilder.create<mlir::func::FuncOp>(funcLoc, "entry", funcType);
    auto block_builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock() /* TODO: Add logger here */);

    // Affix target information attribute to the module to be used, at its discretion,
    // by the MLIR-compiler that consumes this module.
    auto tileSize = IntegerAttr::get(IntegerType::get(context, 32), 32);
    auto tileSizeKey = StringAttr::get(context, "tile_size");
    DataLayoutEntryInterface tileSizeEntry = DataLayoutEntryAttr::get(context, tileSizeKey, tileSize);

    int numThreadsInt = 1;
    if (char* numThreadsEnv = std::getenv("OMP_NUM_THREADS")) {
        numThreadsInt = std::atoi(numThreadsEnv);
    }
    auto numThreads = IntegerAttr::get(IntegerType::get(context, 32), numThreadsInt);
    auto numThreadsKey = StringAttr::get(context, "num_threads");
    DataLayoutEntryInterface numThreadsEntry = DataLayoutEntryAttr::get(context, numThreadsKey, numThreads);

    int L1CacheSizeInt = 49152;
    if (char* L1CacheSizeEnv = std::getenv("L1_CACHE_SIZE")) {
        L1CacheSizeInt = std::atoi(L1CacheSizeEnv);
    }
    auto L1CacheSize = IntegerAttr::get(IntegerType::get(context, 32), L1CacheSizeInt);
    auto L1CacheSizeKey = StringAttr::get(context, "L1_cache_size_in_bytes");
    DataLayoutEntryInterface L1CacheSizeEntry = DataLayoutEntryAttr::get(context, L1CacheSizeKey, L1CacheSize);

    int L2CacheSizeInt = 2097152;
    if (char* L2CacheSizeEnv = std::getenv("L2_CACHE_SIZE")) {
        L2CacheSizeInt = std::atoi(L2CacheSizeEnv);
    }
    auto L2CacheSize = IntegerAttr::get(IntegerType::get(context, 32), L2CacheSizeInt);
    auto L2CacheSizeKey = StringAttr::get(context, "L2_cache_size_in_bytes");
    DataLayoutEntryInterface L2CacheSizeEntry = DataLayoutEntryAttr::get(context, L2CacheSizeKey, L2CacheSize);

    int L3CacheSizeInt = 1966080;
    if (char* L3CacheSizeEnv = std::getenv("L3_CACHE_SIZE")) {
        L3CacheSizeInt = std::atoi(L3CacheSizeEnv);
    }
    auto L3CacheSize = IntegerAttr::get(IntegerType::get(context, 32), L3CacheSizeInt);
    auto L3CacheSizeKey = StringAttr::get(context, "L3_cache_size_in_bytes");
    DataLayoutEntryInterface L3CacheSizeEntry = DataLayoutEntryAttr::get(context, L3CacheSizeKey, L3CacheSize);

    int maxVectorWidthInt = 512;
    if (char* maxVectorWidthEnv = std::getenv("MAX_VECTOR_WIDTH")) {
        maxVectorWidthInt = std::atoi(maxVectorWidthEnv);
    }
    auto maxVectorWidth = IntegerAttr::get(IntegerType::get(context, 32), maxVectorWidthInt);
    auto maxVectorWidthKey = StringAttr::get(context, "max_vector_width");
    DataLayoutEntryInterface maxVectorWidthEntry = DataLayoutEntryAttr::get(context, maxVectorWidthKey, maxVectorWidth);

    TargetDeviceSpecInterface deviceSpec = TargetDeviceSpecAttr::get(context,
                                                                     ArrayRef({tileSizeEntry,
                                                                               numThreadsEntry,
                                                                               L1CacheSizeEntry,
                                                                               L2CacheSizeEntry,
                                                                               L3CacheSizeEntry,
                                                                               maxVectorWidthEntry}));
    auto deviceStr = StringAttr::get(context, "CPU");
    auto sysSpec = TargetSystemSpecAttr::get(context, ArrayRef(std::pair(deviceStr, deviceSpec)));
    module.getOperation()->setAttr("#dlti.sys_spec", sysSpec);

    module.getOperation()->getRegions().front().getBlocks().front().getOperations().front().setAttr(
        "onednn_graph.const_args_index",
        moduleBuilder.getI32ArrayAttr({0, 2}));

    ConversionContext conversion_context(context, &block_builder);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto funcInputVal = func.getArgument(i);
        // transition from memref enclosure to tensor interior
        auto loc = createLocation(context, inputs[i].get_node_shared_ptr());
        auto tensor = block_builder.create<bufferization::ToTensorOp>(loc, funcInputVal, /*restrict = */ true);
        conversion_context.nodeOutputMap.emplace(inputs[i], tensor);

        // FIXME: Avoid pre-population of dimension_map, take dimension values only if needed
        auto input_shape = inputs[i].get_partial_shape();
        auto input_rank = input_shape.rank();
        if(input_rank.is_static()) {
            for(size_t j = 0; j < input_rank.get_length(); ++j) {
                auto dim = input_shape[j];
                if(dim.is_dynamic()) {
                    auto symbol = dim.get_symbol();
                    assert(symbol);
                    symbol = ov::symbol::ancestor_of(symbol);
                    if(dim.is_dynamic() && !conversion_context.dimension_map.count(symbol)) {
                        auto dimSize = block_builder.create<tensor::DimOp>(loc, tensor, j);
                        conversion_context.dimension_map[symbol] = dimSize;
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto node = nodes[i];
        conversion_context.convert(node);
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = conversion_context.nodeOutputMap.at(outputs[i]);
        auto memref = func.getArgument(i + inputs.size());
        auto loc = createLocation(context, outputs[i].get_node_shared_ptr());
        // Ensure the result is stored in the provided function argument.
        // Mark as restrict to avoid temporary buffer and copy.
        // Mark as writable to ensure the output can be written to the buffer.
        block_builder.create<bufferization::MaterializeInDestinationOp>(loc,
                                                                        TypeRange{},
                                                                        tensor,
                                                                        memref,
                                                                        /*restrict=*/true,
                                                                        /*writable=*/true);
    }

    const auto retLoc = createLayerLocation(context, "output", "Output");
    block_builder.create<mlir::func::ReturnOp>(retLoc, ArrayRef(SmallVector<Value>()));

    return module;
}


// This pass converts a group of nodes into a single MLIROp
NodePtr ngraph_to_mlir_op(MLIRContext* context, SubgraphPtr subgraph, MlirMode mode) {
    mlir::OwningOpRef<mlir::ModuleOp> module = ngraph_to_mlir(context, subgraph->inputs, subgraph->nodes, subgraph->outputs);

    const auto& inputs = subgraph->inputs;
    using Index = DimensionsMap::value_type::value_type;
    std::map<SymbolPtr, Index> input_map;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        auto shape = input.get_partial_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            auto dim = shape[j];
            if(shape[j].is_dynamic()) {
                auto symbol = ov::symbol::ancestor_of(dim.get_symbol());
                if(0 == input_map.count(symbol)) {
                    input_map[symbol] = Index(i, j);
                } else {
                    OPENVINO_MLIR_DEBUG_PRINT(
                        "[ DEBUG ] Lost equality constraint for dimensions in output " << input << ".\n" <<
                        "          If the constraint is violated in runtime it will result in the undefined behaviour.\n");
                }
            }
        }
    }

    std::tuple<size_t, size_t> empty(-1, -1);
    const auto& outputs = subgraph->outputs;
    OVOutputTypes output_types;
    DimensionsMap output_map;
    output_map.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        auto shape = output.get_partial_shape();
        output_types.push_back(
            std::make_tuple(output.get_element_type(), shape));
        DimensionsMap::value_type dm;
        dm.reserve(shape.size());
        for (size_t j = 0; j < shape.size(); ++j) {
            auto dim = shape[j];
            dm.push_back(dim.is_dynamic() ? input_map.at(ov::symbol::ancestor_of(dim.get_symbol())) : empty);
        }
        output_map.emplace_back(dm);
    }
    return std::make_shared<MLIROp>(
        subgraph->inputs,
        std::make_shared<MLIREvaluate>(std::move(module), mode),
        output_types,
        output_map
    );
};


void replace_subgraph(SubgraphPtr subgraph, NodePtr node) {
    const auto& output_consumers = subgraph->output_consumers;
    assert(output_consumers.size() == node->get_output_size());
    for(size_t i = 0; i < node->get_output_size(); ++i) {
        auto replacement = node->output(i);
        for(auto consumer: output_consumers[i]) {
            consumer.replace_source_output(replacement);
        }
    }
}


class Partitioner : public ov::pass::ModelPass {
    MLIRContext* context;
    MlirMode mode;
public:
    OPENVINO_RTTI("Partitioner");

    Partitioner(MLIRContext* context, MlirMode mode) :
        context(context),
        mode(mode)
    {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        SubgraphTracker tracker([this](SubgraphPtr subgraph) {
                auto mlir_op = ngraph_to_mlir_op(context, subgraph, mode);
                replace_subgraph(subgraph, mlir_op);
                OPENVINO_MLIR_DEBUG_PRINT("Created MLIR op: " << mlir_op << "\n");
            }
        );
        for(auto node: model->get_ordered_ops()) {
            tracker.add_node(node, get_subgraph_mark(node));
        }
        tracker.finalize();
        return true;
    }
};


void injectMLIR(std::shared_ptr<ov::Model> model, MLIRContext* context, MlirMode mode) {
    ov::pass::Manager manager;
    using namespace ov::op;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::SymbolicPropagation>();
    manager.register_pass<BinaryEltwisePattern<v1::Add, linalg::AddOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Subtract, linalg::SubOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Multiply, linalg::MulOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Divide, linalg::DivOp>>();
    manager.register_pass<ReluPattern>();
    manager.register_pass<MatMulPattern>();
    manager.register_pass<Partitioner>(context, mode);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

void loadDialects(MLIRContext* context) {
    context->loadDialect<mlir::DLTIDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::linalg::LinalgDialect>();
    context->loadDialect<mlir::bufferization::BufferizationDialect>();
}

MLIRContext* get_shared_mlir_context(MlirMode mode) {
    // Gives MLIRContext instance shared for entire OV process and initialized once upon the initial request
    // FIXME: Bind with OpenVINO lifetime in the sutable class instead of dirty tricking with static lifetime

    static std::shared_ptr<MLIRContext> context;
    static bool current_mode = mode;

    if (context) {
        if (current_mode != mode) {
            OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] Switching MLIR mode to: ");
            current_mode = mode;
        } else {
            return context.get();
        }
    } else {
        OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] MLIR mode: ");
    }

#ifdef GRAPH_COMPILER
    if (mode == MLIR_MODE_GC) {
        OPENVINO_MLIR_DEBUG_PRINT("GC\n");
        context = std::make_shared<MLIRContext>(gc::initCompilerAndGetDialects());
    } else {
#endif
        // Initialize the LLVM machinery
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
#ifdef TPP_MLIR
        if (mode == MLIR_MODE_TPP) {
            OPENVINO_MLIR_DEBUG_PRINT("TPP\n");
            // Initialize GPU-related LLVM machinery
            tpp::initializeGpuTargets();
        } else {
#endif
            assert(mode == MLIR_MODE_DEFAULT);
            OPENVINO_MLIR_DEBUG_PRINT("DEFAULT\n");
#ifdef TPP_MLIR
            }
#endif

        // Add the following to include *all* MLIR Core dialects, or selectively
        // include what you need like above. You only need to register dialects that
        // will be *parsed* by the tool, not the one generated
        DialectRegistry registry;
#ifdef TPP_MLIR
        if (mode == MLIR_MODE_TPP) {
            registry.insert<mlir::xsmm::XsmmDialect>();
            registry.insert<mlir::check::CheckDialect>();
            registry.insert<mlir::perf::PerfDialect>();
        }
#endif

        registerAllDialects(registry);
        registerAllExtensions(registry);
        registerAllToLLVMIRTranslations(registry);
        mlir::linalg::registerTransformDialectExtension(registry);
        mlir::tensor::registerTransformDialectExtension(registry);

        context = std::make_shared<MLIRContext>(registry);

#ifdef GRAPH_COMPILER
    }
#endif

    loadDialects(context.get());
    return context.get();
}

} // namespace

void ov::pass::transformMLIR(std::shared_ptr<ov::Model> model) {
    if(util::getenv_bool("OV_MLIR", true)) {
        const char *default_mode =
#ifdef TPP_MLIR
                "TPP";
#elif defined(GRAPH_COMPILER)
                "GC";
#else
                "DEFAULT";
#endif
        auto mode_str = util::getenv_string("OV_MLIR_MODE");

        if (mode_str == "") {
            mode_str = default_mode;
        } else {
            // Convert to uppercase
            std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
        }

        MlirMode mode;

        if (mode_str == "TPP") {
#ifndef TPP_MLIR
            OPENVINO_THROW(
                "[ ERROR ] OpenVINO wasn't compiled with TPP_MLIR support, "
                "but OV_MLIR_MODE environment variable is set to TPP.");
#endif
            mode = MLIR_MODE_TPP;
        } else if (mode_str == "GC") {
#ifndef GRAPH_COMPILER
            OPENVINO_THROW(
                "[ ERROR ] OpenVINO wasn't compiled with GRAPH_COMPILER support, "
                "but OV_MLIR_MODE environment variable is set to GC.");
#endif
            mode = MLIR_MODE_GC;
        } else {
            OPENVINO_ASSERT(mode_str == "DEFAULT");
            mode = MLIR_MODE_DEFAULT;
        }

        injectMLIR(model, get_shared_mlir_context(mode), mode);
    }
}
