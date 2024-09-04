// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/matmul.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "matmul.hpp"
#include "../convert_common.hpp"


namespace {

using namespace ov::mlir;

struct ConvertMatMul {
    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(matmul_node);
        bool isTransposedA = matmul_node->get_transpose_a();
        bool isTransposedB = matmul_node->get_transpose_b();
        assert(!(isTransposedA && isTransposedB));

        const auto inputs = context.getInputs(node);
        mlir::SmallVector<Value, 2> ins{inputs[0]};

        if (isTransposedB) {
            auto shape = node->get_input_partial_shape(1);
            auto transposedShape = ov::PartialShape({shape[1], shape[0]});
            auto transposedType = importTensor(context.context, transposedShape, node->get_input_element_type(1));
            mlir::SmallVector<Value> dynamicDims = context.get_dynamic_dimension_values(transposedShape);;
            auto empty = builder.create<tensor::EmptyOp>(loc, transposedType, dynamicDims);
            auto transposeOp = builder.create<linalg::TransposeOp>(
                    loc, inputs[1], empty, mlir::SmallVector<int64_t>{1, 0});
            ins.push_back(transposeOp.getResult()[0]);
        } else {
            ins.push_back(inputs[1]);
        }

        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        // TODO: Support broadcasts
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = builder.create<linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});
        mlir::SmallVector<Value, 1> outs{fill.getResult(0)};

        Operation* matmul;
        if (isTransposedA) {
            matmul = builder.create<linalg::MatmulTransposeAOp>(loc, ins, outs);
        } else {
            matmul = builder.create<linalg::MatmulOp>(loc, ins, outs);
        }

        context.addOutputs(node, matmul);
    }
};

}

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

MatMulPattern::MatMulPattern() : MarkPattern(
    wrap_type<v0::MatMul>({any_input(), any_input()}, [](const Output<Node>& output) {
        auto node = std::dynamic_pointer_cast<v0::MatMul>(output.get_node_shared_ptr());
        assert(node);
        // FIXME: current code limitation
        return !has_dynamic_rank(node) && !(node->get_transpose_a() && node->get_transpose_b()) &&
               node->get_input_partial_shape(0).rank().get_length() == 2 &&
               node->get_input_partial_shape(1).rank().get_length() == 2;
    }),
    ConvertMatMul()) {
    }


} // namespace mlir
} // namespace ov