// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

// Computes a reduction along the rows of a 2d tensor of shape MxN
// to produce a tensor of shape M
template <typename T>
static Value computeRowwiseReduction(Value a, Value output, Location loc,
                                     OpBuilder &builder,
                                     SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::reduction};
  AffineMap id = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{id, rowMap};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), a, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computePartialSoftmax(Value qkTranspose, Value currentMax,
                                   Location loc, OpBuilder &builder,
                                   SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qkTranspose.getType(), ValueRange{currentMax}, qkTranspose,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[1], args[0]);
        Value result = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value updateAndScale(Value oldMax, Value newMax, Value oldSum,
                            Location loc, OpBuilder &builder,
                            SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(3, identityMap);
  SmallVector<Type> resultTypes{oldSum.getType()};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, resultTypes, ValueRange{oldMax, newMax}, ValueRange{oldSum},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value weight = b.create<math::Exp2Op>(loc, diff);
        Value scaledOldSum = b.create<arith::MulFOp>(loc, weight, args[2]);
        b.create<linalg::YieldOp>(loc, ValueRange{scaledOldSum});
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scalePartialSoftmax(Value softmax, Value inverseNewSum,
                                 Location loc, OpBuilder &builder,
                                 SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, softmax.getType(), ValueRange{inverseNewSum}, softmax, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[1], args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeReciprocal(Value x, Location loc, OpBuilder &builder,
                               SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, x.getType(), ValueRange{}, x, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value result = b.create<arith::DivFOp>(loc, one, args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scaleAccumulator(Value accumulator, Value scaledOldSum,
                              Value inverseNewSum, Location loc,
                              OpBuilder &builder,
                              SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, accumulator.getType(), ValueRange{scaledOldSum, inverseNewSum},
      accumulator, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value ratio = b.create<arith::MulFOp>(loc, args[0], args[1]);
        Value result = b.create<arith::MulFOp>(loc, ratio, args[2]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeQKTranspose(Value query, Value key, Value output,
                                Value zero, Location loc, OpBuilder &builder,
                                SmallVectorImpl<Operation *> &ops) {
  auto fillOp = builder.create<linalg::FillOp>(loc, ValueRange{zero}, output);
  ops.push_back(fillOp);
  Value acc = fillOp.result();
  auto matmulOp = builder.create<linalg::MatmulTransposeBOp>(
      loc, output.getType(), ValueRange{query, key}, acc);
  ops.push_back(matmulOp);
  return matmulOp.getResult(0);
}

static std::tuple<Value, Value, Value>
extractSlices(Value key, Value value, Value query, ArrayRef<int64_t> queryShape,
              ArrayRef<Value> ivs, OpFoldResult sequenceTileLength,
              OpFoldResult headDimension, Type elementType, Location loc,
              OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = sequenceTileLength;
  sizes[2] = headDimension;
  offsets[1] = ivs[0];
  SmallVector<int64_t> tensorShape{queryShape[1], queryShape[2]};
  auto tensorType = RankedTensorType::get(tensorShape, elementType);
  Value keySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, key, offsets, sizes, strides);
  Value valueSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, value, offsets, sizes, strides);

  offsets = SmallVector<OpFoldResult>(queryShape.size(), zero);
  Value querySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, query, offsets, sizes, strides);
  return std::make_tuple(keySlice, valueSlice, querySlice);
}

static scf::LoopNest createLoopNest(SmallVectorImpl<Value> &ivs, Value lb,
                                    Value step, Value ub, ValueRange args,
                                    Location loc, OpBuilder &builder) {
  SmallVector<Value> lbs{lb};
  SmallVector<Value> steps{step};
  SmallVector<Value> ubs{ub};
  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, args,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  for (scf::ForOp loop : loopNest.loops)
    ivs.push_back(loop.getInductionVar());
  return loopNest;
}

static Value truncateToF16(Value input, Value output,
                           SmallVectorImpl<Operation *> &ops,
                           OpBuilder &builder, Location loc) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::TruncFOp>(loc, b.getF16Type(), args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static std::tuple<Value, Value, Value>
createAttentionBody(Value keySlice, Value valueSlice, Value querySlice,
                    Value outputSlice, Value maxSlice, Value sumSlice,
                    OpFoldResult sequenceTileLength, OpFoldResult headDimension,
                    Type elementType, SmallVectorImpl<Operation *> &ops,
                    Location loc, OpBuilder &builder) {

  Type f32Type = builder.getF32Type();
  // Compute matmul(q, transpose(k))
  Value zero =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(f32Type));
  SmallVector<OpFoldResult> resultShape{sequenceTileLength, sequenceTileLength};
  Value emptySquare =
      builder.create<tensor::EmptyOp>(loc, resultShape, f32Type);
  Value qkTranspose = computeQKTranspose(querySlice, keySlice, emptySquare,
                                         zero, loc, builder, ops);

  // Compute current statistics
  Value newMax = computeRowwiseReduction<arith::MaximumFOp>(
      qkTranspose, maxSlice, loc, builder, ops);
  Value partialSoftmax =
      computePartialSoftmax(qkTranspose, newMax, loc, builder, ops);
  Value scaledOldSum =
      updateAndScale(maxSlice, newMax, sumSlice, loc, builder, ops);
  Value newSum = computeRowwiseReduction<arith::AddFOp>(
      partialSoftmax, scaledOldSum, loc, builder, ops);
  Value inverseNewSum = computeReciprocal(newSum, loc, builder, ops);
  Value softmax =
      scalePartialSoftmax(partialSoftmax, inverseNewSum, loc, builder, ops);
  if (elementType.isF16()) {
    Value empty =
        builder.create<tensor::EmptyOp>(loc, resultShape, builder.getF16Type());
    softmax = truncateToF16(softmax, empty, ops, builder, loc);
  }

  // Update accumulator
  Value scaledAcc = scaleAccumulator(outputSlice, scaledOldSum, inverseNewSum,
                                     loc, builder, ops);

  // Compute matmul(softmax, v)
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, scaledAcc.getType(), ValueRange{softmax, valueSlice}, scaledAcc);
  ops.push_back(matmulOp);
  Value result = matmulOp.getResult(0);
  return std::make_tuple(result, newMax, newSum);
}

static Value extractOrInsertOutputSlice(Value src, Value dst,
                                        ArrayRef<int64_t> queryShape,
                                        OpFoldResult sequenceTileLength,
                                        OpFoldResult headDimension,
                                        Location loc, OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(3, one);
  SmallVector<OpFoldResult> sizes = {one, sequenceTileLength, headDimension};
  SmallVector<OpFoldResult> offsets(3, zero);
  Value slice;
  if (!dst) {
    SmallVector<int64_t> accShape{queryShape[1], queryShape[2]};
    Type elementType = src.getType().cast<ShapedType>().getElementType();
    auto tensorType = RankedTensorType::get(accShape, elementType);
    slice = builder.create<tensor::ExtractSliceOp>(loc, tensorType, src,
                                                   offsets, sizes, strides);
  } else {
    slice = builder.create<tensor::InsertSliceOp>(loc, src, dst, offsets, sizes,
                                                  strides);
  }
  return slice;
}

static Value extractOutputSlice(Value src, ArrayRef<int64_t> queryShape,
                                OpFoldResult sequenceTileLength,
                                OpFoldResult headDimension, Location loc,
                                OpBuilder &builder) {
  return extractOrInsertOutputSlice(src, {}, queryShape, sequenceTileLength,
                                    headDimension, loc, builder);
}

static Value insertOutputSlice(Value src, Value dst,
                               OpFoldResult sequenceTileLength,
                               OpFoldResult headDimension, Location loc,
                               OpBuilder &builder) {
  return extractOrInsertOutputSlice(src, dst, {}, sequenceTileLength,
                                    headDimension, loc, builder);
}

} // namespace

SmallVector<Operation *>
tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                          RewriterBase &rewriter) {
  SmallVector<Operation *> ops;
  Location loc = attnOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(attnOp);

  Value query = attnOp.getQuery();
  ShapedType queryType = attnOp.getQueryType();
  Type elementType = queryType.getElementType();
  ArrayRef<int64_t> queryShape = queryType.getShape();
  SmallVector<OpFoldResult> queryDimValues =
      tensor::getMixedSizes(rewriter, loc, query);
  OpFoldResult headDimension = queryDimValues[2];
  OpFoldResult sequenceTileLength = queryDimValues[1];

  Value key = attnOp.getKey();
  Value value = attnOp.getValue();
  SmallVector<OpFoldResult> keyDimValues =
      tensor::getMixedSizes(rewriter, loc, key);
  OpFoldResult sequenceLength = keyDimValues[1];

  // Create output accumulator
  Value output = attnOp.getOutput();
  Type f32Type = rewriter.getF32Type();
  SmallVector<OpFoldResult> accShape{queryDimValues[1], queryDimValues[2]};
  Value accumulatorF32 =
      rewriter.create<tensor::EmptyOp>(loc, accShape, f32Type);

  // Create accumulator, max and sum statistics
  Value outputSlice = extractOutputSlice(output, queryShape, sequenceTileLength,
                                         headDimension, loc, rewriter);
  Value zeroF32 =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(f32Type));
  auto accumulatorFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, accumulatorF32);
  accumulatorF32 = accumulatorFill.result();
  ops.push_back(accumulatorFill);

  Value largeNegativeF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(f32Type, -1.0e+30));
  SmallVector<OpFoldResult> dims{sequenceTileLength};
  Value max = rewriter.create<tensor::EmptyOp>(loc, dims, f32Type);
  auto maxFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{largeNegativeF32}, max);
  Value negativeMax = maxFill.result();
  ops.push_back(maxFill);
  Value sum = rewriter.create<tensor::EmptyOp>(loc, dims, f32Type);
  auto sumFill = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, sum);
  Value zeroSum = sumFill.result();
  ops.push_back(sumFill);

  // Construct sequential loop
  SmallVector<Value> ivs;
  Value zeroValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  scf::LoopNest loopNest = createLoopNest(
      ivs, zeroValue,
      getValueOrCreateConstantIndexOp(rewriter, loc, sequenceTileLength),
      getValueOrCreateConstantIndexOp(rewriter, loc, sequenceLength),
      ValueRange({accumulatorF32, negativeMax, zeroSum}), loc, rewriter);
  ops.push_back(loopNest.loops.back());

  Value iterArgResult = loopNest.loops.back().getRegionIterArg(0);
  Value iterArgMax = loopNest.loops.back().getRegionIterArg(1);
  Value iterArgSum = loopNest.loops.back().getRegionIterArg(2);

  OpBuilder::InsertionGuard guardSecondLoop(rewriter);
  rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());

  // Extract slices
  auto [keySlice, valueSlice, querySlice] =
      extractSlices(key, value, query, queryShape, ivs, sequenceTileLength,
                    headDimension, elementType, loc, rewriter);

  // Create body of innermost loop
  auto [result, newMax, newSum] = createAttentionBody(
      keySlice, valueSlice, querySlice, iterArgResult, iterArgMax, iterArgSum,
      sequenceTileLength, headDimension, elementType, ops, loc, rewriter);

  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        yieldOp, ValueRange{result, newMax, newSum});
  }

  OpBuilder::InsertionGuard yieldGuard(rewriter);
  rewriter.setInsertionPointAfter(loopNest.loops.back());
  if (elementType.isF16()) {
    loopNest.results[0] =
        truncateToF16(loopNest.results[0], outputSlice, ops, rewriter, loc);
  }
  loopNest.results[0] =
      insertOutputSlice(loopNest.results[0], output, sequenceTileLength,
                        headDimension, loc, rewriter);

  attnOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);
  return ops;
}

namespace {

/// This is an implementation of flash attention which
/// is a tiled and fused implementation of the attention operator.
/// The attention operator computes:
/// matmul(softmax(matmul(Q, transpose(K))), V)
/// where: Q is the query matrix [B x N x d]
///        K is the key matrix   [B x S x d]
///        V is the value matrix [B x S x d]
///
/// The core algorithm is as follows:
/// For each element in B,
/// 1. Load a tile from the Q matrix of size T x d -> q
/// 2. Initialize statistics: running_sum, running_max
/// 3. for i = 0 to S with step T
///    a. Load a tile from the K matrix of size T x d -> k
///    b. Load a tile from the V matrix of size T x d -> v
///    c. Compute matmul_transpose_b(q, k) -> qkT
///    d. Compute max(max(qkT) along rows, old_max) -> new_max
///    e. Compute curent estimate of softmax: exp(qKT - current_max) -> s
///    f. Compute product of fixup and old_sum -> fsum
///    g. Compute sum(sum(qkT) along rows, fsum) -> new_sum
///    h. Compute 1.0 / new_sum -> inv_new_sum
///    i. Compute softmax = softmax * inv_new_sum
///    j. Truncate softmax to fp16
///    k. Compute fsum  * inv_new_sum * accumulator -> new_accumulator
///    j. Compute matmul(s, v) and add new_accumulator
///
///
LogicalResult reifyAttentionTransform(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
    tileAndDecomposeAttention(attnOp, rewriter);
    return WalkResult::advance();
  });
  return success();
}

} // namespace

namespace {
struct TileAndDecomposeAttentionPass
    : public TileAndDecomposeAttentionBase<TileAndDecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  if (failed(reifyAttentionTransform(getOperation())))
    return signalPassFailure();
}

std::unique_ptr<Pass> createTileAndDecomposeAttentionPass() {
  return std::make_unique<TileAndDecomposeAttentionPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
