// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
class LLVMCPULowerToAccelUKernelsPass
    : public LLVMCPULowerToAccelUKernelsBase<LLVMCPULowerToAccelUKernelsPass> {
public:
  LLVMCPULowerToAccelUKernelsPass(bool skipIntermediateRoundings)
      : skipIntermediateRoundings(skipIntermediateRoundings) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // This option defaults to `true` both in Passes.td and in C++ code.
    // If either side has `false`, that's a non-default choice, so we let that
    // override a `true` on the other side.
    skipIntermediateRoundings &= optionSkipIntermediateRoundings;
    return success();
  }

private:
  bool skipIntermediateRoundings;
};
} // namespace

/// Returns `true` if an `outsOperand` value is initialized to zero.
static bool isInitializedToZero(Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return false;
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, m_Zero()) ||
         matchPattern(fillVal, m_AnyZeroFloat());
}

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  result.name = std::string("iree_uk_") + ukernelName;
  result.defAttrs.emplace_back(
    rewriter.getStringAttr("hal.import.fields"),
    rewriter.getArrayAttr({rewriter.getStringAttr("processor_data")}));
  result.defAttrs.emplace_back(rewriter.getStringAttr("hal.import.bitcode"),
                               rewriter.getBoolAttr(true));
  result.defAttrs.emplace_back(
      rewriter.getStringAttr("hal.import.cconv"),
      IREE::HAL::CallingConventionAttr::get(
          rewriter.getContext(),
          IREE::HAL::CallingConvention::ParameterStruct));
  }
  return result;
}

/// Matches an (linalg.fill -> )? linalg.matmul operation sequence and converts
/// it into a iree_codegen.ukernel.generic "aie_matmul_f32" operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::MatmulOp op,
                   bool skipIntermediateRoundings) {
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto lhsType = llvm::cast<ShapedType>(lhs.getType());
  auto rhsType = llvm::cast<ShapedType>(rhs.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    flags = 0;
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    flags = 1;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 0);
  Value k = rewriter.create<tensor::DimOp>(loc, rhs, 1);

  auto getDimAsI32 = [](RewriterBase &rewriter, Location loc, Value value,
                        int dim) -> Value {
    return rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, dim));
  };
  Value m0 = getDimAsI32(rewriter, loc, lhs, 2);
  Value n0 = getDimAsI32(rewriter, loc, rhs, 2);
  Value k0 = getDimAsI32(rewriter, loc, rhs, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("aie_matmul_f32", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, m0, n0, k0, flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static uint32_t flagForUser(IREE::LinalgExt::EncodingUser user) {
  switch (user) {
  case IREE::LinalgExt::EncodingUser::MATMUL_F32F32F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_I8I8I32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
  case IREE::LinalgExt::EncodingUser::MATMUL_F16F16F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_F16F16F16:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F16;
  case IREE::LinalgExt::EncodingUser::MATMUL_BF16BF16F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_BF16BF16BF16:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16BF16;
  default: // Unreachable.
    assert(false);
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_NONE;
  }
}

static uint32_t flagForRole(IREE::LinalgExt::EncodingRole role) {
  switch (role) {
  case IREE::LinalgExt::EncodingRole::LHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS;
  case IREE::LinalgExt::EncodingRole::RHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS;
  case IREE::LinalgExt::EncodingRole::RESULT:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT;
  default: // Unreachable.
    assert(false);
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS;
  }
}

namespace {

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;

template <typename OpType>
struct LowerToAccelUKernelPattern : OpRewritePattern<OpType> {
  LowerToAccelUKernelPattern(MLIRContext *context, TargetPredicate targetPredicate,
                        bool skipIntermediateRoundings = false)
      : OpRewritePattern<OpType>(context), targetPredicate(targetPredicate),
        skipIntermediateRoundings(skipIntermediateRoundings) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchDAGForUKernel(rewriter, op, skipIntermediateRoundings);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  bool skipIntermediateRoundings;
};

} // namespace

void LLVMCPULowerToAccelUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.

  // These patterns are unconditionally enabled, because we have strong evidence
  // that it is difficult for codegen to consistently approach microkernels
  // performance, and that consideration overrides the benefit of fusions for
  // these ops.
  auto allTargets = [](auto target) { return true; };
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(
      context, allTargets, skipIntermediateRoundings);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>>
createLLVMCPULowerToAccelUKernelsPass(bool skipIntermediateRoundings) {
  return std::make_unique<LLVMCPULowerToAccelUKernelsPass>(
      skipIntermediateRoundings);
}

} // namespace iree_compiler
} // namespace mlir
