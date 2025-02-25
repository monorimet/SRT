// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMCPU_PASSES
#define IREE_CODEGEN_LLVMCPU_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// LLVMCPU Passes (keep alphabetical)
//------------------------------------------------------------------------------

def ConvertToLLVM :
    Pass<"iree-convert-to-llvm", "ModuleOp"> {
  let summary =
      "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createConvertToLLVMPass()";
  let options = [
    Option<"reassociateFpReductions", "reassociateFpReductions", "bool",
            /*default=*/"false",
           "Specifies if FP add and mult reductions can be reordered">,
  ];
}

def ExpandArithF16ToF32 :
    Pass<"iree-llvmcpu-expand-f16-op-to-f32", ""> {
  let summary =
      "Preform f16 opertaions by expanding them to f32.";
  let constructor =
      "mlir::iree_compiler::createExpandF16OpToF32Pass()";
}

def LLVMCPUAssignConstantOrdinals :
    Pass<"iree-llvmcpu-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all LLVMCPU variants.";
  let constructor = "mlir::iree_compiler::createLLVMCPUAssignConstantOrdinalsPass()";
}

def LLVMCPUAssignImportOrdinals :
    Pass<"iree-llvmcpu-assign-import-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable import ordinals across all LLVMCPU variants.";
  let constructor = "mlir::iree_compiler::createLLVMCPUAssignImportOrdinalsPass()";
}

def LLVMCPUCheckIRBeforeLLVMConversion :
    Pass<"iree-llvmcpu-check-ir-before-llvm-conversion", "ModuleOp"> {
  let summary = "Checks CPU backend specific IR constraints (like no allocas)";
  let constructor = "mlir::iree_compiler::createLLVMCPUCheckIRBeforeLLVMConversionPass()";
}

def LLVMCPUEmitVectorizationRemarks :
    Pass<"iree-llvmcpu-emit-vectorization-remarks", "func::FuncOp"> {
  let summary = "Emit vectorization remarks on Linalg ops";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUEmitVectorizationRemarksPass()";
}

def LLVMCPULinkExecutables :
    Pass<"iree-llvmcpu-link-executables", "mlir::ModuleOp"> {
  let summary = "Links LLVMCPU HAL executables within the top-level program module.";
  let constructor = "mlir::iree_compiler::createLLVMCPULinkExecutablesPass()";
}

def LLVMCPULowerExecutableTarget :
    Pass<"iree-llvmcpu-lower-executable-target",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createLLVMCPULowerExecutableTargetPass()";
}

def LLVMCPULowerToUKernels :
    Pass<"iree-llvmcpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::createLLVMCPULowerToUKernelsPass()";
  let options = [
    Option<"optionSkipIntermediateRoundings", "skip-intermediate-roundings",
      "bool", /*default=*/"true",
      "Allow skipping intermediate roundings, e.g. in f16 ukernels internally doing f32 arithmetic.">,
  ];
}

def LLVMCPUMmt4dVectorLowering
    : Pass<"iree-llvmcpu-mmt4d-vector-lowering", "func::FuncOp"> {
  let summary = "Apply vector lowering logic to vector ops";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUMmt4dVectorLoweringPass()";
}

def LLVMCPUPeel :
    Pass<"iree-llvmcpu-peel", "func::FuncOp"> {
  let summary = "Pass to perform peeling on non-distributed loops.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUPeelPass()";
}

def LLVMCPUSplitReduction : Pass<"iree-llvmcpu-split-reduction", "func::FuncOp"> {
  let summary = "Pass to splitReduce linalg operations.";
  let constructor = "mlir::iree_compiler::createLLVMCPUSplitReductionPass()";
  let options = [
    Option<"enableFpReductionReordering", "enable-fp-reduction-reordering",
      "bool", /*default=*/"false",
      "Flag to enable reduction reordering on floating points.">,
  ];
}

def LLVMCPUSynchronizeSymbolVisibility :
    Pass<"iree-llvmcpu-synchronize-symbol-visibility", "ModuleOp"> {
  let summary = "Synchronizes LLVM linkage with MLIR symbol visibility";
  let constructor = "mlir::iree_compiler::createLLVMCPUSynchronizeSymbolVisibilityPass()";
}

def LLVMCPUTensorPad :
    Pass<"iree-llvmcpu-tensor-pad", "func::FuncOp"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTensorPadPass()";
}

def LLVMCPUTile :
    Pass<"iree-llvmcpu-tile", "func::FuncOp"> {
  let summary = "Pass to tile TilingInterface operations.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTilePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUTileAndFuse :
    Pass<"iree-llvmcpu-tile-and-fuse", "func::FuncOp"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTileAndFusePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUUnfuseFMAOps :
    Pass<"iree-llvmcpu-unfuse-fma-pass", "func::FuncOp"> {
  let summary = "Convert llvm.fma into unfused mulf and addf ops";
  let constructor = "mlir::iree_compiler::createLLVMCPUUnfuseFMAOpsPass()";
}

def LLVMCPUVectorLowering :
    Pass<"iree-llvmcpu-vector-lowering", "func::FuncOp"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
  let options = [
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
    Option<"lowerVectorTransposeToAVX2", "lower-vector-transpose-to-avx2", "bool",
      /*default=*/"false",
      "Add specific transpose to avx2 lowering patterns.">,
  ];
  let constructor =
      "mlir::iree_compiler::createLLVMCPUVectorLoweringPass()";
}

def VectorContractCustomKernels :
    Pass<"iree-llvmcpu-vector-contract-custom-kernels", "func::FuncOp"> {
  let summary = "Enable custom kernels (inline assembly or intrinsics) for some vector.contract ops";
  let constructor = "mlir::iree_compiler::createVectorContractCustomKernelsPass()";
}

def VerifyLinalgTransformLegality :
    Pass<"iree-llvmcpu-verify-linalg-transform-legality", "ModuleOp"> {
  let summary = "Verify that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::createVerifyLinalgTransformLegalityPass()";
}

#endif // IREE_CODEGEN_LLVMCPU_PASSES
