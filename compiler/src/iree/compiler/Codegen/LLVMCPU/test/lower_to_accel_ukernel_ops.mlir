// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-lower-to-accel-ukernels{skip-intermediate-roundings=true},cse,canonicalize))" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-lower-to-accel-ukernels{skip-intermediate-roundings=false},cse,canonicalize))" %s | FileCheck %s --check-prefix=NOSKIPROUND

func.func @matmul_f32f32f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 1281 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[M0_index:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[M0:.+]] = arith.index_cast %[[M0_index]] : index to i32
//  CHECK-DAG:   %[[N0_index:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[N0:.+]] = arith.index_cast %[[N0_index]] : index to i32
//  CHECK-DAG:   %[[K0_index:.+]] = tensor.dim %[[ARG1]], %[[C3]]
//  CHECK-DAG:   %[[K0:.+]] = arith.index_cast %[[K0_index]] : index to i32
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "aie_matmul_f32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[M0]], %[[N0]], %[[K0]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----
