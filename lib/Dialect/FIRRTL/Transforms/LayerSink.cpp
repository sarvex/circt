//===- LayerSink.cpp - Sink ops into layer blocks -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//>
// This file sinks operations into layer blocks.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLConnectionGraph.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "llvm/ADT/SCCIterator.h"

#define DEBUG_TYPE "firrtl-layer-sink"

using namespace circt;
using namespace firrtl;

namespace {
/// A control-flow sink pass.
struct LayerSink : public LayerSinkBase<LayerSink> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LayerSink::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LayerSink "
                      "---------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);
  // auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  // getOperation()->walk([&](LayerBlockOp layerBlock) {
  //   SmallVector<Region *> regionsToSink({&layerBlock.getRegion()});
  //   numSunk = controlFlowSink(
  //       regionsToSink, domInfo,
  //       [](Operation *op, Region *) { return !hasDontTouch(op); },
  //       [](Operation *op, Region *region) {
  //         // Move the operation to the beginning of the region's entry block.
  //         // This guarantees the preservation of SSA dominance of all of the
  //         // operation's uses are in the region.
  //         op->moveBefore(&region->front(), region->front().begin());
  //       });
  // });

  int sccNumber = 0;

  FIRRTLOperation *moduleOp = getOperation();
  for (llvm::scc_iterator<detail::FIRRTLOperation *>
           i = llvm::scc_begin(moduleOp),
           e = llvm::scc_end(moduleOp);
       i != e; ++i) {
    LLVM_DEBUG({
      llvm::dbgs() << "SCC: " << sccNumber++ << "\n";
      for (auto *op : *i) {
        llvm::errs() << "  - " << *op << "\n";
      }
    });
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLayerSinkPass() {
  return std::make_unique<LayerSink>();
}
