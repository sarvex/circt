//===- ModuleSlicer.cpp - Intermodule ConstProp and DCE ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements SCCP:
// https://www.cs.wustl.edu/~cytron/531Pages/f11/Resources/Papers/cprop.pdf
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/APInt.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace circt;
using namespace firrtl;

#define DEBUG_TYPE "firrtl-module-slicer"

namespace {
struct ModuleSlicerPass : public ModuleSlicerBase<ModuleSlicerPass> {
  ModuleSlicerPass(mlir::StringRef name) {
    this->moduleName = std::string(name);
  }
  void runOnOperation() override;
};
} // end anonymous namespace

void ModuleSlicerPass::runOnOperation() {

  auto circuit = getOperation();
  if (moduleName.empty()) {
    mlir::emitError(circuit.getLoc()) << "Please specify a module name";
    return signalPassFailure();
  }

  bool exist = false;
  llvm::SmallDenseSet<StringAttr> liveModules;
  // Delete all annotations.
  StringAttr realName;
  for (auto module : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<firrtl::FModuleOp>())) {
    AnnotationSet::removePortAnnotations(module,
                                         [](auto, auto) { return true; });
    AnnotationSet::removeAnnotations(module, [](auto) { return true; });

    if (!module.getName().ends_with(moduleName)) {
      OpBuilder builder(module);
      auto ext = builder.create<firrtl::FExtModuleOp>(
          module->getLoc(),
          module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
          module.getConventionAttr(), module.getPorts(), StringRef(),
          module.getAnnotationsAttr());
      ext.setPrivate();
      module.erase();
    } else {
      module.setPublic();
      realName = module.getNameAttr();
      module.walk([&](InstanceOp instance) {
        liveModules.insert(instance.getModuleNameAttr().getAttr());
      });
      exist = true;
    }
  }

  if (!exist) {
    mlir::emitError(circuit.getLoc()) << "Module name is not valid";
    return signalPassFailure();
  }

  // Delete all hierpaths.
  for (auto module : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<firrtl::FExtModuleOp>()))
    if (!liveModules.count(module.getNameAttr()))
      module.erase();

  // Delete all hierpaths.
  for (auto module : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<hw::HierPathOp>()))
    module.erase();

  circuit.walk([](Operation *op) {
    AnnotationSet::removeAnnotations(op, [](auto) { return true; });
  });

  circuit.setName(realName);
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createModuleSlicerPass(llvm::StringRef moduleName) {
  return std::make_unique<ModuleSlicerPass>(moduleName);
}
