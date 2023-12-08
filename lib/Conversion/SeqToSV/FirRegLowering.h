//===- FirRegLowering.h - FirReg lowering utilities ===========--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_SEQTOSV_FIRREGLOWERING_H
#define CONVERSION_SEQTOSV_FIRREGLOWERING_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include <bits/chrono.h>
#include <chrono>

namespace circt {
struct ValueSCC {
  struct SccInfo {
    size_t order;
    size_t lowlink;
    size_t componentId;
  };
  llvm::DenseMap<Value, SccInfo> valueInfoMap;
  unsigned componentIdGen = 0;
  unsigned index = 0;
  llvm::SmallVector<Value> stack;
  llvm::DenseSet<Value> onStack;
  llvm::function_ref<bool(Operation *)> filter;
  ValueSCC(hw::HWModuleOp moduleOp,
           llvm::function_ref<bool(Operation *)> f = nullptr) {
    if (f)
      filter = f;
    auto time_begin = std ::chrono::high_resolution_clock::now();
    llvm::errs() << "\n begin scc:";
    for (Operation *rootOp : moduleOp.getOps<seq::FirRegOp>())
      tarjanSCCiterative(rootOp);
    auto time_end = std ::chrono::high_resolution_clock::now();
    llvm::errs() << "\n time to complete scc:"
                 << std::chrono::duration_cast<std::chrono::seconds>(time_end -
                                                                     time_begin)
                        .count();
  };

  void tarjanSCCiterative(Operation *rootOp) {
    // Stack to simulate the recursive call stack
    SmallVector<std::pair<Value, bool>, 64> dfsStack;
    for (auto val : rootOp->getResults())
      dfsStack.push_back({val, true});

    llvm::DenseMap<Value, Value> parentOf;
    while (!dfsStack.empty()) {
      Value visitVal = dfsStack.back().first;
      auto &firstVisit = dfsStack.back().second;

      if (firstVisit) {
        valueInfoMap[visitVal] = {index, index, 0};
        ++index;
        stack.push_back(visitVal);
        onStack.insert(visitVal);
      }

      bool continueTraversal = false;
      size_t minLowLink = valueInfoMap[visitVal].lowlink;
      for (auto *user : visitVal.getUsers()) {
        // If ops need to be filtered ignore them, cannot ignore rootOp,
        // otherwise SCC cannot be computed.
        if (user != rootOp && filter && filter(user))
          continue;
        for (auto childVal : user->getResults()) {
          // set the firstvisit flag of visitVal to false.
          firstVisit = false;
          if (!valueInfoMap.contains(childVal)) {
            // If child not yet visited.
            dfsStack.push_back(std::make_pair(childVal, true));
            continueTraversal = true;
            parentOf[childVal] = visitVal;
            // Simulate dfs traversal, defer visitVal traversal and start
            // childVal traversal.
            break;
          }
          if (parentOf[childVal] == visitVal) {
            // Set lowLink of visitVal, if its the immediate parent of
            // childVal.
            minLowLink = std::min(minLowLink, valueInfoMap[childVal].lowlink);
          } else if (onStack.contains(childVal)) {
            minLowLink = std::min(minLowLink, valueInfoMap[childVal].order);
          }
        }
        if (continueTraversal)
          break;
      }
      if (continueTraversal)
        continue;
      valueInfoMap[visitVal].lowlink = minLowLink;

      if (valueInfoMap[visitVal].lowlink == valueInfoMap[visitVal].order) {
        auto recordSCC = [&]() {
          auto neighbor = stack.pop_back_val();
          onStack.erase(neighbor);
          valueInfoMap[neighbor].componentId = componentIdGen;
        };
        while (stack.back() != visitVal)
          recordSCC();
        recordSCC();

        ++componentIdGen;
      }
      dfsStack.pop_back();
    }
  }

  bool isInSameSCC(Value lhs, Value rhs) const {
    auto lhsIt = valueInfoMap.find(lhs);
    auto rhsIt = valueInfoMap.find(rhs);
    return lhsIt != valueInfoMap.end() && rhsIt != valueInfoMap.end() &&
           lhsIt->getSecond().componentId == rhsIt->getSecond().componentId;
  }

  void erase(Value v) { valueInfoMap.erase(v); }
  ~ValueSCC() = default;
};
/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLowering {
public:
  FirRegLowering(TypeConverter &typeConverter, hw::HWModuleOp module,
                 bool disableRegRandomization = false,
                 bool emitSeparateAlwaysBlocks = false);

  void lower();

  void initBackwardSlice();
  bool needsRegRandomization() const { return needsRandom; }

  unsigned numSubaccessRestored = 0;

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    IntegerAttr preset;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(seq::FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);
  void initializeRegisterElements(Location loc, OpBuilder &builder, Value reg,
                                  Value rand, unsigned &pos);

  void createTree(OpBuilder &builder, Value reg, Value term, Value next);
  std::optional<std::tuple<Value, Value, Value>>
  tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                        hw::ArrayCreateOp nextRegValue);

  void addToAlwaysBlock(Block *block, sv::EventControl clockEdge, Value clock,
                        const std::function<void(OpBuilder &)> &body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        const std::function<void(OpBuilder &)> &resetBody = {});

  void addToIfBlock(OpBuilder &builder, Value cond,
                    const std::function<void()> &trueSide,
                    const std::function<void()> &falseSide);

  hw::ConstantOp getOrCreateConstant(Location loc, const APInt &value) {
    OpBuilder builder(module.getBody());
    auto &constant = constantCache[value];
    if (constant) {
      constant->setLoc(builder.getFusedLoc({constant->getLoc(), loc}));
      return constant;
    }

    constant = builder.create<hw::ConstantOp>(loc, value);
    return constant;
  }

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  using IfKeyType = std::pair<Block *, Value>;
  llvm::SmallDenseMap<IfKeyType, sv::IfOp> ifCache;

  llvm::SmallDenseMap<APInt, hw::ConstantOp> constantCache;
  llvm::SmallDenseMap<std::pair<Value, unsigned>, Value> arrayIndexCache;
  std::unique_ptr<ValueSCC> scc;

  TypeConverter &typeConverter;
  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;

  bool needsRandom = false;
};
} // namespace circt

#endif // CONVERSION_SEQTOSV_FIRREGLOWERING_H
