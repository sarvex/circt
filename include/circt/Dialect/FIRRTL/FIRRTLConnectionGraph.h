//===- FIRRTLConnectionGraph.h - Graph of connections in FIRRTL -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for working with FIRRTL operations using LLVM
// graph utilties.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <variant>

#define DEBUG_TYPE "firrtl-connection-graph"

namespace circt {
namespace firrtl {
namespace detail {

// "Using" is used here to avoid polluting the global namespace with
// CIRCT-specific graph traits.  This pattern is borrowed from HWModuleGraph.h.
using FIRRTLOperation = mlir::Operation;

} // namespace detail

using detail::FIRRTLOperation;

/// An iterator over connections to a module's ports.
class FModuleOpIterator
    : public llvm::iterator_facade_base<
          FModuleOpIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FModuleOp op;
  Value::use_iterator portIterator, portIteratorEnd;
  unsigned portIndex = 0, portIndexEnd;

  /// Skip ports which have no users.  Skip users which are the destinations of
  /// connects.
  void fastforward();

public:
  FModuleOpIterator() : op(nullptr), portIndexEnd(0) {
    // llvm::errs() << "Constructing empty FModuleOpIterator\n";
  }

  FModuleOpIterator(FModuleOp op);

  bool operator==(const FModuleOpIterator &other) const;

  FIRRTLOperation &operator*() const { return *portIterator.getUser(); }

  FModuleOpIterator &operator++();
};

/// An iterator over the connections of an FConnectLikeOp.
class FConnectLikeIterator
    : public llvm::iterator_facade_base<
          FConnectLikeIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FConnectLike op;

  bool visited = false;

public:
  FConnectLikeIterator() : op(nullptr), visited(true) {
    LLVM_DEBUG(
        { llvm::errs() << "Constructing empty FConnectLikeIterator\n"; });
  }

  FConnectLikeIterator(FConnectLike op) : op(op) {
    LLVM_DEBUG({
      llvm::errs() << "Constructing FConnectLikeIterator:\n"
                   << "  op: " << op << "\n"
                   << "  visited: " << visited << "\n";
    });
  };

  bool operator==(const FConnectLikeIterator &other) const;

  FIRRTLOperation &operator*();

  FConnectLikeIterator &operator++();
};

/// Generic iterator for anything that is not an FConnectLike or an FModuleOp.
class ResultIterator
    : public llvm::iterator_facade_base<
          ResultIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FIRRTLOperation *op;

  Value::use_iterator resultIterator, resultIteratorEnd;
  int resultIndex = 0, resultIndexEnd;

  /// Iterate through results which have no users.
  void fastforward();

public:
  ResultIterator() : op(nullptr), resultIndexEnd(0) {
    LLVM_DEBUG({ llvm::errs() << "Constructing empty ResultIterator\n"; });
  }

  ResultIterator(FIRRTLOperation *op);

  bool operator==(const ResultIterator &other) const;

  FIRRTLOperation &operator*() const;

  ResultIterator &operator++();
};

class ConnectionIterator {

  using VariantIterator = std::variant<std::monostate, FModuleOpIterator,
                                       FConnectLikeIterator, ResultIterator>;

  FIRRTLOperation *op = nullptr;

  VariantIterator iterator;

public:
  ConnectionIterator(FIRRTLOperation *op, bool empty = false);

  bool operator==(const ConnectionIterator &other) const;

  bool operator!=(const ConnectionIterator &other) const;

  FIRRTLOperation *operator*();

  ConnectionIterator &operator++();

  ConnectionIterator operator++(int);

  static ConnectionIterator child_begin(FIRRTLOperation *op) {
    return ConnectionIterator(op);
  }

  static ConnectionIterator child_end(FIRRTLOperation *op) {
    return ConnectionIterator(op, /*empty=*/true);
  }
};
} // namespace firrtl
} // namespace circt

namespace llvm {

template <>
struct llvm::GraphTraits<circt::firrtl::detail::FIRRTLOperation *> {
  // using ChildIteratorType = mlir::Operation::user_iterator;
  using ChildIteratorType = circt::firrtl::ConnectionIterator;
  using Node = circt::firrtl::detail::FIRRTLOperation;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef op) { return op; }
  // static ChildIteratorType child_begin(NodeRef op) { return op->user_begin();
  // } static ChildIteratorType child_end(NodeRef op) { return op->user_end(); }
  static ChildIteratorType child_begin(NodeRef op) {
    return circt::firrtl::ConnectionIterator::child_begin(op);
  }
  static ChildIteratorType child_end(NodeRef op) {
    return circt::firrtl::ConnectionIterator::child_end(op);
  }
};

} // namespace llvm

#undef DEBUG_TYPE

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H
