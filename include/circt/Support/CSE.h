#ifndef CIRCT_TRANSFORMS_CSEUTILS_H
#define CIRCT_TRANSFORMS_CSEUTILS_H
#include "mlir/Transforms/CSEUtils.h"
namespace circt {
struct NamehintInsensitiveOperationEquivalenceInterface
    : public mlir::DialectOperationEquivalenceInterface {
  NamehintInsensitiveOperationEquivalenceInterface(Dialect *dialect)
      : DialectOperationEquivalenceInterface(dialect) {}

  bool containsNonEssentialAttribute(DictionaryAttr attr) const final {
    return attr.contains("sv.namehint");
  }
  bool isNonEssentialAttribute(NamedAttribute namedAttr) const final {
    return namedAttr.getName() == "sv.namehint";
  }
};
} // namespace circt
#endif