; RUN: firtool %s | FileCheck %s

circuit Foo: %[[{"class": "circt.MarkRegisterAlwaysRandomized",
                 "target": "~Foo|Foo>r"}]]
  ; CHECK: module Foo
  module Foo:
    input a: UInt<1>
    input clock: Clock
    output b1: UInt<1>
    reg r: UInt<1>, clock
    ; CHECK: `ifdef ENABLE_INITIAL_REG_
    ; CHECK-NOT: RANDOMIZE_REG_INIT
    ; CHECK: r = _RANDOM[/*Zero width*/ 1'b0][0];
    r <= a
    b1 <= r