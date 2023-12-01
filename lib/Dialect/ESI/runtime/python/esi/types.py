#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import esiCppAccel as cpp

from typing import Callable, Dict, List, Optional, Type


def __get_type(cpp_type: cpp.Type):
  # if cpp_type in __esi_mapping:
  #   return __esi_mapping[type(cpp_type)](cpp_type)
  for cpp_type_cls, fn in __esi_mapping.items():
    if isinstance(cpp_type, cpp_type_cls):
      return fn(cpp_type)
  return ESIType(cpp_type)


__esi_mapping: Dict[Type, Callable] = {}


class ESIType:

  def __init__(self, cpp_type: cpp.Type):
    self.cpp_type = cpp_type

  def is_valid(self, obj) -> bool:
    """Is a Python object compatible with HW type?"""
    assert False, "unimplemented"


class VoidType(ESIType):

  def is_valid(self, obj) -> bool:
    return obj is None


__esi_mapping[cpp.VoidType] = VoidType


class BitsType(ESIType):

  def __init__(self, width: int):
    self.width = width

  def is_valid(self, obj) -> bool:
    return isinstance(obj, bytearray) and len(obj) == (self.width + 7) / 8


__esi_mapping[cpp.BitVectorType] = BitsType


class IntType(ESIType):

  def __init__(self, width: int, cpp_type: cpp.Type):
    super().__init__(cpp_type)
    self.width = width

  def is_valid(self, obj) -> bool:
    if self.width == 0:
      return obj is None
    if not isinstance(obj, int):
      return False
    if obj >= 2**self.width:
      return False
    return True


__esi_mapping[cpp.IntegerType] = IntType


class UIntType(IntType):

  def __str__(self) -> str:
    return f"uint{self.width}"


__esi_mapping[cpp.UIntType] = IntType


class SIntType(IntType):

  def __str__(self) -> str:
    return f"sint{self.width}"


__esi_mapping[cpp.SIntType] = IntType


class StructType(ESIType):

  def __init__(self, cpp_type: cpp.StructType):
    self.cpp_type = cpp_type

  def is_valid(self, obj) -> bool:
    fields_count = 0
    if not isinstance(obj, dict):
      obj = obj.__dict__

    for (fname, ftype) in self.cpp_type.fields:
      if fname not in obj:
        return False
      if not ftype.is_valid(obj[fname]):
        return False
      fields_count += 1
    if fields_count != len(obj):
      return False
    return True


class Port:

  def __init__(self, cpp_port: cpp.ChannelPort):
    self.cpp_port = cpp_port
    self.type = cpp_port.type

  def connect(self):
    self.cpp_port.connect()


class WritePort(Port):

  def __init__(self, cpp_port: cpp.WriteChannelPort):
    self.cpp_port = cpp_port
    self.type = cpp_port.type

  def write(self, msg=None) -> bool:
    if not self.type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.type}'")
    msg_bytes: bytearray = self.type.serialize(msg)
    self.cpp_port.write(msg_bytes)
    return True


class ReadPort:

  def __init__(self, cpp_port: cpp.ReadChannelPort):
    self.cpp_port = cpp_port
    self.type = cpp_port.type

  def read(self, blocking_timeout: Optional[float] = 1.0):
    assert False


class BundlePort:

  def __init__(self, cpp_port: cpp.BundlePort):
    self.cpp_port = cpp_port

  def write_port(self, channel_name: str) -> WritePort:
    return WritePort(self.cpp_port.getWrite(channel_name))

  def read_port(self, channel_name: str) -> ReadPort:
    return ReadPort(self.cpp_port.getRead(channel_name))
