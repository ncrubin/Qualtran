# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: qualtran/protos/registers.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from qualtran.protos import args_pb2 as qualtran_dot_protos_dot_args__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fqualtran/protos/registers.proto\x12\x08qualtran\x1a\x1aqualtran/protos/args.proto\"\xbf\x01\n\x08Register\x12\x0c\n\x04name\x18\x01 \x01(\t\x12%\n\x07\x62itsize\x18\x02 \x01(\x0b\x32\x14.qualtran.IntOrSympy\x12#\n\x05shape\x18\x03 \x03(\x0b\x32\x14.qualtran.IntOrSympy\x12%\n\x04side\x18\x04 \x01(\x0e\x32\x17.qualtran.Register.Side\"2\n\x04Side\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x08\n\x04LEFT\x10\x01\x12\t\n\x05RIGHT\x10\x02\x12\x08\n\x04THRU\x10\x03\"2\n\tRegisters\x12%\n\tregisters\x18\x01 \x03(\x0b\x32\x12.qualtran.Registerb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'qualtran.protos.registers_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_REGISTER']._serialized_start=74
  _globals['_REGISTER']._serialized_end=265
  _globals['_REGISTER_SIDE']._serialized_start=215
  _globals['_REGISTER_SIDE']._serialized_end=265
  _globals['_REGISTERS']._serialized_start=267
  _globals['_REGISTERS']._serialized_end=317
# @@protoc_insertion_point(module_scope)
