# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Access to the Torch JIT operator registry."""

from typing import Dict, List, Tuple, Union

import io
import itertools

from .utils import TextEmitter

# Note that this utility exists only in the c-extension.
from torch_mlir._mlir_libs._jit_ir_importer import get_registered_ops # pytype: disable=import-error

def _pytype_to_shape_fn_pytype(pytype: str) -> str:
    """Convert a JitOperator pytype to the type relevant in shape functions.

    In particular, this converts `Tensor` to `List[int]`, along with a few
    other special cases.
    """
    # `Scalar` operands (which are represented with pytype "number") can
    # be either `int` or `float`. TorchScript has no way to represent a
    # signature of this type, so we hardcode it to `float`. `Scalar`
    # operands don't participate in shape functions (since they are
    # logically real-valued), so it doesn't really matter much, and
    # `float` helps make it clearer that it's not part of the shape
    # function.
    if pytype == "number":
        return "float"
    if pytype == "Optional[number]":
        return "Optional[float]"
    # `torch.device` is lowercase.
    if pytype == "Device":
        return "device"
    if pytype == "Optional[Device]":
        return "Optional[device]"
    # Shape functions only care about the shape of tensors.
    if pytype == "Tensor":
        return "List[int]"
    if pytype == "Optional[Tensor]":
        return "Optional[List[int]]"
    if pytype == "List[Tensor]":
        return "List[List[int]]"
    if pytype == "List[Optional[Tensor]]":
        return "List[Optional[List[int]]]"
    # Generators don't contribute to shapes, and are not scriptable currently.
    # So just hack them to be passed as "Any".
    if pytype == "Generator":
        return "Any"
    if pytype == "Optional[Generator]":
        return "Any"
    return pytype

class JitOperator:
    """Information about a single registered `torch::jit::Operator`"""
    def __init__(self, op_info: "OP_INFO_DICT"):
        """Create a JitOperator from the raw OP_INFO_DICT extracted from
        the PyTorch JIT operator registry.
        """
        namespace, _, unqualified_name = op_info["name"][0].partition("::")
        self.namespace = namespace
        self.unqualified_name = unqualified_name
        self.overload_name = op_info["name"][1]
        self.is_c10_op = op_info["is_c10_op"]
        self.is_vararg = op_info["is_vararg"]
        self.is_varret = op_info["is_varret"]
        self.is_mutable = op_info["is_mutable"]
        self.arguments = op_info["arguments"]
        self.returns = op_info["returns"]

        self.unique_key = self.create_unique_key()

    def create_unique_key(self) -> str:
        """Create a unique, human-readable key for this JitOperator.

        The key consists of the operator name and its overload name, which
        together form a unique identifier. We also redundantly
        append a signature to the end, which gives some robustness to changes
        in PyTorch and also generally makes things more readable.
        The format is:
        ```
            namespace::unqualified_name[.overload_name] : (type1,type2) -> (type3,type4)
        ```
        This is a modified version of the signature strings seen throughout
        PyTorch. The main difference is the inclusion of return types and the
        extra spacing and `:`. E.g.the above would be just:
        ```
          namespace::kernel_name[.overload_name](type1,type2)
        ```
        (PyTorch doesn't canonically include the result types since they don't
        participate in their dispatch overload resolution, which is of primary
        concern for them)
        """
        overload = "" if not self.overload_name else f".{self.overload_name}"
        if self.is_vararg:
            arg_str = "..."
        else:
            arg_str = ", ".join(arg["type"] for arg in self.arguments)
        if self.is_varret:
            ret_str = "..."
        else:
            ret_str = ", ".join(ret["type"] for ret in self.returns)
        return f"{self.namespace}::{self.unqualified_name}{overload} : ({arg_str}) -> ({ret_str})"

    @property
    def triple(self):
        """Returns the unique 3-tuple identifying this operator.

        This is a useful alternative to the "unique name" for programmatic
        access, such as when needing to convert one op to a related op by
        a programmatic transformation of the triple.
        """
        return self.namespace, self.unqualified_name, self.overload_name

    def get_mlir_names(self):
        """Gets the MLIR op name (excluding `torch.`) and td def name.

        Not all ops are necessarily registered or in the .td file, but these
        are useful in the repr for cross referencing, and it's useful to have
        them in a single point of truth.
        """
        def uppercase_first_letter(s):
            if not s:
                return s
            return s[0].upper() + s[1:]

        op_name_atoms = [self.namespace, self.unqualified_name]
        if self.overload_name:
            op_name_atoms.append(self.overload_name)
        op_name = ".".join(op_name_atoms)

        op_class_name_atoms = []
        for op_name_atom in op_name_atoms:
            for s in op_name_atom.split("_"):
                op_class_name_atoms.append(s if s else "_")
        cpp_class_name = "".join(
            uppercase_first_letter(s) for s in op_class_name_atoms) + "Op"
        # Disallow leading underscores in C++ to avoid illegal names.
        cpp_class_name = cpp_class_name.lstrip("_")
        return op_name, cpp_class_name

    def get_shape_function_signature(self):
        """Gets the Python function signature for this op's shape function.

        While this is technically debug-only output, it is useful to copy-paste
        it from the debug dump into the shape library definitions, as many
        ops have extra default arguments and stuff that are tedious to write out
        right.
        """
        mlir_op_name, _ = self.get_mlir_names()
        # Replace `.` with a valid Python identifier character.
        # `〇` vaguely looks like `.`.
        def_name = "〇".join(mlir_op_name.split("."))
        parameter_decls = []
        for arg in self.arguments:
            pytype = _pytype_to_shape_fn_pytype(arg["pytype"])
            default = ""
            if "default_debug" in arg:
                if "List" in arg["pytype"]:
                    # TorchScript doesn't allow lists as default parameters due
                    # to the weird Python semantics of mutable default
                    # arguments. So munge them into tuples, which work
                    # fine here. We only need these to simplify the invocation
                    # of the shape functions as valid Python for testing against
                    # the real ops, and tuples work fine in all the places this
                    # kicks in (e.g. conv dilations -- we aren't mutating those
                    # lists).
                    default_debug = arg["default_debug"].replace(
                        '[', '(').replace(']', ')')
                elif arg["pytype"] == "str":
                    default_debug = repr(arg["default_debug"]).replace("'", '"')
                else:
                    default_debug = arg["default_debug"]
                default = f" = {default_debug}"
            parameter_name = arg["name"]
            if parameter_name == "from":
                parameter_name = "from_" # Avoid using a Python keyword.
            parameter_decls.append(f"{parameter_name}: {pytype}{default}")
        ret_decls = []
        for ret in self.returns:
            pytype = _pytype_to_shape_fn_pytype(ret["pytype"])
            ret_decls.append(f"{pytype}")
        parameters = ", ".join(parameter_decls)
        result = ", ".join(ret_decls)
        if len(ret_decls) >= 2:
            result = f"Tuple[{result}]"

        return f"def {def_name}({parameters}) -> {result}:"

    def __repr__(self):
        f = io.StringIO()
        emitter = TextEmitter(f)
        p = lambda *args: emitter.print(*args)
        p(f"JitOperator '{self.unique_key}':")
        with emitter.indent():

            # Emit the MLIR names to allow easy reverse lookup if starting
            # from an unregistered op.
            op_name, cpp_class_name = self.get_mlir_names()
            p(f"MLIR op name = torch.{op_name}")
            p(f"MLIR cpp class name = {cpp_class_name}")

            p(f"namespace = {self.namespace}")
            p(f"unqualified_name = {self.unqualified_name}")
            p(f"overload_name = {self.overload_name}")
            p(f"is_c10_op = {self.is_c10_op}")
            p(f"is_vararg = {self.is_vararg}")
            p(f"is_varret = {self.is_varret}")
            p(f"is_mutable = {self.is_mutable}")
            if any(ret["type"] == "Tensor" for ret in self.returns):
                p(f"shape_function_signature = {self.get_shape_function_signature()}")
            if not self.arguments:
                p("arguments = []")
            else:
                p("arguments:")
                with emitter.indent():
                    for arg in self.arguments:
                        p(f"arg: {arg}")
            if not self.returns:
                p("returns = []")
            else:
                p("returns:")
                with emitter.indent():
                    for ret in self.returns:
                        p(f"ret: {ret}")
        return f.getvalue()

    def has_value_semantics(self):
        """Indicates whether the operator HasValueSemantics."""

        # Delegate to is_readonly for handling some incorrectly annotated ops.
        # HasValueSemantics implies ReadOnly.
        if not self.is_readonly():
            return False

        # Vararg and varret ops don't have sufficient annotations for us to
        # be sure whether they satisfy the requirements of HasValueSemantics.
        if self.is_vararg or self.is_varret:
            return False
        # If no operands have aliasing relations, then the op has value semantics.
        # Note that this is different from MLIR's NoSideEffect which is much
        # stronger (for example, it cannot be applied to ops that might emit errors
        # when operand shapes mismatch).
        if any("alias_info" in x for x in itertools.chain(self.arguments, self.returns)):
            return False
        # It seems the FunctionSchema of "prim::unchecked_cast : (t) -> (t)" has
        # incorrect alias information. The result can alias with other tensors
        # but the alias annotation is empty.
        if self.unique_key == "prim::unchecked_cast : (t) -> (t)":
            return False
        # The `is` operator compares object identity, so it does not have
        # value semantics.
        if self.unique_key in ("aten::__is__ : (t1, t2) -> (bool)",
                               "aten::__isnot__ : (t1, t2) -> (bool)"):
            return False
        return True

    def is_readonly(self):
        """Indicates whether the operator is ReadOnly."""

        triple = (self.namespace, self.unqualified_name, self.overload_name)
        # TODO: Handle some exceptions of incorrectly annotated ops.
        # We have incorrectly grown a reliance on the incorrect annotation of
        # `aten::batch_norm`.
        # See https://github.com/pytorch/pytorch/issues/73050#issuecomment-1051382044
        # if triple in [("aten", "batch_norm", ""), ("aten", "instance_norm", "")]:
        #     return False

        # If any argument or return has an alias_info that indicates mutation,
        # the op is not ReadOnly.
        for x in itertools.chain(self.arguments, self.returns):
            if "alias_info" in x:
                if x["alias_info"]["is_write"]:
                    return False
        return True


class Registry:
    """An indexed collection of JitOperators"""
    def __init__(self, operators: List[JitOperator]):
        self.by_unique_key = {}
        self.by_triple = {}
        for o in operators:
            self.by_unique_key[o.unique_key] = o
            self.by_triple[o.triple] = o

    @staticmethod
    def load() -> "Registry":
        return Registry([JitOperator(op_info) for op_info in get_registered_ops()])

    def __getitem__(self, key: str):
        """Looks up a JitOperator by its "unique key"."""
        return self.by_unique_key[key]

    def get_by_triple(self, key: Tuple[str, str, str]):
        """Looks up a JitOperator by its unique "triple"."""
        return self.by_triple[key]


# A List[Dict[str, _]] mapping attribute names to:
#   - str (e.g. {'name': 'dim'} )
#   - int (e.g. {'N': 1} )
#   - Dict[str, List[str]]
#       (e.g. {'alias_info': {'before': ['alias::a'], 'after': ['alias::a']}} )
SIGLIST_TYPE = List[Dict[str, Union[str, int, Dict[str, List[str]]]]]
# A Dict[str, _] describing a registered op. Each field is either
#   - bool (e.g. {'is_mutable': False} )
#   - Tuple[str] (e.g. {'name': ('aten::size', 'int')} )
#   - SIGLIST_TYPE (e.g. {'arguments': [...], 'returns': [...]} )
OP_INFO_DICT = Dict[str, Union[bool, Tuple[str], SIGLIST_TYPE]]
