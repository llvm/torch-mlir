# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class RangeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1, 5)

@register_test_case(module_factory=lambda: RangeIntModule())
def RangeIntModule_basic(module, tu: TestUtils):
    module.forward()


class RangeStepIntModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1, 5, 1)

@register_test_case(module_factory=lambda: RangeStepIntModule1())
def RangeStepIntModule1_basic(module, tu: TestUtils):
    module.forward()  


class RangeStepIntModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1, 5, 2)

@register_test_case(module_factory=lambda: RangeStepIntModule2())
def RangeStepIntModule2_basic(module, tu: TestUtils):
    module.forward()   


class RangeStartNegativeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(-10, 5, 4)

@register_test_case(module_factory=lambda: RangeStartNegativeIntModule())
def RangeStartNegativeIntModule_basic(module, tu: TestUtils):
    module.forward()   


class RangeEndStepNegativeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(10, -1, -2)

@register_test_case(module_factory=lambda: RangeEndStepNegativeIntModule())
def RangeEndStepNegativeIntModule_basic(module, tu: TestUtils):
    module.forward()  


class RangeStartNegativeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(-10, 5, 4)

@register_test_case(module_factory=lambda: RangeStartNegativeIntModule())
def RangeStartNegativeIntModule_basic(module, tu: TestUtils):
    module.forward()   
  

class RangeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0)

@register_test_case(module_factory=lambda: RangeFloatModule())
def RangeFloatModule_basic(module, tu: TestUtils):
    module.forward()  


class RangeStepFloatModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0, 0.2)

@register_test_case(module_factory=lambda: RangeStepFloatModule1())
def RangeStepFloatModule1_basic(module, tu: TestUtils):
    module.forward()   


class RangeStepFloatModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(0.2, 0.8, 0.3)

@register_test_case(module_factory=lambda: RangeStepFloatModule2())
def RangeStepFloatModule2_basic(module, tu: TestUtils):
    module.forward()   

class RangeStepFloatModule3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(0.0, 1.6, 0.5)

@register_test_case(module_factory=lambda: RangeStepFloatModule3())
def RangeStepFloatModule3_basic(module, tu: TestUtils):
    module.forward()  

class RangeStepNegativeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(0.8, 0.2, -0.3)

@register_test_case(module_factory=lambda: RangeStepNegativeFloatModule())
def RangeStepNegativeFloatModule_basic(module, tu: TestUtils):
    module.forward()    


class RangeStartEndStepNegativeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(-1.0, -15.0, -3.4)

@register_test_case(module_factory=lambda: RangeStartEndStepNegativeFloatModule())
def RangeStartEndStepNegativeFloatModule_basic(module, tu: TestUtils):
    module.forward()      

    
class RangeEndStepNegativeFloatModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(10.0, -1.0, -3.0)

@register_test_case(module_factory=lambda: RangeEndStepNegativeFloatModule1())
def RangeEndStepNegativeFloatModule1_basic(module, tu: TestUtils):
    module.forward()


class RangeEndStepNegativeFloatModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(10.0, -1.0, -2.0)

@register_test_case(module_factory=lambda: RangeEndStepNegativeFloatModule2())
def RangeEndStepNegativeFloatModule2_basic(module, tu: TestUtils):
    module.forward()    


class RangeStartNegativeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(-11.0, 5.0, 7.0)

@register_test_case(module_factory=lambda: RangeStartNegativeFloatModule())
def RangeStartNegativeFloatModule_basic(module, tu: TestUtils):
    module.forward()        


class RangeStartEndIntStepFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1, 5, 1.0)

@register_test_case(module_factory=lambda: RangeStartEndIntStepFloatModule())
def RangeStartEndIntStepFloatModule_basic(module, tu: TestUtils):
    module.forward()


class RangeStartEndFloatStepIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0, 1)

@register_test_case(module_factory=lambda: RangeStartEndFloatStepIntModule())
def RangeStartEndFloatStepIntModule_basic(module, tu: TestUtils):
    module.forward()    

     
class RangeStartEndFloatDtypeInt64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0, dtype=torch.int64)

@register_test_case(module_factory=lambda: RangeStartEndFloatDtypeInt64Module())
def RangeStartEndFloatDtypeInt64Module_basic(module, tu: TestUtils):
    module.forward()          


class RangeStartEndFloatDtypeFloat32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0, dtype=torch.float32)

@register_test_case(module_factory=lambda: RangeStartEndFloatDtypeFloat32Module())
def RangeStartEndFloatDtypeFloat32Module_basic(module, tu: TestUtils):
    module.forward()              


class RangeStartEndFloatDtypeFloat64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.ops.aten.range(1.0, 5.0, dtype=torch.float64)

@register_test_case(module_factory=lambda: RangeStartEndFloatDtypeFloat64Module())
def RangeStartEndFloatDtypeFloat64Module_basic(module, tu: TestUtils):
    module.forward()