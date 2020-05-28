//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>

extern "C" void __npcomp_abort_if(bool b) {
  std::cerr << "Aborting if " << b << "\n";
  if (b)
    std::exit(1);
}
