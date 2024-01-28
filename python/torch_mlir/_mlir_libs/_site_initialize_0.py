# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Multi-threading rarely helps the frontend and we are also running in contexts
# where we want to run a lot of test parallelism (and nproc*nproc threads
# puts a large load on the system and virtual memory).
disable_multithreading = True
