# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import factory

from gtc.cuir import cuir

from .common_utils import identifier, undefined_symbol_list


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = cuir.FieldDecl

    name = identifier(cuir.FieldDecl)
    dtype = cuir.DataType.FLOAT32


class ProgramFactory(factory.Factory):
    class Meta:
        model = cuir.Program

    name = identifier(cuir.Program)
    params = undefined_symbol_list(
        lambda name: FieldDeclFactory(name=name), "kernels", "temporaries"
    )
    temporaries = factory.List([])
    kernels = factory.List([])