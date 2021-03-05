# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""GTGraph decorator

Interface functions to define the 'gtgraph' decorator to construct dataflow
graphs from functions that call gtscript stencils.
"""

import ast
import inspect
import networkx as nx
import types
from typing import Any, Callable, Dict, Tuple


_graphs: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def gtgraph(definition=None, **stencil_kwargs) -> Tuple[Any, Dict[str, Any]]:
    def decorator(definition) -> Callable[..., None]:
        def_name = f"{definition.__module__}.{definition.__name__}"
        graph, meta_data = GraphMaker.apply(definition)
        _graphs[def_name] = (graph, meta_data)

        return _graphs[def_name]

    if definition is None:
        return decorator
    else:
        return decorator(definition)


class GraphMaker(ast.NodeVisitor):
    @classmethod
    def apply(cls, definition):
        maker = cls(definition)
        maker(maker.ast_root)
        return maker.graph, maker.meta_data

    def __init__(self, definition):
        assert isinstance(definition, types.FunctionType)
        self.definition = definition
        self.filename = inspect.getfile(definition)
        self.source = inspect.getsource(definition)
        self.ast_root = ast.parse(self.source)
        self.graph = nx.DiGraph()
        self.meta_data: Dict[str, Any] = {}

    def __call__(self, func_node: ast.FunctionDef):
        self.visit(func_node)

    def __contains__(self, node_name: str) -> bool:
        return node_name in self.meta_data

    def _add_node(self, node_name: str, node_meta: Any) -> None:
        self.graph.add_node(node_name)
        self.meta_data[node_name] = dict(node=node_meta)

    def visit(self, node, **kwargs):
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, **kwargs)

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs):
        for arg in node.args.args:
            self.visit(arg)
        for entry in node.body:
            self.visit(entry)

    def visit_arg(self, node: ast.arg, **kwargs):
        arg_name = node.arg
        if isinstance(node.annotation, ast.Subscript):
            container_type = node.annotation.value.id.lower()
            if container_type == "dict":
                arg_type = node.annotation.slice.value.elts[1].value
        else:
            arg_type = node.annotation.id
        if "Field" in arg_type:
            self._add_node(arg_name, node)

    def visit_Assign(self, node: ast.Assign, **kwargs):
        if isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = func.attr if isinstance(func, ast.Attribute) else func.id
            # TODO: This is very fv3core-specific, need to generalize for gt4py...
            if func_name == "copy" or func_name.startswith("make_storage"):
                for target in node.targets:
                    self._add_node(target.id, node)

    def visit_Call(self, node: ast.Call, **kwargs):
        func = node.func
        is_attr = isinstance(func, ast.Attribute)
        func_name = func.attr if is_attr else func.id
        if "Exception" not in func_name:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    arg_name = arg.id
                    if arg_name in self:
                        if func_name not in self:
                            self._add_node(func_name, node)
                        # TODO: How to determine if in/out?
                        self.graph.add_edge(arg.id, func_name)

            for keyword in node.keywords:
                arg_name = keyword.arg
                if arg_name in self:
                    if func_name not in self:
                        self._add_node(func_name, node)
                    self.graph.add_edge(arg.id, func_name)
                elif arg_name == "domain" or arg_name == "origin":
                    self.meta_data[func_name][arg_name] = keyword.value
