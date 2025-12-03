# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import types
import unittest
from typing import Callable, ParamSpec, TypeVar

HAS_ARGS_KWARGS: int = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


P1 = ParamSpec("P1")
R1 = TypeVar("R1")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")
T = TypeVar("T")


class MissingArgument:
    def __init__(self, fn: Callable[P1, R1], name: str):
        self.fn = fn
        self.name = name

    def __repr__(self):
        return f"<Required parameter '{self.name}' for function {self.fn.__name__}>"


def extract_default(fn: Callable[P1, R1], parameter: inspect.Parameter):
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        return ()
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        return {}
    elif parameter.default is inspect.Parameter.empty:
        return MissingArgument(fn, parameter.name)
    return parameter.default


def get_fn_defaults_params(fn: Callable[P1, R1]) -> tuple:
    fn_defaults_params = [
        extract_default(fn, param)
        for param in inspect.signature(fn).parameters.values()
    ]
    for i, default in enumerate(fn_defaults_params):
        if not isinstance(default, MissingArgument):
            fn_defaults_params = fn_defaults_params[i:]
            break
    return tuple(fn_defaults_params)


def eliminate_positional_only(fn: Callable[P1, R1]) -> Callable[P1, R1]:
    code = fn.__code__
    co_flags: int = code.co_flags & ~HAS_ARGS_KWARGS
    co_flags = code.co_flags

    # TODO: currently, only support Python3.10
    if hasattr(code, "co_posonlyargcount"):
        argcount = (
            code.co_argcount
            + code.co_kwonlyargcount
            + bool(code.co_flags & inspect.CO_VARARGS)
            + bool(code.co_flags & inspect.CO_VARKEYWORDS)
        )

        new_code = types.CodeType(
            argcount,  # co_argcount
            0,  # posonlyargcount
            0,  # kwonlyargcount
            code.co_nlocals,
            code.co_stacksize,
            co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )
    else:
        raise ValueError

    fn_defaults_params = get_fn_defaults_params(fn)
    new_fn = types.FunctionType(
        new_code,
        fn.__globals__,
        fn.__name__,
        fn_defaults_params,
        fn.__closure__,
    )
    new_fn.__name__ = fn.__name__
    new_fn.__doc__ = fn.__doc__
    new_fn.__annotations__ = fn.__annotations__
    new_fn.__kwdefaults__ = None
    return new_fn


class TestEliminatePositionalOnly(unittest.TestCase):
    def test_no_positional_only_args(self):
        def normal_func(a, b, c=3):
            return a + b + c

        result_func = eliminate_positional_only(normal_func)

        self.assertEqual(result_func(1, 2), normal_func(1, 2))
        self.assertEqual(result_func(1, 2, 3), normal_func(1, 2, 3))
        self.assertEqual(result_func(a=1, b=2), normal_func(a=1, b=2))

    def test_with_positional_only_args(self):
        def pos_only_func(a, b, /, c, d=4, *, f=6, e=5, g=8):
            return a + b + c + d + e + f + g

        result_func = eliminate_positional_only(pos_only_func)

        with self.assertRaises(TypeError):
            pos_only_func(a=1, b=2, c=3)

        self.assertEqual(result_func(1, 2, 3), pos_only_func(1, 2, 3))
        self.assertEqual(result_func(1, 2, c=3), pos_only_func(1, 2, c=3))
        self.assertEqual(
            result_func(1, 2, c=3, d=10), pos_only_func(1, 2, c=3, d=10)
        )
        self.assertEqual(
            result_func(a=1, b=2, c=3, e=10), pos_only_func(1, 2, c=3, e=10)
        )

    def test_mixed_parameter_types(self):
        def complex_func(a, b, /, c, d=4, *, e=5, f, **kwargs):
            return a + b + c + d + e + f + sum(kwargs.values())

        result_func = eliminate_positional_only(complex_func)

        with self.assertRaises(TypeError):
            complex_func(a=1, b=2, c=3, f=6)

        self.assertEqual(
            result_func(a=1, b=2, c=3, f=6), complex_func(1, 2, 3, f=6)
        )
        self.assertEqual(result_func(1, 2, 3, f=6), complex_func(1, 2, 3, f=6))

    def test_function_attributes_preserved(self):
        def func_with_attrs(a: int, b: int, /, c: int = 3) -> int:
            """__doc__"""
            return a + b + c

        result_func = eliminate_positional_only(func_with_attrs)

        self.assertEqual(result_func.__name__, func_with_attrs.__name__)
        self.assertEqual(result_func.__doc__, func_with_attrs.__doc__)
        self.assertEqual(
            result_func.__annotations__,
            {"a": int, "b": int, "c": int, "return": int},
        )
        self.assertEqual(
            result_func.__kwdefaults__, func_with_attrs.__kwdefaults__
        )

    def test_no_parameters(self):
        def no_params():
            return 42

        result_func = eliminate_positional_only(no_params)
        self.assertEqual(result_func(), no_params())

    def test_only_positional_only(self):
        def only_pos_only(a, b, /):
            return a * b

        result_func = eliminate_positional_only(only_pos_only)
        with self.assertRaises(TypeError):
            only_pos_only(a=2, b=3)

        self.assertEqual(result_func(a=2, b=3), only_pos_only(2, 3))
        self.assertEqual(result_func(2, 3), only_pos_only(2, 3))

    def test_keyword_only_remain_unchanged(self):
        def kw_only_func(a, /, b, *, c):
            return a + b + c

        result_func = eliminate_positional_only(kw_only_func)
        with self.assertRaises(TypeError):
            kw_only_func(1, 2, 3)

        self.assertEqual(result_func(1, 2, 3), kw_only_func(1, 2, c=3))

    def test_default_values(self):
        def func_with_defaults(a, b=10, /, c=20, *, d=30):
            return a + b + c + d

        result_func = eliminate_positional_only(func_with_defaults)

        self.assertEqual(result_func(1), func_with_defaults(1))
        self.assertEqual(result_func(1, 2), func_with_defaults(1, 2))
        self.assertEqual(result_func(a=1, c=25), func_with_defaults(1, c=25))

    def test_closure_and_scope(self):
        outer_var = 10

        def closure_func(a, b, /):
            return a + b + outer_var

        result_func = eliminate_positional_only(closure_func)
        self.assertEqual(result_func(1, 2), closure_func(1, 2))
        self.assertEqual(result_func(a=1, b=2), closure_func(1, 2))

    def test_edge_case_all_parameter_kinds(self):
        def all_kinds(a, b, /, c, d=4, *args, e=5, f, **kwargs):
            total = a + b + c + d + e + f

            total += sum(args)
            total += sum(kwargs.values())
            return total

        result_func = eliminate_positional_only(all_kinds)

        self.assertEqual(
            result_func(1, 2, 3, f=6), all_kinds(1, 2, 3, f=6)
        )  # 1+2+3+4+5+6
        # self.assertEqual(result_func(a=1, b=2, c=3, f=6), 21)
        # self.assertEqual(result_func(1, 2, 3, 10, f=6), 31)  # 1+2+3+10+5+6
        # self.assertEqual(result_func(1, 2, 3, f=6, g=10), 31)  # 1+2+3+4+5+6+10


if __name__ == "__main__":
    unittest.main()


def infer_meta(x_meta, y_meta):
    out_meta = x_meta
    out_meta.set_dtype("float32")
    return out_meta
