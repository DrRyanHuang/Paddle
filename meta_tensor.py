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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ParamSpec

HAS_ARGS_KWARGS: int = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


P1 = ParamSpec("P1")
R1 = TypeVar("R1")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")
T = TypeVar("T")


T = TypeVar("T")


class Attribute(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get_data(self) -> T:
        return self.value

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.value}")'


class StringAttribute(Attribute[str]): ...


class IntAttribute(Attribute[int]): ...


class FloatAttribute(Attribute[float]): ...


class ArrayAttribute(Attribute[list[Attribute]]): ...


class Value: ...


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
    if code.co_flags & inspect.CO_VARARGS:
        raise ValueError("Currently, not support functions with *args")

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


class CustomPyOp:
    def __init__(
        self,
        name: str,
        inputs: list[Value],
        outputs: list[Value],
        attributes: dict[str, Attribute],
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes


class Tensor: ...


@dataclass
class MetaTensor:
    shape: Tuple[int, ...]
    dtype: str
    device: str

    stop_gradient: bool = True
    requires_grad: bool = False
    stride: Optional[Tuple[int, ...]] = None
    storage_offset: int = 0

    def __post_init__(self):
        self.stop_gradient = not self.requires_grad


def in_dynamic_mode() -> bool: ...


def bind_constants(fn, type, *args, **kwargs):
    # 将非 Tensor 变量 bind
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    params = bound_args.arguments

    mutable_params = {
        k: v
        for k, v in params.items()
        if judge_tensor_or_vector_of_tensor(v, type)
    }
    mutable_arg_names = list(mutable_params.keys())
    const_params = {
        k: v for k, v in params.items() if not isinstance(v, Tensor)
    }

    fn = eliminate_positional_only(fn)
    return mutable_arg_names, partial(fn, **const_params)


def call_fn_with_mutable_args(fn, args, mutable_arg_names: list[str]):
    assert len(args) == len(mutable_arg_names)
    kwargs = {name: arg for name, arg in zip(mutable_arg_names, args)}
    return fn(**kwargs)


def judge_tensor_or_vector_of_tensor(v: Any, type=Tensor):
    if isinstance(v, type):
        return True
    elif isinstance(v, list):
        return all([judge_tensor_or_vector_of_tensor(item) for item in v])
    else:
        return False


@overload
def register_op(
    fn: Callable[P1, R1],
    /,
    *,
    infer_meta_fn: Callable[..., Any] | None,
) -> Callable[P1, R1]: ...


@overload
def register_op(
    fn: None,
    /,
    *,
    infer_meta_fn: Callable[..., Any] | None,
) -> Callable[[Callable[P1, R1]], Callable[P1, R1]]: ...


def register_op(
    fn=None,
    /,
    *,
    infer_meta_fn=None,
):
    def _register_op(
        fn: Callable[P1, R1],
    ) -> Callable[P1, R1]:
        @wraps(fn)
        def wrapped_fn(*args: P1.args, **kwargs: P1.kwargs) -> R1:
            if in_dynamic_mode():
                return fn(*args, **kwargs)
            # Static mode handling can be added here

            mutable_arg_names, bound_constants_fn = bind_constants(
                my_custom_op, Tensor
            )
            mutable_arg_names_from_infer_meta, bound_constants_fn_infer_meta = (
                bind_constants(my_custom_op_infer_meta, MetaTensor)
            )
            assert mutable_arg_names == mutable_arg_names_from_infer_meta

            fn = CustomPyOp(
                name="my_custom_op",
                inputs=[Value(), Value(), Value()],
                outputs=[Value()],
                attributes={
                    "infer_meta_fn_ptr": IntAttribute(
                        cast(
                            "int",
                            incref(
                                bind_constants(
                                    my_custom_op_infer_meta, MetaTensor
                                )
                            ),
                        )
                    ),
                    "fn_ptr": IntAttribute(
                        cast(
                            "int", incref(bind_constants(my_custom_op, Tensor))
                        )
                    ),
                    "mutable_arg_names": ArrayAttribute(
                        [StringAttribute(name) for name in mutable_arg_names]
                    ),
                },
            )

            return fn(*args, **kwargs)

        return wrapped_fn

    if fn is None:
        return _register_op
    return _register_op(fn)


def my_custom_op_infer_meta(
    x: MetaTensor,
    y: MetaTensor,
    z: list[MetaTensor],
    /,
    attr1: str,
    attr2: list[str],
    attr3: list[tuple[int, int]] | None = None,
    *,
    attr4: int = 42,
) -> MetaTensor: ...


@register_op(infer_meta_fn=my_custom_op_infer_meta)
def my_custom_op(
    x: Tensor,
    y: Tensor,
    z: list[Tensor],
    /,
    attr1: str,
    attr2: list[str],
    attr3: list[tuple[int, int]] | None = None,
    *,
    attr4: int = 42,
): ...


def incref(obj):
    # Dummy implementation of incref
    # This should call Py_INCREF in actual C extension
    return obj
