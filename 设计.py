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

from collections.abc import Callable
from typing import TypeVar, overload, Any, cast
from typing_extensions import ParamSpec
from functools import wraps
import inspect
from functools import partial

P1 = ParamSpec("P1")
R1 = TypeVar("R1")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")


class Attribute: ...


class StringAttribute(Attribute):
    def __init__(self, value: str):
        self.value = value

    def get_data(self) -> str:
        return self.value

    def __repr__(self):
        return f'StringAttribute("{self.value}")'


class ArrayAttribute(Attribute):
    def __init__(self, values: list[Attribute]):
        self.values = values

    def get_data(self) -> list[Attribute]:
        return self.values

    def __repr__(self):
        return f"ArrayAttribute({self.values})"


class IntAttribute(Attribute):
    def __init__(self, value: int):
        self.value = value

    def get_data(self) -> int:
        return self.value

    def __repr__(self):
        return f"IntAttribute({self.value})"


class Value: ...


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


class MetaTensor: ...


def in_dynamic_mode() -> bool: ...


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
) -> MetaTensor | list[MetaTensor]: ...

    return MetaTensor([x.shape[1], y.shape[0]], dtype=x.dtype)


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


def eliminate_positional_only(fn: Callable[P1, R1]) -> Callable[P1, R1]:
    # Remove positional-only parameters, make all parameters keyword or positional-or-keyword
    # TODO: implement this function
    # This should modify the code and reconstruct the function object
    return fn


def incref(obj):
    # Dummy implementation of incref
    # This should call Py_INCREF in actual C extension
    return obj


def bind_constants(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    params = bound_args.arguments

    mutable_params = {k: v for k, v in params.items() if isinstance(v, Tensor)}
    mutable_arg_names = list(mutable_params.keys())
    const_params = {k: v for k, v in params.items() if not isinstance(v, Tensor)}

    fn = eliminate_positional_only(fn)
    return mutable_arg_names, partial(fn, **const_params)


def call_fn_with_mutable_args(fn, args, mutable_arg_names: list[str]):
    assert len(args) == len(mutable_arg_names)
    kwargs = {name: arg for name, arg in zip(mutable_arg_names, args)}
    return fn(**kwargs)


CustomPyOp(
    name="my_custom_op",
    inputs=[Value(), Value(), Value()],
    outputs=[Value()],
    attributes={
        "$infer_meta_fn_ptr": IntAttribute(cast(int, my_custom_op_infer_meta)),
        "$fn_ptr": IntAttribute(cast(int, my_custom_op)),
        "attr1": StringAttribute("example"),
        "attr2": ArrayAttribute([StringAttribute(s) for s in ["a", "b", "c"]]),
        "attr3": ArrayAttribute(
            [ArrayAttribute([IntAttribute(i) for i in t]) for t in [(1, 2), (3, 4)]]
        ),
        "attr4": IntAttribute(42),
    },
)
# ->
mutable_arg_names, bound_constants_fn = bind_constants(my_custom_op)
mutable_arg_names_from_infer_meta, bound_constants_fn_infer_meta = bind_constants(
    my_custom_op_infer_meta
)
assert mutable_arg_names == mutable_arg_names_from_infer_meta
CustomPyOp(
    name="my_custom_op",
    inputs=[Value(), Value(), Value()],
    outputs=[Value()],
    attributes={
        "infer_meta_fn_ptr": IntAttribute(
            cast(int, incref(bind_constants(my_custom_op_infer_meta)))
        ),
        "fn_ptr": IntAttribute(cast(int, incref(bind_constants(my_custom_op)))),
        "mutable_arg_names": ArrayAttribute(
            [StringAttribute(name) for name in mutable_arg_names]
        ),
    },
)
