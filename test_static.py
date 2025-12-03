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

import numpy as np

import paddle
from paddle import Tensor
from paddle.static import MetaTensor
from paddle.static.custom_pyop import register_op



def add_infer_meta(x_meta: MetaTensor, y_meta: MetaTensor) -> MetaTensor:
    z_meta = MetaTensor(shape=x_meta.shape, dtype=x_meta.dtype)
    return z_meta

@register_op(
    name="custom_add",
    infer_meta=add_infer_meta,
    input_names=["x", "y"],
    output_names=["z"],
    inplace_map={},
)
def add_fn(x: Tensor, y: Tensor) -> Tensor:
    # return x + y
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    return paddle.to_tensor(x + y).astype(x.dtype)


@paddle.jit.to_static(full_graph=False)
def mul_add(x, y, z):
    return add_fn(x, y) * z


if __name__ == "__main__":
    # 再包一个装饰器
    paddle.enable_static()

    x = paddle.static.data(name="x", shape=[4, 4, 5], dtype="float32")
    y = paddle.static.data(name="y", shape=[4, 4, 5], dtype="float32")
    z = add_fn(x, y)

    print(z)
    print(paddle.static.default_main_program())

    place = paddle.CUDAPlace(0)
    # place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    x_np = np.random.random(size=(4, 4, 5)).astype("float16")
    y_np = np.random.random(size=(4, 4, 5)).astype("float16")

    (z_np,) = exe.run(
        feed={"x": x_np, "y": y_np},
        fetch_list=[z],
    )
    print(z_np)

    np.testing.assert_allclose(z_np, x_np + y_np)

