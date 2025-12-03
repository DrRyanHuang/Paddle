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

import paddle
from paddle.static import MetaTensor


def custom_add(x: paddle.Tensor, y: paddle.Tensor, const: int):
    return x + y + const


def custom_add_infer_meta(
    x_meta: MetaTensor, y_meta: MetaTensor, const: int
) -> MetaTensor:
    out_meta = MetaTensor()
    out_meta.set_dtype(x_meta.dtype)
    out_meta.set_shape(x_meta.shape)
    return out_meta
