// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
// #include "paddle/fluid/framework/custom_operator_utils.h"
// #include "paddle/fluid/framework/op_registry.h"
// #include "paddle/fluid/framework/operator.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include <functional>

namespace paddle {
namespace framework {

using WrapPythonFunction = std::function<std::vector<Tensor>(std::vector<Tensor>&)>;
// // Register custom op api: register op directly
// std::unordered_map<std::string, std::vector<OpMetaInfo>>
// RegisterOperatorWithMetaInfoMap(const paddle::OpMetaInfoMap&
// op_meta_info_map,
//                                 void* dso_handle = nullptr);

using IrTensor = paddle::dialect::IrTensor;
using WrapInferMetaPythonFunction = std::function<std::vector<IrTensor>(const std::vector<paddle::dialect::IrTensor>&)>;

void RegisterPyOperator(
    const std::string& op_name,
    std::vector<std::string>&& op_inputs,
    std::vector<std::string>&& op_outputs,
    std::vector<std::string>&& op_attrs,
    std::unordered_map<std::string, std::string>&& op_inplace_map,
    WrapPythonFunction func,
    WrapInferMetaPythonFunction infer_meta);

}  // namespace framework
}  // namespace paddle
