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

#include "paddle/fluid/framework/custom_pyoperator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include <utility>  // for std::move
// #include <algorithm>
// #include <functional>
// #include <iostream>
// #include <map>
// #include <string>
// #include <tuple>
// #include <unordered_map>
// #include <unordered_set>
// #include <utility>
// #include <vector>

#include "paddle/phi/api/ext/op_meta_info.h"
// #include "paddle/fluid/eager/api/utils/global_utils.h"
// #include "paddle/fluid/framework/attribute.h"
// #include "paddle/fluid/framework/convert_utils.h"
// #include "paddle/fluid/framework/phi_utils.h"
// #include "paddle/fluid/framework/tensor.h"
// #include "paddle/phi/api/all.h"
// #include "paddle/phi/backends/dynload/dynamic_loader.h"
// #include "paddle/phi/core/compat/convert_utils.h"
// #include "paddle/phi/core/platform/device/gpu/gpu_info.h"
// #include "paddle/phi/core/tensor_utils.h"
// #include "paddle/utils/any.h"
// #include "paddle/utils/string/string_helper.h"

// #include "paddle/common/flags.h"
// #include "paddle/phi/api/include/operants_manager.h"
// #include "paddle/phi/api/include/tensor_operants.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"

// Is necessary for this file?
COMMON_DECLARE_string(tensor_operants_mode);
COMMON_DECLARE_bool(enable_pir_in_executor);

namespace paddle::framework {






// load op api
void RegisterPyOperator(
    const std::string& op_name,
    std::vector<std::string>&& op_inputs,
    std::vector<std::string>&& op_outputs,
    std::vector<std::string>&& op_attrs,
    std::unordered_map<std::string, std::string>&& op_inplace_map,
    WrapPythonFunction func,
    WrapInferMetaPythonFunction infer_meta____) {
  ::paddle::OpMetaInfoBuilder __op_meta_info_builder =
      ::paddle::OpMetaInfoBuilder(std::string(op_name), 0);
  __op_meta_info_builder.Inputs(std::move(op_inputs))
      .Outputs(std::move(op_outputs))
      .Attrs(std::move(op_attrs))
      .SetInplaceMap(std::move(op_inplace_map));

  __op_meta_info_builder.SetPyCustomPyOpFunction(func);
  __op_meta_info_builder.SetPyCustomPyOpInferMetaFunction(infer_meta____);

  const std::vector<paddle::OpMetaInfo>& op_meta_info_vector =
      OpMetaInfoMap::Instance()[op_name];

  if (op_meta_info_vector.size() != 1) {
    // PADDLE_THROW(paddle::platform::errors::InvalidArgument(
    //     "Currently, PaddlePaddle only supports operators with one meta
    //     info."));
    std::cout << "WTF! "
              << "op_meta_info_vector.size() " << op_meta_info_vector.size()
              << std::endl;
  }
  const auto& op_meta_info = op_meta_info_vector.back();

  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(op_meta_info);
  auto postfix = inplace_map.empty() ? "" : "_";

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  auto* custom_pyop_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomPyOpDialect>();

  // Custom dialect register
  if (custom_pyop_dialect->HasRegistered(
          paddle::framework::kCustomPyDialectPrefix + op_name + postfix)) {
    std::cout << "The operator `" << op_name
              << "` has been registered. "
                 "Therefore, we will not repeat the registration here.";
    return;
  }

  std::cout << "register pir custom op : "
            << OpMetaInfoHelper::GetOpName(op_meta_info) << std::endl;
  custom_pyop_dialect->RegisterCustomPyOp(op_meta_info);

  // auto diff_map = RegisterOperatorWithMetaInfoMap(op_meta_info_map, handle);
}

}  // namespace paddle::framework
