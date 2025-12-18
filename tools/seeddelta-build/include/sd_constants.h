#pragma once

#include <string_view>

namespace sd_constants {

inline constexpr std::string_view decision_no_gating              = "no_gating";
inline constexpr std::string_view decision_metric_unavailable      = "metric_unavailable";
inline constexpr std::string_view decision_gating_fail             = "gating_fail";
inline constexpr std::string_view decision_pass_gating             = "pass_gating";
inline constexpr std::string_view decision_missing_tensor         = "missing_tensor";
inline constexpr std::string_view decision_missing_imatrix        = "missing_imatrix";
inline constexpr std::string_view decision_disabled               = "disabled";
inline constexpr std::string_view decision_stack_cost_cap         = "stack_cost_cap";
inline constexpr std::string_view decision_autotune_failed_keep_dense = "autotune_failed_keep_dense";

inline constexpr std::string_view ffn_proxy_reason_kind_not_supported = "proxy_kind_not_supported";
inline constexpr std::string_view ffn_proxy_reason_unavailable         = "proxy_unavailable";
inline constexpr std::string_view ffn_proxy_reason_requires_eval_x     = "proxy_requires_eval_x";
inline constexpr std::string_view ffn_proxy_reason_requires_coo       = "proxy_requires_coo";
inline constexpr std::string_view ffn_proxy_scope_replace_only_current_tensor = "replace_only_current_tensor";

inline constexpr std::string_view scheme_coo  = "coo";
inline constexpr std::string_view scheme_block = "block";

} // namespace sd_constants
