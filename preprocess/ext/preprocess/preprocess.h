#pragma once

#include "common.h"


std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map_train(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np, array_i32_d patch,
                 float bwd_depth_thresh, bool invalid_depth_to_inf);

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map_test(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np, array_i32_d patch,
                 float bwd_depth_thresh, bool invalid_depth_to_inf);
