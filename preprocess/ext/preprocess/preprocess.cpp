#include "preprocess.h"

#include <iostream>
#include <unordered_set>


std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map_test(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np,
                 array_i32_d patch_np, float bwd_depth_thresh,
                 bool invalid_depth_to_inf) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int tgt_height = tgt_dm_np.shape(0);
    int tgt_width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (src_dms_np.ndim() != 3) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    int n_views = src_dms_np.shape(0);
    int src_height = src_dms_np.shape(1);
    int src_width = src_dms_np.shape(2);
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }
    if (patch_np.ndim() != 1 || patch_np.shape(0) != 4) {
        throw std::invalid_argument("patch hast to be a 4 vector");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_Pi;
    tgt_Pi.leftCols<3>() = tgt_K.inverse();
    tgt_Pi.rightCols<1>() = -tgt_t;
    tgt_Pi = tgt_R.transpose() * tgt_Pi;

    depthmap_t tgt_dm(tgt_dm_np.data(), tgt_height, tgt_width);

    std::vector<proj_t> src_Ps;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(
            depthmap_t(src_dms_np.data() + vidx * src_height * src_width,
                       src_height, src_width));
    }

    int h_from = patch_np.data()[0];
    int h_to = patch_np.data()[1];
    int w_from = patch_np.data()[2];
    int w_to = patch_np.data()[3];
    int patch_height = h_to - h_from;
    int patch_width = w_to - w_from;

    std::vector<float> maps(n_views * patch_height * patch_width * 3,
                            float(-10));
    std::vector<float> valid_depth_masks(n_views * patch_height * patch_width,
                                         0);
    std::vector<float> valid_map_masks(n_views * patch_height * patch_width, 0);
    for (int tgt_h = 0; tgt_h < patch_height; ++tgt_h) {
        for (int tgt_w = 0; tgt_w < patch_width; ++tgt_w) {
            float dt = tgt_dm(tgt_h + h_from, tgt_w + w_from);
			if (dt <= 0) dt = 100;
            Eigen::Vector4f tgt_uvd(dt * (float(tgt_w + w_from) ),
                                    dt * (float(tgt_h + h_from) ), dt, 1);
            Eigen::Vector3f xyz = tgt_Pi * tgt_uvd;
            Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);
            int idx = (0 * patch_height + tgt_h) * patch_width + tgt_w;
            valid_map_masks[idx] = 1;
			maps[idx * 3 + 0] = xyz(0); 
			maps[idx * 3 + 1] = xyz(1); 
			maps[idx * 3 + 2] = xyz(2); 
        }
    }
    return std::make_tuple(create_arrayN<float>(maps, {n_views, patch_height, patch_width, 3}),
            create_arrayN<float>(valid_depth_masks,
                                 {n_views, 1, patch_height, patch_width}),
            create_arrayN<float>(valid_map_masks,
                                 {n_views, 1, patch_height, patch_width}));
}










std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map_train(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np,
                 array_i32_d patch_np, float bwd_depth_thresh,
                 bool invalid_depth_to_inf) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int tgt_height = tgt_dm_np.shape(0);
    int tgt_width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (src_dms_np.ndim() != 3) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    int n_views = src_dms_np.shape(0);
    int src_height = src_dms_np.shape(1);
    int src_width = src_dms_np.shape(2);
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }
    if (patch_np.ndim() != 1 || patch_np.shape(0) != 4) {
        throw std::invalid_argument("patch hast to be a 4 vector");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_Pi;
    tgt_Pi.leftCols<3>() = tgt_K.inverse();
    tgt_Pi.rightCols<1>() = -tgt_t;
    tgt_Pi = tgt_R.transpose() * tgt_Pi;

    depthmap_t tgt_dm(tgt_dm_np.data(), tgt_height, tgt_width);

    std::vector<proj_t> src_Ps;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(
            depthmap_t(src_dms_np.data() + vidx * src_height * src_width,
                       src_height, src_width));
    }

    int h_from = patch_np.data()[0];
    int h_to = patch_np.data()[1];
    int w_from = patch_np.data()[2];
    int w_to = patch_np.data()[3];
    int patch_height = h_to - h_from;
    int patch_width = w_to - w_from;

    std::vector<float> maps(n_views * patch_height * patch_width * 3,
                            float(-10));
    std::vector<float> valid_depth_masks(n_views * patch_height * patch_width,
                                         0);
    std::vector<float> valid_map_masks(n_views * patch_height * patch_width, 0);
    for (int tgt_h = 0; tgt_h < patch_height; ++tgt_h) {
        for (int tgt_w = 0; tgt_w < patch_width; ++tgt_w) {
            float dt = tgt_dm(tgt_h + h_from, tgt_w + w_from);
			if (dt <= 0) dt = 0;
            Eigen::Vector4f tgt_uvd(dt * (float(tgt_w + w_from) ),
                                    dt * (float(tgt_h + h_from) ), dt, 1);
            Eigen::Vector3f xyz = tgt_Pi * tgt_uvd;
            Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);
            int idx = (0 * patch_height + tgt_h) * patch_width + tgt_w;
            valid_map_masks[idx] = 1;
			maps[idx * 3 + 0] = xyz(0); 
			maps[idx * 3 + 1] = xyz(1); 
			maps[idx * 3 + 2] = xyz(2); 
        }
    }
    return std::make_tuple(create_arrayN<float>(maps, {n_views, patch_height, patch_width, 3}),
            create_arrayN<float>(valid_depth_masks,
                                 {n_views, 1, patch_height, patch_width}),
            create_arrayN<float>(valid_map_masks,
                                 {n_views, 1, patch_height, patch_width}));
}
