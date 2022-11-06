//
// Created by Abdullah Al Redwan Newaz on 11/6/22.
//
#include "filter_base.h"

Eigen::Vector4f filter_base::motion_model(const VEC_STATE& x, const VEC_INPUT& u) {
    MAT_COV F_ = MAT_COV::Identity();


    Eigen::Matrix<float, STATE_DIM, INPUT_DIM> B_;
    B_<< dt_ * std::cos(x(2,0)),  0,
            dt_ * std::sin(x(2,0)),  0,
            0.0,  dt_,
            1.0,  0.0;

    return F_ * x + B_ * u;
}



filter_base::filter_base(float dt): dt_(dt) {
    std::random_device rd2{};
    uni_d_ = std::make_unique<std::uniform_real_distribution<>>(1.0, 2.0);
    gen_ = std::make_unique<std::mt19937>(rd2());

}

float filter_base::perturbation(int NP) {
    return uni_d_->operator()(*gen_) / NP;
}

Eigen::Vector2f filter_base::observation_model(const VEC_STATE& x) {
    Eigen::Matrix<float, INPUT_DIM, STATE_DIM> H_;
    H_<< 1, 0, 0, 0,
            0, 1, 0, 0;
    return H_ * x;
}
