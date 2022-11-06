//
// Created by Abdullah Al Redwan Newaz on 11/6/22.
//
#include "filter_base.h"

Eigen::VectorXf filter_base::motion_model(const VEC_STATE& x, const VEC_INPUT& u) {
    MAT_COV F_ = MAT_COV::Identity();


    Eigen::Matrix<float, STATE_DIM, INPUT_DIM> B_;
//    B_<< dt_ * std::cos(x(2,0)),  0,
//            dt_ * std::sin(x(2,0)),  0,
//            0.0,  dt_,
//            1.0,  0.0;

    float theta = x(3, 0);
    float phi = x(4, 0);

    // populate x, y, z
    B_(0, 0) = sin(theta) * cos(phi);
    B_(1, 0) = sin(theta) * sin(phi);
    B_(2,0) = cos(theta);

    // populate p, q, r
    float d = sqrt(B_(1,0) * B_(1,0) + B_(0,0) * B_(0,0));
    B_(3, 0) = atan2(B_(1,0), B_(0,0));
    B_(4, 0) = atan2(d, B_(2, 0));
    B_(5,3) = dt_;

    // derivative of x, y, z, r
    B_(6, 0) = 1;
    B_(7, 0) = 1;
    B_(8, 0) = 1;
    B_(11, 0) = 1;

    return F_ * x + B_ * u;;
}



filter_base::filter_base(float dt): dt_(dt) {
    std::random_device rd2{};
    uni_d_ = std::make_unique<std::uniform_real_distribution<>>(1.0, 2.0);
    gen_ = std::make_unique<std::mt19937>(rd2());

}

float filter_base::perturbation(int NP) {
    return uni_d_->operator()(*gen_) / NP;
}

Eigen::VectorXf filter_base::observation_model(const VEC_STATE& x) {
    Eigen::Matrix<float, INPUT_DIM, STATE_DIM> H_;
//    H_<< 1, 0, 0, 0,
//            0, 1, 0, 0;

    H_(0, 0) = 1;
    H_(1, 1) = 1;
    H_(2, 2) = 1;
    H_(3, 5) = 1;

    return H_ * x;
}
