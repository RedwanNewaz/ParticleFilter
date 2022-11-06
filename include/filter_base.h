//
// Created by Abdullah Al Redwan Newaz on 11/6/22.
//

#ifndef PARTICLEFILTER_FILTER_BASE_H
#define PARTICLEFILTER_FILTER_BASE_H
#include<iostream>
#include<vector>
#include<random>
#include<cmath>
#include<Eigen/Eigen>
#include <memory>



class filter_base{
public:
    using VEC_STATE = Eigen::Vector4f;
    using MAT_COV = Eigen::Matrix4f;
    using VEC_INPUT = Eigen::Vector2f;

    filter_base(float dt);

    Eigen::Vector4f motion_model(const VEC_STATE& x, const VEC_INPUT& u);

    Eigen::Vector2f observation_model(const VEC_STATE& x);

private:
    float dt_;
    std::unique_ptr<std::mt19937> gen_;
    std::unique_ptr<std::uniform_real_distribution<>> uni_d_;

protected:
    /**
     * agitated each particle state with uniformly random noise
     * @param NP number of particles
     * @return uniform random number [0, 1]
     */
    float perturbation(int NP);
    const float Q = 0.01;
};
#endif //PARTICLEFILTER_FILTER_BASE_H
