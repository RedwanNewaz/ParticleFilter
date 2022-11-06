//
// Created by Abdullah Al Redwan Newaz on 11/6/22.
//

#ifndef PARTICLEFILTER_PARTICLE_FILTER_H
#define PARTICLEFILTER_PARTICLE_FILTER_H
#include "filter_base.h"

#define PI 3.141592653
#define MAX_RANGE 20.0
#define NP 100
#define NTh NP/2


class particle_filter : public filter_base{

    using MAT_PARTICLES = Eigen::Matrix<float, STATE_DIM, NP>;
    using MAT_WEIGHTS = Eigen::Matrix<float, NP, 1>;
public:
    explicit particle_filter(float dt);


    void update(const std::vector<Eigen::RowVector4f>& z, const VEC_INPUT& ud);
    Eigen::VectorXf getState();


protected:

    static float gauss_likelihood(float x, float sigma);

    static MAT_COV calc_covariance(
            const VEC_STATE& xEst,
            const MAT_PARTICLES& px,
            const MAT_WEIGHTS& pw
            );
    MAT_WEIGHTS cumsum(const MAT_WEIGHTS& pw);

private:
    VEC_STATE xEst_;
    MAT_COV PEst_;
    MAT_PARTICLES px_;
    MAT_WEIGHTS pw_;


private:
    void pf_localization(const std::vector<Eigen::RowVector4f>& z,
                         const VEC_INPUT& ud
    );

    void pf_resampling();

};


#endif //PARTICLEFILTER_PARTICLE_FILTER_H
