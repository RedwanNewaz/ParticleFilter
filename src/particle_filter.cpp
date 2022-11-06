//
// Created by Abdullah Al Redwan Newaz on 11/6/22.
//

#include "particle_filter.h"

void particle_filter::pf_localization(const std::vector<Eigen::RowVector3f>& z,
                                      const VEC_INPUT& ud) {

        for(int ip=0; ip<NP; ip++){
            VEC_STATE x = px_.col(ip);
            float w = pw_(ip);

            x = motion_model(x, ud);

            for(unsigned int i=0; i<z.size(); i++){
                Eigen::RowVector3f item = z[i];
                float dx = x(0) - item(1);
                float dy = x(1) - item(2);
                float prez = std::sqrt(dx*dx + dy*dy);
                float dz = prez - item(0);
                w = w * gauss_likelihood(dz, std::sqrt(Q));
            }
            px_.col(ip) = x;
            pw_(ip) = w;
        }

        pw_ = pw_ / pw_.sum();

        xEst_ = px_ * pw_;
        PEst_ = calc_covariance(xEst_, px_, pw_);

}

float particle_filter::gauss_likelihood(float x, float sigma) {
    float p = 1.0 / std::sqrt(2.0 * PI * sigma * sigma) * \
      std::exp(-x * x / (2 * sigma * sigma));
    return p;
}

void particle_filter::pf_resampling() {
    float Neff = 1.0 / (pw_.transpose() * pw_);
    if (Neff < NTh) {
        MAT_WEIGHTS wcum = cumsum(pw_);
        MAT_WEIGHTS base = cumsum(pw_ * 0.0 + MAT_WEIGHTS::Ones() * 1.0 / NP) -
                                           MAT_WEIGHTS::Ones() * 1.0 / NP;
        MAT_WEIGHTS resampleid;
        MAT_PARTICLES output;
        for (int j = 0; j < pw_.rows(); j++) {
            resampleid(j) = base(j) + perturbation(NP);
        }

        int ind = 0;

        for (int i = 0; i < NP; i++) {
            while (resampleid(i) > wcum(ind) && ind < NP - 1) {
                ind += 1;
            }
            output.col(i) = px_.col(ind);
        }

        px_ = output;
        pw_ = MAT_WEIGHTS::Ones() * 1.0 / NP;
    }

}

particle_filter::MAT_COV particle_filter::calc_covariance(const VEC_STATE& xEst,
                                                 const MAT_PARTICLES& px,
                                                 const MAT_WEIGHTS& pw) {

        MAT_COV PEst_ = MAT_COV::Zero();
        for(int i=0; i<px.cols(); i++){
            VEC_STATE dx = px.col(i) - xEst;
            PEst_ += pw(i) * dx * dx.transpose();
        }

        return PEst_;
}

particle_filter::MAT_WEIGHTS particle_filter::cumsum(const MAT_WEIGHTS& pw) {
    MAT_WEIGHTS cum;
    cum(0) = pw(0);
    for(int i=1; i<pw.rows(); i++){
        cum(i) = cum(i-1) + pw(i);
    }
    return cum;
}

particle_filter::particle_filter(float dt) :
filter_base(dt) {

    xEst_ = VEC_STATE::Zero();
    PEst_ = MAT_COV::Identity();
    px_ = MAT_PARTICLES::Zero();
    pw_ = MAT_WEIGHTS::Ones() * 1.0/NP;
}

void particle_filter::update(const std::vector<Eigen::RowVector3f>& z, const VEC_INPUT& ud) {

    xEst_ = motion_model(xEst_, ud);
    pf_localization(z, ud);
    pf_resampling();
    std::cout << "[state] " << observation_model(xEst_).transpose() << std::endl;

}

Eigen::VectorXf particle_filter::getState() {
    return xEst_;
}
