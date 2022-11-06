#include <iostream>
#include "particle_filter.h"

#define SIM_TIME 50.0
#define DT 0.1
const float Qsim = 0.04;
std::vector<Eigen::RowVector3f> get_observations(const Eigen::Vector4f& xEst,
                                                 const Eigen::Matrix<float, 4, 2>& Landmarks,
                                                 std::mt19937& gen,
                                                 std::normal_distribution<>& gaussian_d
                                                 )
{
    std::vector<Eigen::RowVector3f> z;
    for(int i=0; i<Landmarks.rows(); i++){
        float dx = xEst(0) - Landmarks(i, 0);
        float dy = xEst(1) - Landmarks(i, 1);
        float d = std::sqrt(dx*dx + dy*dy);
        if (d <= MAX_RANGE){
            float dn = d + gaussian_d(gen) * Qsim;
            Eigen::RowVector3f zi;
            // sensor reading dn, ground truth landmarks
            zi<<dn, Landmarks(i, 0), Landmarks(i, 1);
            z.push_back(zi);
        }
    }
    return z;
}

int main() {
    std::cout << "Particle Filter!" << std::endl;
    float time=0.0;

    // control input
    Eigen::Vector2f u;
    u<<1.0, 0.1;

    // Landmarks remarks
    Eigen::Matrix<float, 4, 2> Landmarks;
    Landmarks<<10.0, 0.0,
            10.0, 10.0,
            0.0,  15.0,
            -5.0, 20.0;

    // observation model
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> gaussian_d{0,1};

    Eigen::Matrix2f Rsim;
    // Observation model simulation error
    Rsim = Eigen::Matrix2f::Identity();
    Rsim(0,0)=1.0 * 1.0;
    Rsim(1,1)=30.0/180.0 * PI * 30.0/180.0 * PI;



    particle_filter filter( DT);

    do {
        auto x = filter.getState();
        auto z = get_observations(x, Landmarks, gen, gaussian_d);
        filter.update(z, u);

        time += DT;
    } while (time <= SIM_TIME);


    return 0;
}
