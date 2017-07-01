#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // Check estimations dimension
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        cout << "Invalid dimension of estimation vector" << endl;
        return rmse;
    }
    
    // Claculate RMSE
    int n = estimations.size();
    for(int i = 0; i < n; ++i) {
        VectorXd res = estimations[i] - ground_truth[i];
        VectorXd resSquare = res.array() * res.array();
        rmse += resSquare;
    }
    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
    MatrixXd Hj(3, 4);
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    // Pre-compute components for Jacobian
    float c1 = px * px + py * py;
    float c2 = sqrt(c1);
    float c3 = (c1 * c2);
    // Avoid dividing by zero
    if(fabs(c1) < 0.0001) {
        cout << "Error, division by zero" << endl;
        return Hj;
    }
    // Compute Jacobian
    Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(vy*px - vx*py)/c3, px/c2, py/c2;
    return Hj;
}
