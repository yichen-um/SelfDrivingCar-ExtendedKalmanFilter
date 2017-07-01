#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
    x_ = F_ * x_;
    P_ = F_ * P_ *F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
    * update the state by using Kalman Filter equations
  */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Ht * Si;
    x_ = x_ + (K * y);
    int x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    
    if(fabs(px) < 0.0001) {
        px = 0.0001;
    }
    // convert states into estimated sesnor measurement
    float rho = sqrt(px * px + py * py);
    float phi = atan2(py, px);
    float drho = (px * vx + py * vy)/rho;
    
    if(fabs(rho) < 0.001){
        rho = 0.001;
    }
    VectorXd z_pred = VectorXd(3);
    z_pred << rho, phi, drho;
    
    // Regulate the difference of phi to be -pi to pi
    VectorXd y = z - z_pred;
    float pi = 3.1415926;
    while (y(1) > pi) {
        y(1) -= 2*pi;
    }
    while (y(1) < -pi) {
        y(1) += 2*pi;
    }
    
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Ht * Si;
    x_ = x_ + (K * y);
    int x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
