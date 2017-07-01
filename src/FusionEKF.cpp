#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;
  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
    0, 1, 0, 0;
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;
  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   * Initialize the state ekf_.x_ with the first measurement.
   ****************************************************************************/
    if (!is_initialized_) {
    // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
        
        ekf_.F_ = MatrixXd(4, 4);
        ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
             Initialize states with radar measurements
             */
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float px = rho*cos(phi);
            float py = rho*sin(phi);
            
            ekf_.x_ << px, py, 0, 0;
            ekf_.R_ = R_radar_;
            ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
             Initialize states with laser measurements
             */
            float px = measurement_pack.raw_measurements_[0];
            float py = measurement_pack.raw_measurements_[1];
            ekf_.x_ << px, py, 0, 0;
            ekf_.R_ = R_laser_;
            ekf_.H_ = H_laser_;
        }
        // Set time stamp for first step
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    // Update the state transition matrix F according to the new elapsed time (sec)
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;
    
    // Update the process noise covariance matrix.
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    
    // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    float noise_ax = 9;
    float noise_ay = 9;
    
    // Process noise covariance
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt4 / 4 * noise_ax, 0, dt3 / 2 * noise_ax, 0,
    0, dt4 / 4 * noise_ay, 0, dt3 / 2 * noise_ay,
    dt3 / 2 * noise_ax, 0, dt2 * noise_ax, 0,
    0, dt3 / 2 * noise_ay, 0, dt2 * noise_ay;
    
    // Predict states
    ekf_.Predict();
    previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
      ekf_.R_ = R_radar_;
      ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
      ekf_.R_ = R_laser_;
      ekf_.H_ = H_laser_;
      ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
