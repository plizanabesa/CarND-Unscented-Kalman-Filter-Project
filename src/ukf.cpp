#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; //between 2 and 5

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2; //between 0.2 and 0.5

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_= 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //1. Initialize variables
  if (!is_initialized_) {
    // first measurement
    x_ << 0, 0, 0, 0, 0;

    //TODO: Check variables
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      //float v = meas_package.raw_measurements_(2);
      float px = ro * cos(phi);
      float py = ro * sin(phi);
      x_ << px, py, 0, 0, 0;
      
      /*P_ << std_radr_*cos(phi)*std_radr_*cos(phi), 0, 0, 0, 0,
            0, std_radr_*sin(phi)*std_radr_*sin(phi), 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_*std_radphi_, 0,
            0, 0, 0, 0, 0.5;*/
        
      P_ << 2, 0, 0, 0, 0,
            0, 4, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0.5, 0,
            0, 0, 0, 0, 0.5;
        
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
      /*P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 0.5, 0, 0,
            0, 0, 0, 0.5, 0,
            0, 0, 0, 0, 0.5;*/
      
      P_ << 2, 0, 0, 0, 0,
            0, 4, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0.5, 0,
            0, 0, 0, 0, 0.5;
    
    }

    //update previous time
    time_us_ = meas_package.timestamp_;
      
    //done initializing, no need to predict or update
    is_initialized_=true;
      
    //std::cout << "x_ initialization = " << std::endl << x_ << std::endl;
    //std::cout << "P_ initialization = " << std::endl << P_ << std::endl;
    
    //generate Weights
    weights_ << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {
      double weight = 0.5/(n_aug_+lambda_);
      weights_(i) = weight;
    }

    //std::cout << "weights = " << std::endl << weights_ << std::endl;
      
    return;
  }

  //compute the time elapsed between the current and previous measurements
  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; //delta_t - expressed in seconds

  //2. Predict

  //2.1 Generate Sigma Points
  MatrixXd Xsig_pts = MatrixXd(n_x_, 2 * n_x_ + 1);
  UKF::GenerateSigmaPoints(&Xsig_pts);

  //2.2 Augment Sigma Points
  MatrixXd Xsig_pts_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  UKF::AugmentedSigmaPoints(&Xsig_pts_aug);

  //2.3 Predict Sigma Points
  SigmaPointPrediction(Xsig_pts_aug, delta_t);
    
  //2.4 Predict Mean State and Covariance
  UKF::PredictMeanAndCovariance();

  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    //2.4A Predict lidar measurement
  	int n_z = 2;
  	VectorXd z_pred_lidar = VectorXd(n_z);
  	MatrixXd S_lidar = MatrixXd(n_z,n_z);
  	S_lidar.fill(0.0);
    MatrixXd Zsig_lidar = MatrixXd(n_z, 2 * n_aug_ + 1);
    UKF::PredictionLidarMeasurement(&z_pred_lidar, &S_lidar, &Zsig_lidar);

    //3A Update lidar measurement
    UKF::UpdateLidar(meas_package, Zsig_lidar, z_pred_lidar, S_lidar);

    time_us_ = meas_package.timestamp_;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    //2.4B Predict radar measurement
    int n_z = 3;
    VectorXd z_pred_radar = VectorXd(n_z);
    MatrixXd S_radar = MatrixXd(n_z,n_z);
    S_radar.fill(0.0);
    MatrixXd Zsig_radar = MatrixXd(n_z, 2 * n_aug_ + 1);
    UKF::PredictionRadarMeasurement(&z_pred_radar, &S_radar, &Zsig_radar);

    //3B Update radar measurement
    UKF::UpdateRadar(meas_package, Zsig_radar, z_pred_radar, S_radar);

    time_us_ = meas_package.timestamp_;
  }

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();
  //std::cout << "A = " << std::endl << A << std::endl;
    
  //set first column of sigma point matrix
  Xsig.col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }

  //print result
  //std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  //write result
  *Xsig_out = Xsig;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  //std::cout << "P_aug = " << std::endl << P_aug << std::endl;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //std::cout << "L = " << std::endl << L << std::endl;
    
  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd &Xsig_aug, double delta_t) {

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //print result
  //std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance() {
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);
    
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
    
  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred_.col(i);
  }
    
  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }
    
  //print result
  //std::cout << "Predicted state x:" << std::endl << x << std::endl;
  //std::cout << "Predicted covariance matrix P:" << std::endl << P << std::endl;
  
  x_ = x;
  P_ = P;
}

void UKF::PredictionLidarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {
  //set measurement dimension, lidar can measure px, py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // measurement model
    Zsig(0,i) = p_x;                                //p_x
    Zsig(1,i) = p_y;                                //p_y
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S << 0, 0,
       0, 0;
    
  //S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S = S + R;

  //print result
  //std::cout << "Zsig: " << std::endl << Zsig << std::endl;
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::PredictionRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    if (p_x == 0 && p_y == 0) {
      Zsig(0,i) = 0;
      Zsig(1,i) = 0;
      Zsig(2,i) = 0;
    } else {
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S << 0, 0, 0,
       0, 0, 0,
       0, 0, 0;
  //S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  //print result
  //std::cout << "Zsig: " << std::endl << Zsig << std::endl;
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out= Zsig;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package, const MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S) {
  //set measurement dimension, lidar can measure px and py
  int n_z = 2;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  float px = meas_package.raw_measurements_(0);
  float py = meas_package.raw_measurements_(1);
  z << px, py;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //angle normalization
    if (fabs(x_diff(3)) > M_PI){
      //std::cout << "phi before normalization:" << x_diff(3) << std::endl;
      x_diff(3) -= round(x_diff(3) / (2.0 * M_PI)) * (2.0 * M_PI);
      //std::cout << "phi angle normalized:" << x_diff(3) << std::endl;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  //angle normalization
  if (fabs(x_(3)) > M_PI){
    x_(3) -= round(x_(3) / (2.0 * M_PI)) * (2.0 * M_PI);
  }
  P_ = P_ - K*S*K.transpose();

  //print result
  //std::cout << "z lidar measurement: " << std::endl << z << std::endl;
  //std::cout << "Updated state x with lidar: " << std::endl << x_ << std::endl;
  //std::cout << "Updated state covariance P with lidar: " << std::endl << P_ << std::endl;

  // Calculate NIS lidar
  VectorXd z_pred_new(2);
  z_pred_new(0) = x_(0);
  z_pred_new(1) = x_(1);

  VectorXd z_diff_new = z - z_pred_new;
  NIS_laser_ = z_diff_new.transpose()* S.inverse() *z_diff_new;

  //std::cout << "NIS lidar: " << std::endl << NIS_laser_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package, const MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S) {
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  float ro = meas_package.raw_measurements_(0);
  float phi = meas_package.raw_measurements_(1);
  float v = meas_package.raw_measurements_(2);

  z << ro, phi, v;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
      
    //angle normalization
    if (fabs(x_diff(3)) > M_PI){
      //std::cout << "phi before normalization:" << x_diff(3) << std::endl;
      x_diff(3) -= round(x_diff(3) / (2.0 * M_PI)) * (2.0 * M_PI);
      //std::cout << "phi angle normalized:" << x_diff(3) << std::endl;
    }
      
    /*
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;*/

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  //angle normalization
  if (fabs(x_(3)) > M_PI){
    x_(3) -= round(x_(3) / (2.0 * M_PI)) * (2.0 * M_PI);
  }
  P_ = P_ - K*S*K.transpose();

  //print result
  //std::cout << "z radar measurement: " << std::endl << z << std::endl;
  //std::cout << "Updated state x with radar: " << std::endl << x_ << std::endl;
  //std::cout << "Updated state covariance P with radar: " << std::endl << P_ << std::endl;

  // Calculate NIS radar
  double p_x=0, p_y=0, v_new=0, yaw=0, ro_new=0, phi_new=0;
  p_x = x_(0);
  p_y = x_(1);
  yaw = x_(3);
    
  VectorXd z_pred_new(3);
  ro_new = sqrt(p_x*p_x + p_y*p_y);
  z_pred_new(0) = ro_new;
    
  phi_new = 0.001;
  if(fabs(p_x) > 0.001)
    phi_new = atan2(p_y, p_x);

  //angle normalization
  while (phi_new> M_PI) phi_new-=2.*M_PI;
  while (phi_new<-M_PI) phi_new+=2.*M_PI;

  z_pred_new(1) = phi_new;

  v_new = x_(2);
  if (fabs(ro_new) < 0.001) {
    z_pred_new(2) = 0.001;
  } 
  else {
    z_pred_new(2) = (p_x*cos(yaw) + p_y*sin(yaw))*v_new/ro_new;
  }

  VectorXd z_diff_new = z - z_pred_new;
  NIS_radar_ = z_diff_new.transpose()* S.inverse() *z_diff_new;

  //std::cout << "NIS radar: " << std::endl << NIS_radar_ << std::endl;
}
