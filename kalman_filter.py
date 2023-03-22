# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
import numpy as np


class KalmanFilter(object):
    """
    This code implements a Kalman filter to estimate the position and velocity of a shuttlecock in 3D space. 
    The dynamics of the shuttlecock are assumed to be such that the x and z velocity of the shuttlecock should be constant,
    while the y velocity is dependent on the acceleration due to gravity. 
    The filter takes as input measurements of the shuttlecock's position,
    and uses these to estimate the current state of the shuttlecock. 
    The filter also uses a model of the dynamics of the shuttlecock to predict the state of the shuttlecock at the next time step, 
    and updates its estimate based on the next measurement.

    Args:
        object (Class): Kalman filter
    """
    def __init__(self, framerate):
        # shuttlecock parameters
        self.accel_gravity = -9.8 # Acceleration due to gravity
        self.shuttle_mass = 0.005 # Average weight of a badminton shuttlecock in kg
        self.dt = 1.0 / framerate
        
        # Initialize state estimation parameters
        self.mu_k = np.zeros((6, 1)) # Initial estimate of the mean of the shuttlecock state
        self.sigma_k = np.eye(6) * 100.0 # Initial estimate of the covariance of the shuttlecock state
        self.Q = np.identity(6) * self.dt # Model noise covariance
        self.R = 0.05 * np.identity(3) # Measurement noise covariance, threshold for ray intersect distance

        # Initialize the matrices for the shuttlecock dynamics and the measurement model
        self.A_k = np.identity(6) + np.eye(6, k=3) * self.dt # Dynamics matrix
        self.Bk_uk = np.array([[0], [0.5 * self.accel_gravity * self.dt * self.dt], [0], [0], [self.accel_gravity * self.dt], [0]]) # Control input matrix
        self.C_k = np.hstack((np.identity(3), np.zeros((3, 3)))) # Measurement matrix
    
    def update(self, measured_pos):
        """
        Takes as input a measurement of the ball position
        Update the estimate of the state based on the input measurement
        
        :param measured_pos (array): Current measured position of the shuttlecock
        
        :returns up_mu_k (array): New state estimate of the shuttlecock
        :returns up_sigma_k (matrix): New Covariance matrix of the shuttlecock state
        """
        # Reshape the measurement vector
        measured_pos = np.reshape(measured_pos, (3,1))
        
        # Compute covariances of the measurements
        cov = self.C_k @ self.sigma_k @ self.C_k.T + self.R
        cov_inv = np.linalg.inv(cov)
        sigma_c = self.sigma_k @ self.C_k.T # Cross covariance btw state and measurement
        
        # Compute measurement error
        meas_err = measured_pos - self.C_k @ self.mu_k
        
        # Update estimates and state covariances
        up_mu_k = self.mu_k + sigma_c @ cov_inv @ meas_err
        up_sigma_k = self.sigma_k - sigma_c @ cov_inv @ (self.C_k @ self.sigma_k)
        
        self.mu_k = up_mu_k
        self.sigma_k = up_sigma_k
        
        # return up_mu_k, up_sigma_k
    
    def predict(self):
        """
        Predict the state of the shuttlecock at the next time step
        based on current state estimates
        
        :returns hit_ground (bool): Flag to check if the shuttlecock has hit the ground or not
        """
        hit_ground = False
        
        # Update state covariance and state estimate
        new_sigma_k = self.A_k @ self.sigma_k @ self.A_k.T + self.Q
        new_mu_k = self.A_k @ self.mu_k + self.Bk_uk
        
        # Check if the position hits the ground, setting a small postive threshold value to account for noise
        if new_mu_k[1] <= 0.05:
            hit_ground = True
        
        self.mu_k = new_mu_k
        self.sigma_k = new_sigma_k
        
        return hit_ground, new_mu_k, new_sigma_k
