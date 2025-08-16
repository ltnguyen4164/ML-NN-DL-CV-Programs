# Long Nguyen
# 1001705873

import numpy as np
import cv2
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.measure import label, regionprops

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt

        self.control = np.zeros((2, 1))   # Control input
        self.initial = np.zeros((4, 1))   # Initial state
        
        # State transition matrix
        self.trans_matrix = np.array([[1, 0, self.dt, 0],
                                      [0, 1, 0, self.dt],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])                  
        # Control Matrix
        self.control_matrix = np.array([[0.5 * self.dt**2, 0],
                                        [0, 0.5 * self.dt**2],
                                        [self.dt, 0],
                                        [0, self.dt]])
        # Measurement Matrix
        self.m_matrix = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]])
        
        # Initial covariance matrix
        self.covar_matrix = np.eye(4)
        # Measurement noise covariance
        self.m_covar_matrix = np.array([[0.1, 0],
                                        [0, 0.1]])
    
    def predict(self):
        # State prediction
        self.x = self.trans_matrix @ self.initial + self.control_matrix @ self.control

        # Covariance prediction
        self.E = self.trans_matrix @ self.covar_matrix @ self.trans_matrix.T

        return int(self.initial[0, 0]), int(self.initial[1, 0])

    def update(self, z):
        # Compute Kalman Gain
        S = self.m_matrix @ self.covar_matrix @ self.m_matrix.T + self.m_covar_matrix
        K = self.covar_matrix @ self.m_matrix.T @ np.linalg.inv(S)

        # Update state estimate
        y = np.array(z).reshape((2, 1)) - self.m_matrix @ self.x
        self.x += K @ y

        # Update covariance matrix
        I = np.eye(self.E.shape[0])
        self.E = (I - K @ self.m_matrix) @ self.covar_matrix

        return int(self.initial[0, 0]), int(self.initial[1, 0])

class MotionTracker:
    def __init__(self, alpha, tau, delta, s, N, KF):
        self.a = alpha              # Frame hysteresis
        self.motion_thresh = tau    # Motion threshold
        self.dist_thresh = delta    # Distance threshold
        self.s = s                  # Number of frames to be skipped
        self.N = N                  # Max number of objects to track
        self.KF = KF                # Kalman Filter object

    def update(self, frames, i):
        s = self.s
        # Ensure enough frames are available for processing
        if i + 2*s >= len(frames):
            return None

        # Convert RGB frames to grayscale
        ppframe = rgb2gray(frames[i])       # prev-prev frame
        pframe = rgb2gray(frames[i + s])    # prev frame
        cframe = rgb2gray(frames[i + 2*s])  # current frame

        # Calculate diff between consecutive frames
        motion_frame = np.minimum(np.abs(cframe - pframe), np.abs(pframe - ppframe))
        thresh_frame = motion_frame > 0.05

        # Create and cache the dilation kernel if not already created
        if not hasattr(self, "kernel"):
            self.kernel = np.ones((9, 9))
        # Dilate the thresholded frame to merge nearby motion regions
        dilated_frame = dilation(thresh_frame, self.kernel)

        # Label connected components in the dilated frame
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)
        centers = []    # Variable to store valid object centers

        from itertools import islice

        # Generator expression: filter regions by area
        valid_regions = (
            region for region in regions
            if 100 <= (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) <= 1000
        )

        # Process up to N valid regions
        for region in islice(valid_regions, self.N):
            minr, minc, maxr, maxc = region.bbox
            # Calculate the center of the bounding box
            center = np.array([(minr + maxr)/2, (minc + maxc)/2])

            x1, y1 = self.KF.predict()
            x2, y2 = self.KF.update([[center[0]], [center[1]]])

            # Check if the detected object is close to the predicted location
            # If so, draw bounding box on the original RGB frame
            if np.linalg.norm([x2 - x1, y2 - y1]) <= self.dist_thresh:
                cv2.rectangle(frames[i], (minc, minr), (maxc, maxr), (0, 0, 255), 2)
                centers.append(region.bbox)