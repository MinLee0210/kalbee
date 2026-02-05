import numpy as np

from kalbee.modules.filters.base import BaseFilter


class KalmanFilter(BaseFilter):
    """
    Standard Linear Kalman Filter implementation.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ):
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict the next state:
        x = Fx
        P = FPF' + Q
        """
        F = self.transition_matrix
        Q = self.transition_covariance

        # In a real KF, F might depend on dt.
        # For simplicity, we assume F is already set or passed in kwargs.
        if "F" in kwargs:
            F = kwargs["F"]

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update the state with a measurement:
        y = z - Hx (residual)
        S = HPH' + R (innovation covariance)
        K = P H' S^-1 (Kalman gain)
        x = x + Ky
        P = (I - KH)P
        """
        z = measurement
        H = self.measurement_matrix
        R = self.measurement_covariance
        P = self.covariance
        x = self.state

        # Innovation
        y = z - H @ x
        # Innovation covariance
        S = H @ P @ H.T + R
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = x + K @ y
        identity = np.eye(P.shape[0])
        self.covariance = (identity - K @ H) @ P

        return self.state
