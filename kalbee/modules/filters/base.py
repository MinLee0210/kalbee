from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for state estimation filters.

    This class defines the interface for filters like Kalman Filter,
    Extended Kalman Filter, and alpha-beta-gamma filters.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
        transition_covariance: Optional[np.ndarray] = None,
        measurement_matrix: Optional[np.ndarray] = None,
        measurement_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initialize the filter.

        Args:
            state: Initial state vector (n x 1)
            covariance: Initial state uncertainty matrix (n x n)
            transition_matrix: Matrix F that defines state progression.
            transition_covariance: Process noise covariance matrix Q.
            measurement_matrix: Matrix H that maps state to measurement.
            measurement_covariance: Measurement noise covariance matrix R.
        """
        self.state = np.asanyarray(state).astype(float)
        self.covariance = np.asanyarray(covariance).astype(float)

        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.measurement_matrix = measurement_matrix
        self.measurement_covariance = measurement_covariance

    @property
    def x(self) -> np.ndarray:
        """Current state estimate."""
        return self.state

    @property
    def P(self) -> np.ndarray:
        """Current state covariance."""
        return self.covariance

    @abstractmethod
    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict the next state of the system.

        Args:
            dt: Time step since the last update.
            **kwargs: Additional parameters for specific implementations.

        Returns:
            The predicted state vector.
        """
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update the state estimate with a new measurement.

        Args:
            measurement: The observed measurement vector.
            **kwargs: Additional parameters for specific implementations.

        Returns:
            The updated state vector.
        """
        pass

    def measure(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map a state to the measurement space.

        Args:
            state: The state to map. If None, uses internal state.

        Returns:
            The expected measurement.
        """
        if state is None:
            state = self.state
        if self.measurement_matrix is None:
            raise ValueError("Measurement matrix H is not defined.")
        return self.measurement_matrix @ state

    def _check_input(
        self, state: Optional[np.ndarray], measurement: Optional[np.ndarray]
    ):
        """
        Utility for backward compatibility and input validation.
        """
        if state is None:
            state = self.state
        if measurement is None and self.measurement_matrix is not None:
            measurement = np.random.rand(self.measurement_matrix.shape[0], 1)
        return state, measurement
