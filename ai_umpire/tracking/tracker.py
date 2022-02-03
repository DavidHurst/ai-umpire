__all__ = ["Tracker", "KalmanFilter"]

from typing import Tuple, List

import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(
        self, num_variables: int, init_estimate_uncertainty: float, delta_t: float = 1
    ) -> None:
        self.n_variables: int = num_variables
        self.measurement_shape: Tuple[int, int] = (2, 1)
        self.dt: float = delta_t  # Measurement period
        self.x: np.ndarray = np.zeros((self.n_variables, 1))
        self.P: np.ndarray = np.identity(self.n_variables) * init_estimate_uncertainty
        self.sigma_acc: float = 0.2  # Random acceleration std. dev.
        self.sigma_x: float = 3  # Measurement x error std. dev.
        self.sigma_y: float = self.sigma_x  # Measurement y error std. dev.

        # Observation matrix
        self.H: np.ndarray = np.zeros((self.measurement_shape[0], self.n_variables))
        self.H[0, 0] = 1
        self.H[1, 3] = 1

        # State transition matrix
        self.F: np.ndarray = np.identity(self.n_variables)
        self.F[0, 1] = self.dt
        self.F[0, 2] = self.dt ** 2 * 0.5
        self.F[1, 2] = self.dt
        self.F[3, 4] = self.dt
        self.F[3, 5] = self.dt ** 2 * 0.5
        self.F[4, 5] = self.dt

        # Process noise matrix
        self.Q: np.ndarray = np.zeros_like(self.F)
        self.Q[0, 0] = self.dt ** 4 / 4
        self.Q[0, 1] = self.dt ** 3 / 2
        self.Q[0, 2] = self.dt ** 2 / 2

        self.Q[1, 0] = self.dt ** 3 / 2
        self.Q[1, 1] = self.dt ** 2
        self.Q[1, 2] = self.dt

        self.Q[2, 0] = self.dt ** 2 / 2
        self.Q[2, 1] = self.dt
        self.Q[2, 2] = 1

        self.Q[3:, 3:] = self.Q[0:3, 0:3]

        # Measurement uncertainty
        self.R: np.ndarray = np.zeros((2, 2))
        self.R[0, 0] = self.sigma_x ** 2
        self.R[1, 1] = self.sigma_x ** 2

        # Kalman gain
        self.K: np.ndarray = np.zeros((self.n_variables, self.measurement_shape[0]))

    def _predict(self) -> None:
        """
        1. Extrapolate the state
        2. Extrapolate the uncertainty
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def _correct(self, measurement: np.ndarray) -> None:
        """
        1. Compute the Kalman Gain
        2. Update state estimate with measurement
        3. Update the estimate uncertainty
        """
        self.K = np.dot(
            np.dot(self.P, self.H.T),
            inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R),
        )
        self.x = self.x + np.dot(self.K, measurement - np.dot(self.H, self.x))
        I_less_KH: np.ndarray = np.eye(self.K.shape[0]) - np.dot(self.K, self.H)
        self.P = np.dot(np.dot(I_less_KH, self.P), I_less_KH.T) + np.dot(
            np.dot(self.K, self.R), self.K.T
        )

    def process_measurements(self, measurements: np.ndarray) -> List[Tuple]:
        position_preds: List = []
        self._predict()
        print(f"{'-' * 15} Initialisation {'-' * 15}")
        print("x:")
        print(self.x)
        print("P:")
        print(self.P)
        print("K:")
        print(self.K)
        print(f"{'-' * 15}--------------{'-' * 15}")
        # predictions: List[Tuple] = []
        for i, measurement in enumerate(measurements):
            self._correct(measurement)
            self._predict()

            position_preds.append((self.x[0], self.x[3]))

        print(f"{'-' * 15} Iteration #{len(measurements)} {'-' * 15}")
        print("x:")
        print(np.round_(self.x, 2))
        print("P:")
        print(np.round_(self.P, 2))
        print("K:")
        print(np.round_(self.K, 4))
        print(f"{'-' * 15}--------------{'-' * 15}")

        return position_preds


class Tracker:
    def __init__(self):
        pass
