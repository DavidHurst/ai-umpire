__all__ = ["KalmanFilter"]

from typing import Tuple

import numpy as np
from numpy.linalg import inv

from ai_umpire.util import multivariate_norm_pdf


class KalmanFilter:
    def __init__(
        self,
        *,
        n_variables: int,
        measurements: np.ndarray,
        mu_p: np.ndarray,
        mu_m: np.ndarray,
        phi: np.ndarray,
        psi: np.ndarray,
        sigma_p: np.ndarray,
        sigma_m: np.ndarray,
    ) -> None:
        self._n_variables: int = n_variables
        self._x: np.ndarray = measurements.copy()
        self.K: np.ndarray  # Kalman gain

        self._t: int = 0  # Current time-step

        # Temporal parameters
        self._mu_p: np.ndarray = mu_p.copy()  # The mean change in the state
        self._psi: np.ndarray = (
            psi.copy()  # Relates the measurement to the state at time step t
        )
        self._sigma_p: np.ndarray = sigma_p.copy()  # Covariance of temporal model

        # Measurement parameters
        self._mu_m: np.ndarray = mu_m.copy()  # The measurement mean?
        self._phi: np.ndarray = (
            phi.copy()  # Relates current state to state at previous time step
        )
        self._sigma_m: np.ndarray = sigma_m.copy()  # Covariance of measurement model

        # Initialise mean and covariance
        self.mu = np.zeros_like(self._mu_p)
        self.cov = np.identity(self.mu.shape[0]) * 500

        # print("Init".center(40, "-"))
        # print("mu_p:\n", self._mu_p)
        # print("mu_m:\n", self._mu_m)
        # print("psi:\n", self._psi)
        # print("phi:\n", self._phi)
        # print("sigma_p:\n", self._sigma_p)
        # print("sigma_m:\n", self._sigma_m)
        #
        # print("mu:\n", self.mu)
        # print("cov:\n", self.cov)

    def get_trajectory(self) -> np.ndarray:
        return self._x

    def prob_of_point(self, point: np.ndarray) -> float:
        return multivariate_norm_pdf(point, self.mu, self.cov)

    def _predict(self) -> None:
        self.mu = self._mu_p + (self._psi @ self.mu)  # State prediction
        self.cov = self._sigma_p + (
            self._psi @ self.cov @ self._psi.T
        )  # Covariance prediction

    def _compute_kalman_gain(self) -> None:
        self.K = (
            self.cov
            @ self._phi.T
            @ np.linalg.inv(self._sigma_m + (self._psi @ self.cov @ self._psi.T))
        )

    def _correct(self) -> None:
        # State update
        self.mu = self.mu + (
            self.K
            @ (
                np.reshape(self._x[self._t], (3, 1))
                - self._mu_m
                - (self._phi @ self.mu)
            )
        )

        # Covariance update
        I = np.identity(self.K.shape[0])
        self.cov = (I - (self.K @ self._phi)) @ self.cov

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._t < self._x.shape[0]:
            self._predict()
            self._compute_kalman_gain()
            self._correct()
            self._t += 1  # Increment time step

            # print(f"Time Step #{self._t}".center(40, "-"))
            # print("mu:\n", self.mu)
            # print("cov:\n", self.cov)
            # print("K:\n", self.K)
        else:
            print("All measurements processed, returning final KF internal state.")

        return self.mu, self.cov

    def get_n_variables(self) -> int:
        return self._n_variables

    def get_t_step(self) -> int:
        return self._t
