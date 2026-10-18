"""MNL-UCB from NeurIPS'21: UCB-based Algorithms for Multinomial Logistic Regression Bandits."""

import numpy as np
from scipy.optimize import NonlinearConstraint

from algorithms.base import AlgorithmRequirements
from utils.numerics import checked_minimize, checked_newton

from .common import CategoricalHistory


class MNLUCB(CategoricalHistory):
    """Estimate multinomial reward probabilities and choose an optimistic action.

    S is the global Frobenius bound; outcomes counts nonbaseline categories.
    Equation (21) is solved locally, so this implementation does not assert
    finding the paper's global projection optimum or its theorem guarantee.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("sigmoid", "softmax"),
        feedback_kinds=("bernoulli", "categorical"),
        stationary_parameters=True,
        categorical_parameters=True,
    )
    DEFAULTS = {"delta": 0.05, "regularization": None}

    def __init__(
        self,
        *,
        dimension,
        horizon,
        parameter_bound,
        action_bound,
        rng,
        outcomes=1,
        reward_vector=None,
        delta=0.05,
        regularization=None,
    ):
        super().__init__(
            dimension=dimension,
            horizon=horizon,
            parameter_bound=parameter_bound,
            action_bound=action_bound,
            rng=rng,
            outcomes=outcomes,
            reward_vector=reward_vector,
            delta=delta,
        )
        self._unit_actions()
        if regularization is None and horizon < 2:
            raise ValueError(
                "MNL-UCB default regularization requires horizon >= 2; supply a positive regularization for horizon=1"
            )
        self.regularization = self._regularization(
            outcomes * dimension * np.log(horizon) if regularization is None else regularization
        )
        self.likelihood_regularization = self.regularization
        # Conservative analytical bounds, Section 3 and Equation (26); no oracle theta.
        log_kappa = self.parameter_bound + 2 * np.logaddexp(
            0.0, np.log(self.outcomes) + self.parameter_bound
        )
        if max(log_kappa, log_kappa + np.log(self.regularization)) > np.log(np.finfo(float).max):
            raise ValueError(
                "parameter_bound gives an MNL-UCB design outside finite numerical range"
            )
        self.kappa = float(np.exp(log_kappa))
        self.L = float(
            1
            / (
                1
                + np.exp(-self.parameter_bound)
                + (self.outcomes - 1) * np.exp(-2 * self.parameter_bound)
            )
        )
        self.R = float(np.linalg.norm(self.reward_vector[1:]))
        self.resolved_parameters = {
            "delta": self.delta,
            "regularization": self.regularization,
            "kappa_bound": self.kappa,
            "link_jacobian_bound": self.L,
            "reward_norm": self.R,
        }
        self.reset()

    def reset(self):
        self.t, self.projection_count = 0, 0
        self.actions, self.outcomes_history = [], []
        self.theta_hat, self.theta_tilde = np.zeros(self.size), np.zeros(self.size)
        self._gram = np.zeros((self.dimension, self.dimension))
        self.V = self.kappa * self.regularization * np.eye(self.dimension)
        self.inv_V = np.eye(self.dimension) / (self.kappa * self.regularization)
        self.H = self.regularization * np.eye(self.size)

    def projection_objective(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        return float(diff @ np.linalg.solve(self.hessian(theta), diff))

    def projection_gradient(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        dual = np.linalg.solve(self.hessian(theta), diff)
        result = 2 * diff
        # d(q^T H^-1 q)=2H^T H^-1 q - dual^T(dH)dual.
        # Each sample's third derivative is obtained from categorical variance.
        matrix = dual.reshape(self.outcomes, self.dimension)
        for x in self.actions:
            p = self.probabilities(theta, x)[1:]
            a = matrix @ x
            mean = p @ a
            dvar = (np.diag(p) - np.outer(p, p)) @ (a * a - 2 * mean * a)
            result -= np.kron(dvar, x)
        return result

    def _projection(self):
        radius = self.parameter_bound
        constraint = NonlinearConstraint(
            lambda u: u @ u,
            -np.inf,
            1.0,
            jac=lambda u: 2 * u,
            hess=lambda u, multipliers: 2 * multipliers[0] * np.eye(self.size),
        )
        result = checked_minimize(
            lambda u: self.projection_objective(radius * u),
            np.zeros(self.size),
            jac=lambda u: radius * self.projection_gradient(radius * u),
            hess="3-point",
            constraints=constraint,
            method="trust-constr",
            options={"gtol": 1e-9, "xtol": 1e-11, "barrier_tol": 1e-11, "maxiter": 500},
            operation="MNL-UCB reward-space projection",
            constraint_violation=lambda u: max(0.0, np.linalg.norm(radius * u) - radius),
        )
        self.projection_count += 1
        return radius * result.x

    def update(self, x, feedback):
        x = self._feature(x)
        outcome = self._outcome(feedback)
        self.actions.append(x.copy())
        self.outcomes_history.append(outcome)
        # Accumulate observations separately: the conservative kappa ridge can
        # exceed 1e16, where adding one feature outer product loses its diagonal.
        self._gram += np.outer(x, x)
        self.V = self.kappa * self.regularization * np.eye(self.dimension) + self._gram
        self.inv_V = np.linalg.solve(self.V, np.eye(self.dimension))
        result = checked_newton(
            self.loss,
            self.theta_hat,
            jac=self.loss_gradient,
            hess=self.hessian,
            operation="MNL-UCB regularized maximum likelihood",
        )
        residual = np.linalg.norm(self.loss_gradient(result.x), ord=np.inf)
        if residual > 1e-6:
            raise RuntimeError(
                f"MNL-UCB maximum likelihood gradient residual {residual:.3g} exceeds 1e-6"
            )
        self.theta_hat = result.x
        self.H = self.hessian(self.theta_hat)
        self.theta_tilde = (
            self.theta_hat.copy()
            if np.linalg.norm(self.theta_hat) <= self.parameter_bound
            else self._projection()
        )
        self.t += 1

    def scores(self, features):
        xs = self._candidates(features)
        # Theorem 1 uses the next round t=n+1, rather than the number of observations.
        root = np.sqrt(self.regularization)
        factor = self.dimension * self.outcomes**1.5 / root
        beta = root * self.parameter_bound + np.sqrt(self.regularization / self.outcomes) / 2
        beta += 2 * factor * np.log(2 / self.delta) + factor * np.log1p(
            (self.t + 1) / (self.dimension * self.regularization)
        )
        predictions = np.array(
            [self.reward_vector @ self.probabilities(self.theta_tilde, x) for x in xs]
        )
        return predictions + 2 * self.R * self.L * np.sqrt(
            self.kappa * (1 + 2 * self.parameter_bound)
        ) * beta * self._norms(xs, self.inv_V)
