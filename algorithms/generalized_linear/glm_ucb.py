"""Regularized GLM-UCB, based on NeurIPS'10: Parametric Bandits: The Generalized Linear Case."""

import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.special import expit

from algorithms.base import AlgorithmRequirements
from utils.numerics import checked_minimize, checked_newton

from .common import CategoricalHistory


class GLMUCB(CategoricalHistory):
    """GLM-UCB with regularized logistic regression and a locally solved projection.

    The projection follows Filippi's reward-space construction, with the source
    ridge coefficient lambda*c_mu. It is solved locally; no global projection
    optimality or theorem guarantee is asserted by this numerical implementation.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("sigmoid",),
        feedback_kinds=("bernoulli",),
        stationary_parameters=True,
        categorical_parameters=True,
    )
    DEFAULTS = {"delta": 0.05, "regularization": 1.0}

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
        regularization=1.0,
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
        self._binary()
        self.regularization = self._regularization(regularization)
        self.c_mu = float(
            expit(parameter_bound * action_bound) * expit(-parameter_bound * action_bound)
        )
        if not self.c_mu > 0:
            raise ValueError(
                "parameter_bound * action_bound exceeds logistic curvature numerical range"
            )
        self.likelihood_regularization = self.regularization * self.c_mu
        self.resolved_parameters = {
            "delta": self.delta,
            "regularization": self.regularization,
            "c_mu": self.c_mu,
            "k_mu": 0.25,
            "bernoulli_subgaussian_scale": 0.5,
        }
        self.reset()

    @property
    def outcomes_history(self):
        return self._observations

    @outcomes_history.setter
    def outcomes_history(self, value):
        self._observations = value

    def reset(self):
        self.t, self.projection_count = 0, 0
        self.actions, self._observations = [], []
        self.theta_hat = np.zeros(self.dimension)
        self.theta_tilde = np.zeros(self.dimension)
        self.V = self.regularization * np.eye(self.dimension)
        self.inv_V = np.eye(self.dimension) / self.regularization
        self.H = self.likelihood_regularization * np.eye(self.dimension)

    def loss(self, theta):
        value = 0.5 * self.likelihood_regularization * (theta @ theta)
        features, outcomes = self._history()
        if len(features):
            logits = features @ theta
            value += np.sum(np.logaddexp(0.0, logits) - outcomes * logits)
        return float(value)

    def gt(self, theta):
        value = self.likelihood_regularization * np.asarray(theta)
        xs, _ = self._history()
        if len(xs):
            value = value + xs.T @ expit(xs @ theta)
        return value

    def loss_gradient(self, theta):
        value = self.gt(theta)
        features, outcomes = self._history()
        if len(features):
            value -= features.T @ outcomes
        return value

    def hessian(self, theta):
        value = self.likelihood_regularization * np.eye(self.dimension)
        xs, _ = self._history()
        if len(xs):
            p = expit(xs @ theta)
            value += xs.T @ ((p * (1 - p))[:, None] * xs)
        return value

    def projection_objective(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        return float(diff @ self.inv_V @ diff)

    def projection_gradient(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        return 2 * self.hessian(theta) @ self.inv_V @ diff

    def projection_hessian(self, theta):
        hessian = self.hessian(theta)
        dual = self.inv_V @ (self.gt(theta) - self.gt(self.theta_hat))
        result = 2 * hessian @ self.inv_V @ hessian
        xs, _ = self._history()
        if len(xs):
            p = expit(xs @ theta)
            third = p * (1 - p) * (1 - 2 * p)
            result += 2 * xs.T @ ((third * (xs @ dual))[:, None] * xs)
        return result

    def _projection(self):
        # theta=S*u is an exact coordinate change. Unit constraints avoid the
        # vanishing constraint Jacobian at small configured parameter radii.
        radius = self.parameter_bound
        constraint = NonlinearConstraint(
            lambda u: u @ u,
            -np.inf,
            1.0,
            jac=lambda u: 2 * u,
            hess=lambda u, multipliers: 2 * multipliers[0] * np.eye(self.dimension),
        )
        result = checked_minimize(
            lambda u: self.projection_objective(radius * u),
            self.theta_tilde / radius,
            jac=lambda u: radius * self.projection_gradient(radius * u),
            hess=lambda u: radius**2 * self.projection_hessian(radius * u),
            constraints=constraint,
            method="trust-constr",
            options={"gtol": 1e-9, "xtol": 1e-11, "barrier_tol": 1e-11, "maxiter": 500},
            operation="GLM-UCB reward-space projection",
            constraint_violation=lambda u: max(0.0, np.linalg.norm(radius * u) - radius),
        )
        self.projection_count += 1
        return radius * result.x

    def update(self, x, feedback):
        x = self._feature(x)
        outcome = self._outcome(feedback)
        self.actions.append(x.copy())
        self._observations.append(outcome)
        self.V += np.outer(x, x)
        self.inv_V = np.linalg.solve(self.V, np.eye(self.dimension))
        result = checked_newton(
            self.loss,
            self.theta_hat,
            jac=self.loss_gradient,
            hess=self.hessian,
            operation="GLM-UCB regularized maximum likelihood",
        )
        residual = np.linalg.norm(self.loss_gradient(result.x), ord=np.inf)
        if residual > 1e-6:
            raise RuntimeError(
                f"GLM-UCB maximum likelihood gradient residual {residual:.3g} exceeds 1e-6"
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
        beta = np.sqrt(self.regularization) * self.c_mu * self.parameter_bound + 0.5 * np.sqrt(
            2 * np.log(1 / self.delta)
            + self.dimension
            * np.log1p(self.t * self.action_bound**2 / (self.regularization * self.dimension))
        )
        return expit(xs @ self.theta_tilde) + (0.5 / self.c_mu) * beta * self._norms(xs, self.inv_V)
