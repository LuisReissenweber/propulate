from typing import Callable, Dict, Generator, List, Union

import numpy as np

from ..population import Individual
from .base import Propagator


class ABC(Propagator):
    """
    This propagator implements approximate bayesian computation.

    See Also
    --------
    :class:`Propagator` : The parent class.
    """

    def __init__(
        self,
        loss_fn: Union[Callable, Generator[float, None, None]],
        limits: Dict,
        perturbation_scale: float = 1.0,
        k: int = 10,
        tol: float = 5.0,
        tolerance_scale: float = 0.99,
    ) -> None:
        """
        Initialize the ABC propagator.

        Parameters
        ----------
        loss_fn : Union[Callable, Generator[float, None, None]]
            The loss function to be minimized.
        limits : Dict
            Search-space limits for each gene.
        perturbation_scale : float
            Scale factor for the Gaussian perturbation covariance.
        k : int
            Number of newest individuals to use as the population.
        tol : float
            Distance tolerance for accepting a new sample (default=5.0).
        rng : Optional[Random]
            Random number generator; if None, a new one is created.
        """
        self.loss_fn = loss_fn
        self.limits = limits
        self.perturbation_scale = perturbation_scale
        self.k = k
        self.tol = tol
        self.tolerance_scale = tolerance_scale
        self.generation = 0

    def weighted_covariance(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Unbiased weighted covariance with bias correction.

        Returns zero matrix if there is only one effective sample (denominator <= 0).

        Parameters
        ----------
        values : np.ndarray
            The values for which to compute the covariance.
        weights : np.ndarray
            The weights for each value.

        Returns
        -------
        np.ndarray
            The weighted covariance matrix.
        """
        values = np.asarray(values)
        weights = np.asarray(weights)
        # Compute weighted mean
        mean = np.average(values, axis=0, weights=weights)
        # Sum of weights
        w = weights.sum()
        # Denominator for bias correction
        denom = w**2 - np.sum(weights**2)
        if denom <= 0:
            # Cannot compute unbiased covariance with <=1 effective samples
            return np.zeros((values.shape[1], values.shape[1]))
        factor = w / denom
        # Compute weighted scatter
        diffs = values - mean
        cov = np.zeros((values.shape[1], values.shape[1]))
        for i in range(len(weights)):
            cov += weights[i] * np.outer(diffs[i], diffs[i])
        return factor * cov

    def __call__(self, inds: List[Individual]) -> Individual:
        """
        Generate a new individual using the ABC algorithm.

        If no parents provided, generate the first individual from the prior.
        Otherwise, take the k newest parents, build a perturbation kernel,
        and sample until the distance is <= tol.

        Parameters
        ----------
        inds : List[propulate.population.Individual]
            The individuals the propagator is applied to.

        Returns
        -------
        propulate.population.Individual
            The individual after application of the propagator.
        """
        # Initial generation: no parents
        if len(inds) == 0:
            while True:
                sample = {}
                for key, limit in self.limits.items():
                    sample[key] = np.random.uniform(limit[0], limit[1])

                child = Individual(position=sample, limits=self.limits, generation=self.generation + 1)
                child.loss = self.loss_fn(child)
                if child.loss <= self.tol:
                    self.generation += 1
                    self.tol = self.tolerance_scale * self.tol
                    return child

        # Select k newest (or all if fewer)
        pop = inds[-self.k :] if len(inds) > self.k else inds
        # Extract raw position arrays
        positions = np.stack([ind.position for ind in pop])
        pop_size, d = positions.shape

        # Uniform weights
        weights = np.ones(pop_size) / pop_size

        # Compute perturbation covariance
        cov = self.weighted_covariance(positions, weights)
        cov += 1e-6 * np.eye(d)
        kernel_cov = self.perturbation_scale * cov

        # Sample until within tolerance
        while True:
            idx = np.random.choice(pop_size, p=weights)
            parent = pop[idx]
            candidate_pos = parent.position + np.random.multivariate_normal(np.zeros(d), kernel_cov)
            child = Individual(position=candidate_pos, limits=self.limits, rank=parent.rank)
            child.generation = self.generation
            loss_c = self.loss_fn(child)
            child.loss = loss_c
            if loss_c <= self.tol:
                self.tol = self.tolerance_scale * self.tol
                self.generation += 1
                return child
