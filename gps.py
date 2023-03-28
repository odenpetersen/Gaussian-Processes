#!python3
import jax
from jax import numpy as jnp
from numpy import random as nprand

jax.config.update("jax_enable_x64", True)

def rosenbrock_function(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def rbf_kernel(x, y):
    x = jnp.array(x)
    y = jnp.array(y)
    if len(x) != len(y):
        raise ValueError(f"Incompatible lengths {len(x)} and {len(y)}.")
    return jnp.exp(-jnp.sum(jnp.square(x-y))/2/len(x)**2)

def upper_confidence_bound(mu, sigma, k=1):
    return mu+k*sigma

class GPOptimiser():
    def __init__(self, loss_function, dim=None, kernel=rbf_kernel, acquisition_function=upper_confidence_bound, sampled_points=[], observed_values = []):
        if len(jnp.array(sampled_points)) == 0 and dim is None:
            raise ValueError("Could not deduce dimension of loss function domain. Please provide an argument for either sampled_points or dims.")
        elif dim is None:
            dim = len(sampled_points[0])

        sampled_points = jnp.array(sampled_points).reshape((-1,dim))
        sampled_points = jnp.array([sampled_points] if len(sampled_points.shape) <= 1 else sampled_points)
        observed_values = jnp.array(observed_values)
        if sampled_points.shape[0] != observed_values.shape[0]:
            print(sampled_points,observed_values)
            raise ValueError(f"Length of sampled_points is {len(sampled_points)} while length of observed_values is {len(observed_values)}. These should be the same.")

        self.dim = dim
        self.sampled_points = sampled_points
        self.observed_values = observed_values
        self.in_sample_covs = jnp.array([[self.kernel(p1,p2) for p2 in sampled_points] for p1 in sampled_points])
        self.kernel = kernel
        self.loss_function = loss_function
        self.acquisition_function = acquisition_function

    def add_samples(self, sampled_points, observed_values):
        sampled_points = jnp.array(sampled_points).reshape((-1,self.dim))
        observed_values = jnp.array(observed_values).reshape((-1,))
        if sampled_points.shape[0] != observed_values.shape[0]:
            raise ValueError(f"Length of sampled_points is {sampled_points.shape[0]} while length of observed_values is {observed_values.shape[0]} These should be the same.")

        new_sample_covs = jnp.array([[self.kernel(p1, p2) for p1 in sampled_points] for p2 in sampled_points])

        if self.observed_values.size > 0:
            between_covs = jnp.array([[self.kernel(new_point, old_point) for old_point in self.sampled_points] for new_point in sampled_points])
            self.sampled_points = jnp.concatenate([self.sampled_points, sampled_points])
            self.observed_values = jnp.concatenate([self.observed_values, observed_values])
            self.in_sample_covs = jnp.block([[self.in_sample_covs, between_covs.T], [between_covs, new_sample_covs]])
        else:
            self.in_sample_covs = new_sample_covs
            self.sampled_points = sampled_points
            self.observed_values = observed_values

    def suggest_sample(self, acquisition_function = upper_confidence_bound, lr = 0.01, gradient_steps = 50):
        @jax.jit
        def gp_objective(x):
            between_covs = jnp.array([self.kernel(x,point) for point in self.sampled_points]) 
            exploit_mu = between_covs @ jnp.linalg.solve(self.in_sample_covs, self.observed_values)
            explore_sigma = jnp.sqrt(self.kernel(x,x) - between_covs @ jnp.linalg.solve(self.in_sample_covs, between_covs))
            return self.acquisition_function(exploit_mu, explore_sigma)

        objective_grad = jax.grad(gp_objective)

        x = jnp.zeros(self.dim)

        for _ in range(gradient_steps):
            step = objective_grad(x)
            if not all(jnp.isfinite(step)):
                step = nprand.normal(size=self.dim)
            step /= jnp.linalg.norm(step)
            x += lr * step

        return x

    def optimise(self,n=100,verbose=False):
        for _ in range(n):
            if self.sampled_points.shape[0] == 0:
                x = nprand.normal(size=self.dim)
            else:
                x = self.suggest_sample()
            loss = self.loss_function(x)
            self.add_samples(x, loss)
            if verbose:
                print(loss)
        
        return self.sampled_points[jnp.argmin(self.observed_values)]
