from gaussian_process import GP
import jax
from jax import numpy as jnp
from numpy import random as nprand

jax.config.update("jax_enable_x64", True)

def rosenbrock_function(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def lower_confidence_bound(mu, sigma, k=1):
    return mu-k*sigma

class GPOptimiser(GP):
    def __init__(self, loss_function, dim=None, kernel=None, acquisition_function=lower_confidence_bound, sampled_points=[], observed_values = []):
        self.acquisition_function = acquisition_function
        super().__init__(loss_function, dim, kernel, sampled_points, observed_values)

    def suggest_sample(self, lr = 0.01, gradient_steps = 50):
        @jax.jit
        def gp_objective(x):
            between_covs = jnp.array([self.kernel(x,point) for point in self.sampled_points]) 
            exploit_mu = self.mean(x)
            explore_sigma = self.var(x)
            return self.acquisition_function(exploit_mu, explore_sigma)

        objective_and_grad = jax.value_and_grad(gp_objective)

        x = nprand.normal(size=self.dim)

        best_x = x
        best_val = gp_objective(x)

        for _ in range(gradient_steps):
            val,step = objective_and_grad(x)
            if val < best_val:
                best_x = x
                best_val = val
            if not all(jnp.isfinite(step)):
                step = nprand.normal(size=self.dim)
            step /= jnp.linalg.norm(step)
            x -= lr * step

        #return x
        return best_x

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
