from gaussian_process import GP

def lower_confidence_bound(mu, sigma, k=1):
    return mu-k*sigma

class GPOptimiser(GP):
    def suggest_sample(self, lr = 0.01, gradient_steps = 50):
        @jax.jit
        def gp_objective(x):
            between_covs = jnp.array([self.kernel(x,point) for point in self.sampled_points]) 
            exploit_mu = between_covs @ jnp.linalg.solve(self.in_sample_covs, self.observed_values)
            explore_sigma = jnp.sqrt(self.kernel(x,x) - between_covs @ jnp.linalg.solve(self.in_sample_covs, between_covs))
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
