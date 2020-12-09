import numpy as np
import pandas as pd
from scipy import stats
from tqdm.notebook import trange

def mcmc_sampler(data, samples, mean_seed=0.5, gamma_seed=10, mean_gamma=1e4, gamma_var=4e-3, verbose=False):
    def calc_log_like(data, alpha, beta):
        log_like = stats.beta(alpha, beta).logpdf(data).sum()
        return log_like
    
    mean = [mean_seed]
    gamma = [gamma_seed]
    log_like = calc_log_like(data, mean[-1]*gamma[-1], (1-mean[-1])*gamma[-1])

    sample_success = 0
    t = trange(samples) if verbose else range(samples)
    for i in t:
        mean_dist = stats.beta(mean[-1]*mean_gamma, (1-mean[-1])*mean_gamma)
        new_mean = mean_dist.rvs()

        gamma_dist = stats.gamma(gamma[-1]/gamma_var, scale=gamma_var)
        new_gamma = gamma_dist.rvs()

        forward_log_like = mean_dist.logpdf(new_mean) + gamma_dist.logpdf(new_gamma)
        back_log_like = stats.beta(new_mean*mean_gamma, (1-new_mean)*mean_gamma).logpdf(mean[-1]) \
            + stats.gamma(new_gamma/gamma_var, scale=gamma_var).logpdf(gamma[-1])
        move_scaler = back_log_like - forward_log_like

        new_log_like = calc_log_like(data, new_mean*new_gamma, (1-new_mean)*new_gamma)
        like_scaler = new_log_like - log_like

        shift_like = np.exp(like_scaler + move_scaler)
        if shift_like > np.random.uniform():
            mean.append(new_mean)
            gamma.append(new_gamma)
            log_like = new_log_like
            sample_success += 1
        else:
            mean.append(mean[-1])
            gamma.append(gamma[-1])

        success_rate = sample_success / (i+1)
        if verbose:
            t.set_postfix({
                'success': success_rate,
                'mean': mean[-1],
                'gamma': gamma[-1],
            })
            
    coef_samples = pd.DataFrame([mean, gamma], index=['mean', 'gamma']).T
    
    return coef_samples