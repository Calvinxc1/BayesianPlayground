def fit_beta(sample):
    gamma_val = ((sample.mean() * (1-sample.mean())) / sample.var()) - 1
    alpha_val = sample.mean() * gamma_val
    beta_val = (1-sample.mean()) * gamma_val
    return {'alpha': alpha_val, 'beta': beta_val}
