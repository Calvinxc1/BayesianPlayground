def fit_gamma(sample):
    shape_val = (sample.mean()**2) / sample.var()
    scale_val = sample.var() / sample.mean()
    return {'shape': shape_val, 'scale': scale_val}
