import numpy as np
import pandas as pd
from tqdm.notebook import trange

def vuln_sim(dists, sim_years, days_in_year=365, sample_count=1000000, clip=True, verbose=False):
    samples = pd.DataFrame({key:val.rvs(sample_count) for key, val in dists.items()})
    
    variants = {}
    active = []
    vuln_days = []
    t = trange(sim_years * days_in_year) if verbose else range(sim_years * days_in_year)
    for i in t:
        var_occur = np.random.choice(samples['occurence']) / days_in_year
        if var_occur > np.random.uniform():
            var_idx = len(variants)
            variants[var_idx] = {
                'start_day': i,
                'identification': np.random.choice(samples['identification']),
                'remediation': np.random.choice(samples['remediation']),
            }
            variants[var_idx]['duration'] = variants[var_idx]['identification'] + variants[var_idx]['remediation']
            active.append(var_idx)

        efficacy = np.random.choice(samples['variant']) if len(active) > 0 \
            else np.random.choice(samples['efficacy'])

        if efficacy < np.random.uniform():
            vuln_days.append(i)

        for var_idx in [*active]:
            var_end = 1 / variants[var_idx]['duration']
            if var_end > np.random.uniform():
                active.remove(var_idx)
                variants[var_idx]['end_day'] = i
                
    vuln_vals = (pd.Series(vuln_days) // days_in_year).value_counts().reindex(np.arange(1000)).fillna(0) / days_in_year
    if clip: vuln_vals = vuln_vals.clip(1/sim_years, 1 - (1/sim_years))
    
    return vuln_vals, variants