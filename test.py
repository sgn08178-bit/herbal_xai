import pandas as pd
df = pd.read_csv('/data/JSM/convnext/2_ig_cache/convergence/steps_convergence.csv')
print(df[['steps', 'delta_f', 'ig_sum', 'rel_error']].head(15))