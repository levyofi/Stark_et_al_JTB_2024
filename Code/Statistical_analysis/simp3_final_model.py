import pandas as pd
import numpy as np
import pymc as pm
import pytensor
import pytensor.printing as tp
import pytensor.tensor as tt
import pymc.sampling.jax as pyjax #- need cuda 11.8 and above for that
from statsmodels.stats.stattools import durbin_watson as dwtest
pytensor.config.exception_verbosity = 'high'

import arviz as az
az.rcParams["plot.matplotlib.show"] = True  # bokeh plots are automatically shown by default

# Read the data
data = pd.read_csv("Data/categorical_data_for_statistics_1_min_filtered.csv")

# Standardize the temperature for each microhabitat
tground = data[['Temp_Bush', 'Temp_Open', 'Temp_Rock']].values
tground = (tground - tground.mean(axis=0)) / (2 * tground.std(axis=0))

summer = (data["Season"] == "Summer").astype(int).values

# 1-way interactions
summer_tground = summer[:, None] * tground

# Assuming that 'habitat' is a column in your data with three categories: 'open', 'bush', 'rock'
habitat = pd.Categorical(data['selected_habitat']).codes

# Assuming that 'habitat' is a column in your data with three categories: 'open', 'bush', 'rock'
data['previous_habitat'] = habitat
data['previous_habitat'] = data['previous_habitat'].shift(1)

# Fill the first row's NaN value with a default value (e.g., -1 or any other value that makes sense in your context)
data['previous_habitat'].fillna(-1, inplace=True)

# Convert to integer type
data['previous_habitat'] = data['previous_habitat'].astype(int)
# Create separate columns for each habitat type in 'previous_habitat'
data['previous_habitat_bush'] = (data['previous_habitat'] == 0).astype(int)
data['previous_habitat_rock'] = (data['previous_habitat'] == 2).astype(int)
data['previous_habitat_open'] = (data['previous_habitat'] == 1).astype(int)

# Create new variables that are equal to 'previous_habitat_bush' and 'previous_habitat_rock' but are set to 0 where the ID changes
data['previous_habitat_bush_effect'] = data['previous_habitat_bush']
data['previous_habitat_rock_effect'] = data['previous_habitat_rock']
data['previous_habitat_open_effect'] = data['previous_habitat_open']


# To ignore previous habitat when individuals change, reate a mask where the ID changes
id_change_mask = data['ID'] != data['ID'].shift(1)
data.loc[id_change_mask, 'previous_habitat_bush_effect'] = 0
data.loc[id_change_mask, 'previous_habitat_rock_effect'] = 0
data.loc[id_change_mask, 'previous_habitat_open_effect'] = 0

# Convert to integer type
data['previous_habitat_bush_effect'] = data['previous_habitat_bush_effect'].astype(int)
data['previous_habitat_rock_effect'] = data['previous_habitat_rock_effect'].astype(int)
data['previous_habitat_open_effect'] = data['previous_habitat_open_effect'].astype(int)

# Create a new variable that is equal to 'previous_habitat' but is set to 0 where the ID changes
data['previous_habitat_effect'] = data['previous_habitat']
data.loc[id_change_mask, 'previous_habitat_effect'] = 0

# Convert to integer type
data['previous_habitat_effect'] = data['previous_habitat_effect'].astype(int)

# Assign 'previous_habitat_bush_effect' and 'previous_habitat_rock_effect' to shorter variable names
prev_bush = data['previous_habitat_bush_effect'].values
prev_rock = data['previous_habitat_rock_effect'].values
prev_open = data['previous_habitat_open_effect'].values

prev_bush_tground = prev_bush[:, None]*tground
prev_rock_tground = prev_rock[:, None]*tground
prev_open_tground = prev_open[:, None]*tground

prev_bush_summer = prev_bush*summer
prev_rock_summer = prev_rock*summer
prev_open_summer = prev_open*summer

prev_bush_summer_tground = prev_bush[:, None]*summer[:, None]*tground
prev_rock_summer_tground = prev_rock[:, None]*summer[:, None]*tground
prev_open_summer_tground = prev_open[:, None]*summer[:, None]*tground

X_bush = np.column_stack([np.ones_like(summer), tground[:, 0], summer,
                     summer_tground[:, 0],
                     prev_bush, prev_open, prev_rock, prev_bush_tground[:, 0],
                     prev_bush_summer , prev_bush_summer_tground[:, 0]])


X_open = np.column_stack([np.ones_like(summer), tground[:, 1], summer,
                     summer_tground[:, 1],
                     prev_bush, prev_open, prev_rock, prev_open_tground[:, 1],
                     prev_open_summer, prev_open_summer_tground[:, 1] ])

X_rock = np.column_stack([np.ones_like(summer), tground[:, 2], summer,
                     summer_tground[:, 2],
                     prev_bush, prev_open, prev_rock, prev_rock_tground[:, 2],
                     prev_rock_summer, prev_rock_summer_tground[:, 2] ])

X = np.stack((X_bush, X_open, X_rock), axis=2)

# The rest of the model code remains the same...
# Number of samples
N = X.shape[0]
B = X.shape[1]

ids, _ = data["ID"].factorize(sort=True)
arenas, _ = data["Arena"].factorize(sort=True)

ID = np.array(range(31))
Arena = np.array(range(3))
arena_per_id = data.groupby('ID')['Arena'].first().values

with pm.Model() as model:
    # Fixed effects
    beta = pm.Normal("beta", mu=0, sigma=1, shape=(B, 3))  # Note the shape is now (17, 3) because we have three categories

    # Random effects
    sigmarena = pm.HalfNormal("sigmenc", sigma=1)
    sigmid = pm.HalfNormal("sigmaid", sigma=1)
    arena = pm.Normal("arena", mu=0, sigma=sigmarena, shape=(2,3))
    id = pm.Normal("id", mu=arena[arena_per_id-2, :], sigma=sigmid, shape=(14,3))

    # Expected outcome
    yp_bush = pm.math.dot(X_bush, beta[:,0]) + id[ids, 0]
    yp_open = pm.math.dot(X_open, beta[:, 1]) + id[ids, 1]
    yp_rock = pm.math.dot(X_rock, beta[:, 2]) + id[ids, 2]
    yp = pm.math.stack((yp_bush, yp_open, yp_rock), axis=1)
    p = pm.math.softmax(yp, axis=1)  # Use softmax to ensure probabilities sum to 1

    # The expected value of the outcome
    p_multinomial = pm.Deterministic('p_multinomial', p)

    obs = pm.Categorical("obs", p=p_multinomial, observed=habitat)

    # Sample using nuts
    trace = pyjax.sample_numpyro_nuts(1000, tune=1000, chains=3, idata_kwargs={"log_likelihood": True})

#check model convergence using traceplots
az.plot_trace(trace, var_names=('beta'))
#get model estimates and Rhats for parameters
summary = az.summary(trace, var_names=["beta"])#, "rho"])
print(summary)
summary.to_csv("trace_category_model_final.csv")

summary = az.summary(trace, var_names=["id"])
print(summary)
summary = az.summary(trace, var_names=["sigmenc", "sigmaid", "arena"])
print(summary)

#create posterior predictions for model evaluation
with model:
    pp_samples = pm.sample_posterior_predictive(trace, var_names=["obs", "p_multinomial"], extend_inferencedata=True)

#save model traces az.waic(trace)
trace.to_netcdf("trace_category_models_final.nc")
pp_samples.to_netcdf("trace_category_models_final.nc")

result = az.waic(trace)
print(az.waic(trace))
print(az.loo(trace))
