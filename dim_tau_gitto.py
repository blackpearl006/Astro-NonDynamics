# !python -m pip install -U giotto-tda
import os
import numpy as np
import pandas as pd
from gtda.time_series import takens_embedding_optimal_parameters
import matplotlib.pyplot as plt

def get_data(train=False, path='/path/to/astronomical-time-series'):
    if not train:
        data_dir = f'{path}/Rawdata'
        data = {}
        for j in os.listdir(data_dir):
            if j[:3]=='sac':
                data[j[9:]] = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
        return data
    if train == 'stochastic':
        data_dir = f'{path}/sto'
        data = {}
        for j in os.listdir(data_dir):
            if j[-6:-4]=='se':
                full_data = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
                for iter in range(6):
                    iter_data = full_data[iter*5000:5000+iter*5000]
                    for subiter in range(10):
                        data[f'{j[:-4]}{iter}{subiter}'] = iter_data[100*subiter:]
        return data
    if train == 'non_stochastic':
        data_dir = f'{path}/nonsto'
        data = {}
        for j in os.listdir(data_dir):
            full_data = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
            for iter in range(10):
                data[f'{j}{iter}'] = full_data[100*iter:]
        return data

phase = 'stochastic'
phase_data = get_data(train=phase)
data_dict = []
for i,ts in phase_data.items():
    optimal_params = takens_embedding_optimal_parameters(ts, max_time_delay=100, max_dimension=30)
    data_dict.append({'Series': i, 'DIM': optimal_params[1], 'Tau': optimal_params[0]})
    print(f'Series: {i}, DIM: {optimal_params[1]}, Tau: {optimal_params[0]}')
df = pd.DataFrame(data_dict)
df.to_csv('astro_optimal_parameters.csv', index=False)