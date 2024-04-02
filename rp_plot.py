'''
This code is to create recurrence plots of synthetic stochastic data on the stochatic time series
This code is to run in the local machine without using mpi4py
The stochastic data has 30k timestamp , i divide it into 6 parts each with 5000 timestamp
Find the pairwise distance and get a matrix of the state, diff, angle of state and angle of difference
tau = 1 and embedding dimension = 2
I am finding the pairwise values once and getting a matrix ,, which is then cropped to get more images
'''

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def state_rec_plot(df, name=None, tau=1, dim = 2):
    '''This function implemenst the idea of recurrence plot the pdist function takes the L2 norm (Euclidean distance)
    between the states and the sqaureform function craetes a symetric data matrix from the pairwise distances.
    Essentailly plotting the states of the time series in a different dimension.
    Input : 1 Dimensional column vector 
    Output : PNG image
    the patterns and inferences from the image is highly dependent of tau and dims values (important)
    dim : optimal embedding dimension (cao'a algo , false neighbourhood algorithm)
    tau : time lag (first minima of autocorrelation)
    '''
    df = df.reshape(-1,1)
    tuple_vector = [[ df[i+j*tau][0] for j in range(dim)] for i in range(len(df) - dim*tau)]
    states = pdist(tuple_vector)
    m_states =  squareform(states)  

    fig = plt.figure(frameon=False)
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(m_states, aspect='auto', cmap='gray')
    plt.savefig(name+'state.png',dpi=56, bbox_inches='tight')
    plt.close()
    gc.collect()
    return ''

def get_data(train=False):
    '''
    function to get the data from the csv files
    data is returned as a dictionary object with the keys being the 2 claases 0 (healthy) and 1 (mci) which in turn is a dictionary
    the 2 classes have all subjects as thier key and values are a numpy array of the shape (number of rois, 187)
    Input : folder path to the network folder which has 2 folders healthy and mci which in turn has xx subjects .csv folders
    Output : dictionary[ class { 0, 1} ][ subject_name ][ ROI number ]
    '''
    if not train:
        data_dir = f'/Users/ninad/Documents/_CBR/Data/Blackhole_images/Rawdata'
        data = {}
        for j in os.listdir(data_dir):
            if j[:3]=='sac':
                data[j[9:]] = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
        return data
    if train == 'stochastic':
        data_dir = f'/Users/ninad/Documents/_CBR/Data/Blackhole_images/Rawdata'
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
        data_dir = f'/Users/ninad/Documents/_CBR/Data/Blackhole_images/nonsto'
        data = {}
        for j in os.listdir(data_dir):
            full_data = np.array(pd.read_csv(os.path.join(data_dir,j), header=None).values)
            for iter in range(10):
                data[f'{j}{iter}'] = full_data[100*iter:]
        return data

train = False

dim_tau_df = pd.read_csv(f'/Users/ninad/Documents/_CBR/Scripts/RTX data/nonsto_ACF.csv')
timeseries_data = get_data(train)
class_dir = f'/Users/ninad/Documents/_CBR/Data/RTX/Test'
os.makedirs(class_dir, exist_ok=True)

excluded_list = []
for ts in timeseries_data.keys():
    try:
        time_data = timeseries_data[ts]
        print(f'{ts} time series processing started !')
        plt.figure(figsize=(4,4))
        plt.plot(time_data)
        plt.axis('off')
        plt.savefig(f'{class_dir}/{ts}.png',dpi=56)
        plt.close()
        print(f'{ts} time series processing ended !')
    except Exception as e:
        excluded_list.append((ts, 'out of bound'))

# for ts in timeseries_data.keys():
#     subject_data = dim_tau_df[dim_tau_df['Series']== f'sac_ascf_{ts}']
#     try:
#         dim = int(subject_data['DIM'].iloc[0])
#         tau = int(subject_data['Tau'].iloc[0])
#         time_data = timeseries_data[ts]
#         print(f'{ts} time series processing started !')
#         state_rec_plot(time_data[:13000], name=f'{class_dir}/{ts}_ACF_13000', tau=tau, dim=dim)
#         print(f'{ts} time series processing ended !')
#     except Exception as e:
#         excluded_list.append((ts, 'out of bound'))
