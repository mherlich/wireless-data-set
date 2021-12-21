import os
import datetime
import itertools
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
import utils

import pandas as pd
import tensorflow as tf
import numpy as np
import lime
import lime.lime_tabular

#Load dataset
dataset = 'SRFG-v1.fth'

#Prepare data similar to simple.py
train_frac = 0.8
random_seed = 42
activations = "relu"
scale_output = 1000000

#Choose the same inputs as in the model that should be explained (here similar to scenario optimal)
inputs: 'lat, long, ele, rsrq, rsrp, rssi, sinr, signal, pci, dlong, dlat'
input_time_of_day: True
input_days_since_start: True
input_prevDR: 3
input_succDR: 3
input_cell_id_categories: 100
fill_na: "median"    # or "mean"
mark_na: True

# Iteratively reserve GPU memory to allow parallel runs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Prepare complex data structures
if len(inputs) > 0:
    inputsList = [s.strip() for s in inputs.split(',')]
else:
    inputsList = []
activationsIter = itertools.cycle([s.strip() for s in activations.split(',')])

# Set all used random seeds
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Load dataset
df = pd.read_feather(dataset)
dont_normalize = []

# Construct time of day input
if input_time_of_day:
    df["time_of_day"] = (df["time"].dt.hour * 60 + df["time"].dt.minute) * 60 + df["time"].dt.second
    inputsList.append("time_of_day")

# Construct days since start
if input_days_since_start:
    df["days_since_start"] = (df["time"] - df["time"].min())/datetime.timedelta(days=1) // 1
    inputsList.append("days_since_start")

# One hot vectors for cell_id
tids = list(df.groupby("cell_id")["time"].count().sort_values().tail(input_cell_id_categories).index)
for i in tids:
    df["cell_id_"+i] = 1*(df["cell_id"]==i)
    inputsList.append("cell_id_"+i)
    dont_normalize.append("cell_id_"+i)

# Previous data rates
for n in range(1, input_prevDR+1):
    df["prevDR"+str(n)] = df["datarate"].shift(n)
    inputsList.append("prevDR"+str(n))

# Succeeding data rates
for n in range(1, input_succDR+1):
    df["succDR"+str(n)] = df["datarate"].shift(-n)
    inputsList.append("succDR"+str(n))

# Fill missing values
df = utils.fillna(df, inputsList, fill_na, mark_na)
if mark_na:
    for i in inputsList.copy():   
        inputsList.append(i + "_na")
        dont_normalize.append(i + "_na")

# Choose prepared data as data for explaination and convert it in an numpy array
x_exp = df[inputsList]
expinput = x_exp.to_numpy()

# Load model for explanaition
model = tf.keras.models.load_model('model')

# Choose name for parameters in explaination
name = list(x_exp.columns)

# Put inputs in LimeTabularExplainer to create the Limeexplanationmodel
explainer = lime.lime_tabular.LimeTabularExplainer(expinput, feature_names = name, mode = 'regression')

# Set input parameters for for-loop over a chosen number of random samples for the limeexplanation
zeros = np.zeros(len(name))
df = pd.DataFrame({i : zeros}, index = name, columns = [i])

for k in range(1000):  # Number of random samples to be explained
    j = np.random.randint(0, x_exp.shape[0])    # Choose random sample
    exp = explainer.explain_instance(expinput[j], model.predict, num_features = 50)     # Explain influence of random sample
    list = exp.as_list()
    # Isolate the name out of the explained element
    new = pd.DataFrame(list, columns = ['alt', j])
    for l in new['alt']:
        iso = l.split()
        if len(iso) == 5:
            index = iso.pop(2)
        else:
            index = iso.pop(0)
        new['alt'] = new['alt'].replace([l], [index])
    new.set_index('alt', inplace = True)
    df = pd.concat([df,new], axis = 1)

# Estimate the mean over the sum of the absolute values over all explainations
absdf = df.abs()
cols = len(df.axes[1])
absdf['sum'] = absdf.sum(axis=1)
absdf['mean'] = absdf['sum']/cols
absdf = absdf.sort_values(by=['mean'], ascending = False)
absdf = absdf[0:25]     # Choose the first 25 values for visualizing the explaination
absdf = absdf.reset_index()

# Plot the explaination as pgf-file
plt.barh(absdf['index'], absdf['mean'])
plt.grid(axis = 'x')
plt.gcf().set_size_inches(5, 4)
plt.savefig('limeexplanation', bbox_inches = "tight", format = 'pgf')
plt.close()
