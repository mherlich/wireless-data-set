import sys
import os
import time
import datetime
import itertools
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

import utils

# Hyperparameters and other config
random_seed = 42

dataset = 'SRFG-v1.fth'
train_frac = 0.8 # Fraction of dataset used for training (rest is used for test)

epochs = 1
normalize_input = True
hidden_layers = 0
neurons_per_layer = 1
dropout_rate = 0
activations = "relu"
scale_output = 1000000
batch_size = 128

inputs = 'lat, long, ele, rsrq, rsrp, rssi, sinr, signal, pci, dlong, dlat' # These are filtered specifically for this use case, look at the dataframe for other possibilities
input_time_of_day = False
input_days_since_start = False
input_prevDR = 0
input_succDR = 0
input_cell_id_categories = 0
output = 'datarate'
fill_na = "median" # "median" or "mean" or "zero"
mark_na = False

loss_function = 'mean_absolute_error'
optimizer = 'adam'
learning_rate = 0.1

update_freq = "epoch"

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
# Alternative implementation: StringLookup and then Category Encoding, how to include in toolchain?
# Or this https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39
 

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

# Split dataset into train and test
train_data = df.sample(frac=train_frac, random_state=random_seed)
test_data = df.drop(train_data.index)

x_train = train_data[inputsList]
x_test = test_data[inputsList]
y_train = train_data[output]
y_test = test_data[output]

# Make sure log_dir is unique
while (True):
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    if not os.path.exists(log_dir): break
    time.sleep(1)

# Create Model
model = tf.keras.models.Sequential()
normalization_problems = []
if normalize_input:
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer_train = np.array(x_train)
    # Check for normalization problems
    for c in x_train.columns[(np.var(x_train) < 1e-10)]:
        if not c in dont_normalize:
            dont_normalize.append(c)
            normalization_problems.append(c)
            print("Error on normalizing", c, "(variance =", np.var(x_train[c]), "-> did not normalize)")
    print("Normalization-problems:", len(normalization_problems))
    # Prevent normalizer from normalizing some columns
    # TODO: Ugly, but unsure how to do otherwise
    for i in dont_normalize:
        c = inputsList.index(i)
        normalizer_train[::2, c] = 1
        normalizer_train[1::2, c] = -1
    normalizer.adapt(normalizer_train)
    model.add(normalizer)
for i in range(hidden_layers):
    model.add(tf.keras.layers.Dense(units=neurons_per_layer, activation=next(activationsIter)))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(units=1, name="Output"))
if scale_output != 1:
    model.add(tf.keras.layers.Lambda(lambda x: x * scale_output, name="Output_scaler"))
model.compile(optimizer=optimizer, loss=loss_function)
model.optimizer.learning_rate = learning_rate

# Train
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=update_freq)
history = model.fit(x_train, y_train, batch_size=batch_size, 
    epochs=epochs, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
model.save(log_dir+"/model")
hist = pd.DataFrame(history.history)
hist.to_csv(log_dir+"/history.csv")

# Evaluate
p = pd.DataFrame(index=y_test.index)
p["NN"] = model.predict(x_test, batch_size=batch_size)
r = utils.evaluate(p, y_test)
r.to_csv(log_dir+"/evaluation.csv")

# Create plots
p["tv"] = y_test
p["absdiff"] = (p.NN-p.tv).abs()
p["reldiff"] = ((p.NN-p.tv)/p.tv).abs()
p["long"] = test_data["long"]
p.at[p.reldiff > 1, 'reldiff'] = 1
utils.create_plots(p, log_dir+"/plots")

# Add results to tensorflow logs
writer = tf.summary.create_file_writer(log_dir+"/evaluation")
with writer.as_default():
    for c in r.columns:
        if c != 'Strategy':
            if np.isfinite(r[c][0]):
                tf.summary.scalar(c, data=r[c][0], step=epochs-1)
    writer.flush()
