import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import statistics
from sklearn.metrics import r2_score
import sklearn.preprocessing as pre
import math

dataset = 'SRFG-v1.fth'

def evaluate(prediction, true_values):

    c1 = 0
    c2 = 0
    cs1 = [0, 0, 0, 0, 0]
    cs2 = [0, 0, 0, 0, 0]
    for sample in range(len(prediction)):

        equal1 = True
        equal2 = True

        for j in range(5):
            if true_values[sample][j] != round(prediction[sample][j]):
                equal1 = False
            else:
                cs1[j] = cs1[j] + 100/len(prediction)
            if abs(true_values[sample][j] - round(prediction[sample][j])) > 1:
                equal2 = False
            else:
                cs2[j] = cs2[j] + 100/len(prediction)

        if equal1:
            c1 = c1 + 1
        if equal2:
            c2 = c2 + 1

    return c1, c2, cs1, cs2


class Transformer:

    def __init__(self, x_train):

        # standard scaler
        # t = StandardScaler()
        # t.fit(x_train)

        # min-max scaler
        # t = MinMaxScaler()
        # t.fit(x_train)

        # trivial scaler
        self._transformer = pre.StandardScaler()
        self._transformer.fit(x_train)
        self._transformer.mean_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self._transformer.scale_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def normalize(self, data):

        return self._transformer.transform(data)

    def denormalize(self, data):

        return self._transformer.inverse_transform(data)


# Configuration
train_frac = 0.8
epochs = 10
batch_size = 128
neurons_per_layer = [20, 4, 20]
activations = ['relu', 'linear', 'relu']

# Setting the stage
hidden_layers = len(neurons_per_layer)
signal_parameters = ['rsrq', 'rsrp', 'rssi', 'sinr', 'signal']
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

# Load data set, remove samples with NA values, split data set into train and test
df = pd.read_feather(dataset)
df.dropna(subset=signal_parameters, inplace=True)
train_data = df.sample(frac=train_frac, random_state=42)
test_data = df.drop(train_data.index)
x_train = train_data[signal_parameters].to_numpy()
x_test = test_data[signal_parameters].to_numpy()
print('Number of train data: ', len(x_train))
print('Number of test data: ', len(x_test))

# Normalize data
transformer = Transformer(x_train)
x_train = transformer.normalize(x_train)
x_test = transformer.normalize(x_test)

# Create Model
model = tf.keras.models.Sequential()
for i in range(hidden_layers):
    model.add(tf.keras.layers.Dense(units=neurons_per_layer[i], activation=activations[i]))
model.add(tf.keras.layers.Dense(units=len(x_train[0])))
model.compile(optimizer=optimizer, loss=mse)

# Train
print("\nTrain.\n")
losses_train = []
losses_test = []
for epoch in range(epochs):

    print('Epoch ', epoch+1)
    model.fit(x_train, x_train, batch_size=batch_size, epochs=1, verbose=0)
    loss_train = float(mse(x_train, model.predict(x_train)))
    loss_test = float(mse(x_test, model.predict(x_test)))
    # print('Train loss: ', '{:.3f}'.format(loss_train), ', Test loss: ', '{:.3f}'.format(loss_test))
    print('Train loss: ', '{:.8f}'.format(loss_train), ', Test loss: ', '{:.8f}'.format(loss_test))
    losses_train.append(loss_train)
    losses_test.append(loss_test)

    prediction = model.predict(x_test)
    c1, c2, _, _ = evaluate(transformer.denormalize(prediction), transformer.denormalize(x_test))
    p1 = 100 * c1 / len(prediction)
    p2 = 100 * c2 / len(prediction)
    if p1 >= 99 and p2 >= 99:
        print('Sufficient accuracy reached')
        break

# Plot loss functions
plt.plot(losses_train, label='train')
plt.plot(losses_test, label='test')
plt.legend()
plt.show()

# Evaluate
print("\nEvaluation.\n")
prediction = model.predict(x_test)
x_test = transformer.denormalize(x_test)
prediction = transformer.denormalize(prediction)
c1, c2, cs1, cs2 = evaluate(prediction, x_test)
print('Overall: ', len(prediction))
print('C1 - Completely correct predictions (%): ', c1, ' (', '{:.2f}'.format(100 * c1 / len(prediction)), '% )')
print('C2 - Almost correct predictions (%):', c2, ' (', '{:.2f}'.format(100 * c2 / len(prediction)), '% )')
print('Mean squared error (MSE): ', '{:.3f}'.format(mse(prediction, x_test).numpy()))
print('Mean absolute error (MAE): ', '{:.3f}'.format(mae(prediction, x_test).numpy()))
print('R2: ', '{:.3f}'.format(r2_score(x_test, prediction)))
