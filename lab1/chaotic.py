import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def generateTimeseries():
    x = 1.5
    x_next = []

    for i in range(25):
        x_next.append(x)
        x *= 0.9

    for i in range(1600):
        x_next.append((x_next[-1] + (0.2*x_next[-25]) /
                       (1+x_next[-25]**10) - 0.1 * x_next[-1])+np.random.normal(0, 0.09, 1)[0])

    return x_next


if __name__ == "__main__":

    plot_t = np.linspace(1, 1500, 1500)

    x = generateTimeseries()
    x = np.array(x)

    t_plot = np.linspace(0, 1625, 1625)

    t_vec = np.linspace(301, 1499, 1199)

    plt.figure('Chaotic Time Series')
    plt.plot(t_plot, x)
    plt.axis([0, 1505, 0, 1.5, ])

    print(t_vec.shape)
    x_in = np.array([[x[300-20]], [x[300-15]],
                     [x[300-10]], [x[300-5]], [x[300]]])

    out = np.array([x[300+5]])

    for t in t_vec:
        t = int(t)
        x_temp = np.array([[x[t-20]], [x[t-15]],
                           [x[t-10]], [x[t-5]], [x[t]]])
        x_in = np.hstack((x_in, x_temp))

        out_temp = np.array(x[t+5])
        out = np.hstack((out, out_temp))

    x_train = np.transpose(x_in[:, :1000])
    y_train = out[:1000]
    x_test = np.transpose(x_in[:, 1000:])
    y_test = out[1000:]

    strength = 0.000001
    L1_reg = tf.keras.regularizers.l2(strength)

    # Create the NN
    layersize = 10
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(5, )))
    model.add(tf.keras.layers.Dense(
        layersize, kernel_regularizer=L1_reg, activation='sigmoid'))

    layersize2 = 20
    model.add(tf.keras.layers.Dense(
        layersize2, kernel_regularizer=L1_reg, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(1))
    lrate = 0.01
    optimizer = keras.optimizers.Adam(learning_rate=lrate)
    model.compile(optimizer=optimizer,
                  loss='mse')

    pat = 5
    print("Fit model on training data")
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=pat)
    ]

    history = model.fit(
        x_train,
        y_train,
        shuffle=False,
        validation_split=0.3,
        batch_size=64,
        epochs=2000,
        callbacks=callback)

    weights = model.layers[0].get_weights()[0]
    weights_shape = weights.shape
    weights = np.reshape(weights, (1, weights_shape[0]*weights_shape[1]))

    print(weights.shape)

    print(np.arange(25))
    # weights = np.ndarray.tolist(weights)
    weights = np.squeeze(weights)
    plt.figure("Weights Histogram")
    plt.bar(np.arange(weights.shape[0]), weights)

    plt.savefig("images/weights " + str(layersize))
    plt.title('Connection Weights')
    plt.xlabel('Weights')
    plt.ylabel('Value')

    plt.figure("Learning Curve")
    plt.plot(history.history['loss'], color='k')
    plt.plot(history.history['val_loss'], color='r')
    plt.title('Learning Curves')
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    loss_value = model.evaluate(x_test, y_test)
    prediction = model.predict(x_test)
    plt.figure('Prediction')
    plt.plot(prediction)
    plt.plot(y_test)

    plt.title('Noisy Chaotic Time Series')
    plt.legend(['True values', 'Predicted values'])
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.show()
