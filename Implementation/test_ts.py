import tensorflow as tf
import numpy as np

X = np.arange(10)
y = np.arange(10) * 10
seq_length = 3

ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X,
    targets=y[seq_length:],
    sequence_length=seq_length,
    batch_size=2
)

for b_x, b_y in ds:
    for i in range(len(b_x)):
        print("X:", b_x[i].numpy(), "y:", b_y[i].numpy())
