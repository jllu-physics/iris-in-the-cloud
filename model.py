import tensorflow as tf

def construct_model(n_neurons_per_layer, activation, feature_means, feature_stds):
    normalizer = tf.keras.layers.Normalization(mean=feature_means, variance=feature_stds**2)
    input = tf.keras.layers.Input(shape=(4,))
    normalized = normalizer(input)
    hidden1 = tf.keras.layers.Dense(n_neurons_per_layer, activation = activation)(normalized)
    hidden2 = tf.keras.layers.Dense(n_neurons_per_layer, activation = activation)(hidden1)
    output = tf.keras.layers.Dense(3, activation = 'softmax')(hidden2)
    model = tf.keras.Model(inputs = input, outputs = output)
    return model