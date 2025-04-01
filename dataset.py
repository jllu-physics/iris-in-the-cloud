import os
import tensorflow as tf

N_FEATURES = 4
N_CLASSES = 3
SHUFFLE_BUFFER_SIZE = 32

def parse_line(line):
    fields = tf.io.decode_csv(line, record_defaults=[0.0]*(N_FEATURES + N_CLASSES))
    features = tf.stack(fields[:N_FEATURES])
    labels = tf.stack(fields[N_FEATURES:])
    labels = tf.cast(labels, tf.int32)
    return features, labels

def load_dataset(batch_size=8, variant = 'train', data_dir = './data/'):
    dataset_path = os.path.join(data_dir,'data_' + variant + '.csv')
    dataset = tf.data.TextLineDataset(dataset_path).map(parse_line)
    if variant == 'train':
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def parse_features(line):
    fields = tf.io.decode_csv(line, record_defaults=[0.0]*N_FEATURES)
    return tf.stack(fields)

def parse_labels(line):
    labels = tf.io.decode_csv(line, record_defaults=[0]*N_CLASSES)
    return tf.stack(labels)

def load_dataset_backup(batch_size=8, variant = 'train', data_path = './data/'):
    features_path = data_path + 'X_' + variant + '.csv'
    labels_path = data_path + 'y_' + variant + '.csv'
    features_ds = tf.data.TextLineDataset(features_path).map(parse_features)
    labels_ds = tf.data.TextLineDataset(labels_path).map(parse_labels)

    # Combine features and labels into a single dataset
    dataset = tf.data.Dataset.zip((features_ds, labels_ds))
    if variant == 'train':
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
