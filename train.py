import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# set environ before import tf to suppress warning and info
import json
import argparse
import tensorflow as tf
import dataset
import model
import util
from pydantic import BaseModel
import datetime

N_NEURONS_PER_LAYER = 16
ACTIVATION = 'relu'
OPTIMIZER = 'sgd'

"datetime,n_neurons,activation,optimizer,train_acc,"
"train_loss,valid_acc,valid_loss,checkpoint"

class ModelRecord(BaseModel):
    fit_datetime: datetime.datetime
    n_neurons: int
    activation: str
    optimizer: str
    train_acc: float
    train_loss: float
    valid_acc: float
    valid_loss: float
    checkpoint_filename: str

def add_hyperparameter_record(table_name, record):
    datetime_str = record.fit_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    row = [
        datetime_str,
        record.n_neurons,
        record.activation,
        record.optimizer,
        record.train_acc,
        record.train_loss,
        record.valid_acc,
        record.valid_loss,
        record.checkpoint_filename
    ]
    util.append_to_csv(row, table_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please specify the options for training:")
    parser.add_argument(
        "--n_neurons", 
        type = int, 
        help = "Number of neurons per layer",
        default = N_NEURONS_PER_LAYER
    )
    parser.add_argument(
        "--activation",
        type = str,
        help = 'Activation function for hidden layers',
        default = ACTIVATION
    )
    parser.add_argument(
        "--optimizer",
        type = str,
        help = "Optimizer for fitting",
        default = OPTIMIZER
    )
    args = parser.parse_args()
    n_neurons = args.n_neurons
    activation = args.activation
    optimizer = args.optimizer
    
    with open('feature_statistics.json') as f:
        stats = json.load(f)

    mean = tf.constant(stats['feature_means'], dtype=tf.float32)
    std = tf.constant(stats['feature_stds'], dtype=tf.float32)

    dataset_train = dataset.load_dataset(variant = 'train')
    dataset_valid = dataset.load_dataset(variant = 'valid')
    dataset_test = dataset.load_dataset(variant = 'test')

    nn = model.construct_model(n_neurons, activation, mean, std)
    nn.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    checkpoint_base_name = 'model_' + str(n_neurons) + '_' + activation + '_' + optimizer + '.keras'
    checkpoint_name = util.get_available_filename(checkpoint_base_name, "./checkpoints/")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./checkpoints/" + checkpoint_name, save_best_only = True, monitor='val_accuracy', mode='max')

    nn.fit(dataset_train.repeat(), epochs = 200, validation_data = dataset_valid.repeat(), steps_per_epoch=13, validation_steps = 2,callbacks = [checkpoint_callback,])

    best_nn = tf.keras.models.load_model("./checkpoints/" + checkpoint_name)

    train_loss, train_accuracy = best_nn.evaluate(dataset_train, verbose=0, batch_size = 8, steps = 13)
    print('Train accuracy:', train_accuracy)

    valid_loss, valid_accuracy = best_nn.evaluate(dataset_valid, verbose=0, batch_size = 8, steps = 2)
    print('Validation accuracy:', valid_accuracy)

    record = ModelRecord(
        fit_datetime = datetime.datetime.now(),
        n_neurons = n_neurons,
        activation = activation,
        optimizer = optimizer,
        train_acc = train_accuracy,
        train_loss = train_loss,
        valid_acc = valid_accuracy,
        valid_loss = valid_loss,
        checkpoint_filename = checkpoint_name
    )

    add_hyperparameter_record('hyperparameter_table.csv', record)