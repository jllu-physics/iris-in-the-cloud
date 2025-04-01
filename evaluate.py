import pandas as pd
import tensorflow as tf
import dataset
import datetime
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Please specify the options for setup:")
    parser.add_argument(
        "--model_version",
        type = str,
        help = 'model version',
        default = datetime.datetime.now().strftime('%Y_%m_%d')
    )
    args = parser.parse_args()
    version = args.model_version
    model_path = os.path.join('.','model_' + version + '_assets',)
    df = pd.read_csv(os.path.join(model_path,'hyperparameter_table.csv'))
    best_model_checkpoint = df.sort_values(
        by = ['valid_acc','train_acc','valid_loss','train_loss'], 
        ascending = [False, False, True, True]
    ).head(1).iloc[0]['checkpoint']
    print("Best model is: " + best_model_checkpoint)

    best_nn = tf.keras.models.load_model(os.path.join(model_path, 'checkpoints', best_model_checkpoint))

    dataset_test = dataset.load_dataset(variant = 'test', data_dir=os.path.join(model_path,'data'))

    _, test_accuracy = best_nn.evaluate(dataset_test, verbose=0, batch_size = 8, steps = 13)
    print('Test accuracy:', test_accuracy)

    best_nn.save(os.path.join(model_path, 'checkpoints', 'best_model.keras'))