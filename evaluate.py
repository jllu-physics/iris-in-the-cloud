import pandas as pd
import tensorflow as tf
import dataset

if __name__ == '__main__':
    df = pd.read_csv('hyperparameter_table.csv')
    best_model_checkpoint = df.sort_values(
        by = ['valid_acc','valid_loss','train_acc','train_loss'], 
        ascending = [False, True, False, True]
    ).head(1).iloc[0]['checkpoint']
    print("Best model is: " + best_model_checkpoint)

    best_nn = tf.keras.models.load_model("./checkpoints/" + best_model_checkpoint)

    dataset_test = dataset.load_dataset(variant = 'test')

    _, test_accuracy = best_nn.evaluate(dataset_test, verbose=0, batch_size = 8, steps = 13)
    print('Test accuracy:', test_accuracy)

    best_nn.save("./checkpoints/best_model.keras")