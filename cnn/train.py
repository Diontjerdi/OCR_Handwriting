import sys
sys.path.append("../InStep")  # To avoid import errors
import tensorflow as tf

import cnn.net
from cnn.dataset import read_data, read_data_multi, read_data_test
from tools.process_config import process_config


tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
    # Load configurations from file
    # conf_file = r"conf\train.cfg"
    conf_file = sys.argv[1]
    common_params, dataset_params, net_params, training_params = process_config(conf_file)

    # Load training data
    train_data, train_labels = read_data(dataset_params['path'] , process=dataset_params['process'])

    # Create the Estimator
    config = tf.contrib.learn.RunConfig(keep_checkpoint_max=eval(training_params['keep_checkpoint_max']),
                                        save_checkpoints_steps=int(training_params['save_checkpoint_steps']))

    classifier = tf.estimator.Estimator(model_fn=eval(net_params['name']),
                                        model_dir=training_params['train_dir'],
                                        config=config)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=int(common_params['batch_size']),
                                                        num_epochs=None,
                                                        shuffle=True)
    classifier.train(input_fn=train_input_fn,
                     steps=int(training_params['steps']))
