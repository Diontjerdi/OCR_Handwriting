import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import string
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from cnn.net import cnn_model_fn
from cnn.dataset import read_data, read_data_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6, rotation=45)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_words(predictions, prediction_lens, lookup):
    """
    Translate predicted classes to words.
    :param predictions: (List) Predicted classes
    :param prediction_lens: (NumPy array) Lengths of words to be predicted
    :param lookup: (List) Lookup table to translate from class number(=index) to alphanumerical character(=value).
    :return: (List) List of predicted words.
    """

    predicted_chars = []
    idx = 0
    for word_len in prediction_lens:
        predicted_chars.append(predictions[idx:idx + word_len])
        idx += word_len

    words = []
    for word in predicted_chars:

        words.append("".join([lookup[char] for char in word]))

    return words


if __name__ == "__main__":
    # Load evaluation data
    # FILENAME = r"sundries\data\HASY_Ben_Char74K\mix_deslant_split0.1\test.txt"  # Data tf_model_1000k is trained on
    # FILENAME = r"sundries\data\NIST_TRAIN\labels_tiny.txt"  # NIST data
    # eval_data, eval_labels = read_data(FILENAME, process=True)

    # Test letter
    IN_DIR = r"sundries\data\test_dion"
    eval_data, eval_labels, eval_lens = read_data_test(IN_DIR)

    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        # model_dir=r"models\tf_model_1000k")
                                        model_dir=r"models\tf_model_cheat1000k")  # With NIST data in training set

    # Evaluate the model and print results (Accuracy)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Get predicted labels
    predictions = []
    raw_predictions = classifier.predict(input_fn=eval_input_fn)
    for i in raw_predictions:
        predictions.append(i["classes"])

    # Classification metrics (Precision, Recall, F1-score)
    class_names = list(string.digits) + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    print(classification_report(eval_labels, predictions, target_names=class_names))

    # Get predicted words --> Only if read_data_test()
    words = get_words(predictions, eval_lens, class_names)
    print(words)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(eval_labels, predictions)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
