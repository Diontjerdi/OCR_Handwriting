""" This file is for showing a demo of the image processing and CNN predictions.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import string

from cnn.net import cnn_model_fn
from cnn.eval import get_words
from tools.imgTools import process_char, process_doc


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TensorFlow warning about compilation etc..


img_path = sys.argv[1]
img_type = sys.argv[2]
assert (img_type == "char" or img_type == "doc"), "Image type is either \"char\" or \"doc\"!"


img = cv2.imread(img_path, 0)

classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                    model_dir=r"models\tf_model_1000k")

class_names = list(string.digits) + list(string.ascii_lowercase) + list(string.ascii_uppercase)

if img_type == "char":
    processed_img = process_char(img, False, True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array([processed_img], dtype=np.float32)},
                                                       y=None,
                                                       num_epochs=1,
                                                       shuffle=False)

    predictions = []
    raw_predictions = classifier.predict(input_fn=eval_input_fn)
    for i in raw_predictions:
        predictions.append(i["classes"])

    print("Predicted character: ", class_names[predictions[0]])

    cv2.imshow("Processed image", processed_img)
    cv2.waitKey()


elif img_type == "doc":

    chars, word_lens = process_doc(img)

    images = []
    for char in chars:
        processed_img = process_char(char, skeleton=False, deslant=True)
        images.append(processed_img)

    images = np.asarray(images, dtype=np.float32)

    # Predict
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": images},
                                                       y=None,
                                                       num_epochs=1,
                                                       shuffle=False)

    # Get predicted labels
    predictions = []
    raw_predictions = classifier.predict(input_fn=eval_input_fn)
    for i in raw_predictions:
        predictions.append(i["classes"])

    words = get_words(predictions, word_lens, class_names)
    print("Predicted text: ", ' '.join(words))
