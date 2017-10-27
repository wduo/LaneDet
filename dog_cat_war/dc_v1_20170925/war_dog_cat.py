from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import reading_data as rdata
import cnn_model
from pylab import *

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  tfrecords_file = "d_c_train.tfrecords"
  rdata.create_tfrecords(tfrecords_file)
  img, label = rdata.read_and_decode(tfrecords_file)
  print(type(img), type(label))

  train_data = np.asarray(img)  # Returns np.array
  train_labels = np.asarray(label)
  print(type(train_data), type(train_labels))
  figure,imshow(train_data),show()
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  d_c_classifier = tf.estimator.Estimator(
      model_fn=cnn_model.cnn_model_fn, model_dir="/tmp/d_c_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=20)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=20,
      num_epochs=None,
      shuffle=True)

  # print(train_input_fn)

  d_c_classifier.train(
      input_fn=train_input_fn,
      steps=100,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
