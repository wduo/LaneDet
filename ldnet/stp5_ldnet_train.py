import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import stp3_generate_batches_from_tfrecords
import stp4_ldnet

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_float('learning_rate', 0.01, """Learning rate for train.""")

MAX_STEPS = 300
MODEL_SAVE_PATH = "/tmp/ldnet/trained_ldnet_model/"
MODEL_NAME = "ldnet_model.ckpt"


def ldnet_train():
    images, labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_train.tfrecords")
    images = tf.cast(images, tf.float32)
    print(images, labels)

    logits = stp4_ldnet.ldnet(inputs=images, num_classes=3, print_current_tensor=False)
    print(logits)

    loss = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(MAX_STEPS):
            _, loss_value, global_step_value = sess.run([train_op, loss, global_step])
            if i % 50 == 0:
                print("Global step %d: loss on random sampled %d examples = %.1f" % (
                    global_step_value, FLAGS.batch_size, loss_value))

            if global_step_value == MAX_STEPS:
                if not os.path.exists(MODEL_SAVE_PATH):
                    os.makedirs(MODEL_SAVE_PATH)
                saver.save(sess, MODEL_SAVE_PATH + "/" + MODEL_NAME, global_step=global_step)


def main(_):
    ldnet_train()


if __name__ == '__main__':
    tf.app.run()
