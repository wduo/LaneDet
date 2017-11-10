import tensorflow as tf
import tensorflow.contrib.slim as slim

import stp3_generate_batches_from_tfrecords
import stp4_ldnet

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_float('learning_rate', 0.01, """Learning rate for train.""")

MAX_STEPS = 5000


def ldnet_train():
    images, labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_train.tfrecords")
    images = tf.cast(images, tf.float32)
    print(images, labels)

    logits = stp4_ldnet.ldnet(inputs=images, num_classes=3, print_current_tensor=False)
    print(logits)

    loss = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(MAX_STEPS):
            _, loss_step = sess.run([train_op, loss])
            if i % 50 == 0:
                print("Step %d: loss on random sampled 64 examples = %.1f" % (i, loss_step))


def main(_):
    ldnet_train()


if __name__ == '__main__':
    tf.app.run()
