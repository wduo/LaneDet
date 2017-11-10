import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import stp3_generate_batches_from_tfrecords
import stp4_ldnet

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_float('learning_rate', 0.01, """Learning rate for train.""")

images_size = stp3_generate_batches_from_tfrecords.images_size

MAX_STEPS = 300
NUM_CLASS = 3
MODEL_SAVE_PATH = "/tmp/ldnet/trained_ldnet_model/"
MODEL_NAME = "ldnet_model.ckpt"


def add_training_ops(num_class, global_step):
    with tf.name_scope('input'):
        images_placeholder = tf.placeholder(tf.float32, [None, images_size[0], images_size[1], images_size[2]],
                                            name='ImagesPlaceholder')
        labels_placeholder = tf.placeholder(tf.float32, [None, num_class], name='LabelsPlaceholder')

    logits = stp4_ldnet.ldnet(inputs=images_placeholder, num_classes=NUM_CLASS, print_current_tensor=False)

    final_tensor = tf.nn.softmax(logits, name="Prediction")

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean, global_step=global_step)

    return images_placeholder, labels_placeholder, final_tensor, cross_entropy_mean, train_step


def add_evaluation_step(final_tensor, labels_placeholder):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(final_tensor, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(labels_placeholder, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return evaluation_step, prediction


def ldnet_train():
    train_images, train_labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_train.tfrecords", train_or_validation="train")
    train_images = tf.cast(train_images, tf.float32)
    print(train_images, train_labels)

    validation_images, validation_labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_validation.tfrecords", train_or_validation="validation")
    validation_images = tf.cast(validation_images, tf.float32)
    print(validation_images, validation_labels)

    global_step = tf.Variable(0, trainable=False)

    (images_placeholder, labels_placeholder, final_tensor,
     cross_entropy_mean, train_step) = add_training_ops(num_class=NUM_CLASS, global_step=global_step)

    evaluation_step, prediction = add_evaluation_step(final_tensor, labels_placeholder)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(MAX_STEPS):
            loss_value, _, global_step_value = sess.run([cross_entropy_mean, train_step, global_step],
                                                        feed_dict={images_placeholder: train_images,
                                                                   labels_placeholder: train_labels})
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
