from datetime import datetime
import os
import tensorflow as tf
# import tensorflow.contrib.slim as slim

import stp3_generate_batches_from_tfrecords
import stp4_ldnet

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
# tf.app.flags.DEFINE_float('learning_rate', 0.01, """Learning rate for train.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 10, """How often to evaluate the training results.""")

# global constants.
images_size = stp3_generate_batches_from_tfrecords.images_size
NUM_EXAMPLES = stp3_generate_batches_from_tfrecords.NUM_EXAMPLES
images_amount_counter = stp3_generate_batches_from_tfrecords.images_amount_counter

# constants describing the current file.
MAX_STEPS = 5000
NUM_CLASS = 3
MODEL_SAVE_PATH = "/tmp/ldnet/saved_model"
MODEL_NAME = "ldnet_model.ckpt"
SUMMARIES_PATH = "/tmp/ldnet/summaries"
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES * images_amount_counter['train']
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.
NUM_EPOCHS_PER_DECAY = 32  # Epochs after which learning rate decays.


def add_training_ops(num_class, global_step):
    with tf.name_scope('input'):
        # the size of images_placeholder: [batch_size, image_width, image_height, image_channel]
        images_placeholder = tf.placeholder(tf.float32, [None, images_size[0], images_size[1], images_size[2]],
                                            name='ImagesPlaceholder')
        # the size of labels_placeholder: [batch_size]
        labels_placeholder = tf.placeholder(tf.int32, [None], name='LabelsPlaceholder')

    # the size of logits: [batch_size, 3]
    logits = stp4_ldnet.ldnet(inputs=images_placeholder, num_classes=num_class, print_current_tensor=False)
    tf.summary.histogram('pre_activations', logits)

    # the size of final_tensor: [batch_size, 3]
    final_tensor = tf.nn.softmax(logits, name="Prediction")
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=tf.one_hot(labels_placeholder, num_class))
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            # tf.summary.scalar('cross_entropy', cross_entropy_mean)
            tf.add_to_collection('losses', cross_entropy_mean)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('loss_averages'):
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.name_scope('train'):
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        with tf.control_dependencies([loss_averages_op]):
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_step = optimizer.minimize(total_loss, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return images_placeholder, labels_placeholder, final_tensor, total_loss, train_op


def add_evaluation_step(final_tensor, labels_placeholder):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # the size of prediction: [batch_size]
            prediction = tf.argmax(final_tensor, 1)
            # the size of correct_prediction: [batch_size], elements type is bool.
            correct_prediction = tf.equal(prediction, tf.cast(labels_placeholder, tf.int64))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', evaluation_step)

    return evaluation_step, prediction


def ldnet_train():
    # get tensor for train and validation.
    train_images, train_labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_train.tfrecords", train_or_validation="train")
    validation_images, validation_labels = stp3_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name="ldnet_validation.tfrecords", train_or_validation="validation")
    # print(train_images, train_labels)
    # print(validation_images, validation_labels)

    # global step for train and validation.
    global_step = tf.Variable(0, trainable=False)

    # add train step and validation step.
    (images_placeholder, labels_placeholder, final_tensor,
     cross_entropy_mean, train_step) = add_training_ops(num_class=NUM_CLASS, global_step=global_step)
    evaluation_step, prediction = add_evaluation_step(final_tensor, labels_placeholder)

    # merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()

    # saver for saving checkpoints.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # write summaries out to the summaries_path
        train_writer = tf.summary.FileWriter(SUMMARIES_PATH + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(SUMMARIES_PATH + '/validation')

        init = tf.global_variables_initializer()
        sess.run(init)

        # start threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # get feed tensor values.
        train_images, train_labels = sess.run([train_images, train_labels])
        validation_images, validation_labels = sess.run([validation_images, validation_labels])

        try:
            for i in range(MAX_STEPS):
                # training
                _, global_step_value, train_summary = sess.run([train_step, global_step, merged],
                                                               feed_dict={images_placeholder: train_images,
                                                                          labels_placeholder: train_labels})
                train_writer.add_summary(train_summary, global_step_value)

                if (i % FLAGS.eval_step_interval) == 0 or i + 1 == MAX_STEPS:
                    # print loss and train accuracy.
                    loss_value, train_accuracy = sess.run([cross_entropy_mean, evaluation_step],
                                                          feed_dict={images_placeholder: train_images,
                                                                     labels_placeholder: train_labels})
                    print("[%s] Step %d: loss = %.1f, train accuracy = %.1f%% (%d examples)" % (
                        datetime.now(), global_step_value, loss_value, train_accuracy * 100, FLAGS.batch_size))

                    # print validation accuracy.
                    validation_accuracy, validation_summary = sess.run([evaluation_step, merged],
                                                                       feed_dict={images_placeholder: validation_images,
                                                                                  labels_placeholder: validation_labels})
                    validation_writer.add_summary(validation_summary, global_step_value)
                    print("[%s]          validation accuracy = %.1f%%" % (datetime.now(), validation_accuracy * 100))

            # final test evaluation on some new images we haven't used before.
            # final_test()

            # save checkpoints.
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, MODEL_SAVE_PATH + "/" + MODEL_NAME, global_step=global_step)

        except Exception as e:
            coord.request_stop(e)

        finally:
            coord.request_stop()
            coord.join(threads)


def main(_):
    ldnet_train()


if __name__ == '__main__':
    tf.app.run()
