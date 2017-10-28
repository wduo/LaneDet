import tensorflow as tf

import stp0_generate_batches_from_tfrecords
import stp2_inception_v3

FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = stp0_generate_batches_from_tfrecords.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

DEFAULT_IMAGE_SIZE = 299
IMAGE_SIZE = [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3]
MAX_STEPS = 5000
NUM_CLASSES = 5

# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 64.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


def train_dog_cat_use_inception_v3():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # image_batch: Images. 4D tensor of [batch_size, height, width, 3] size.
    # labels: label_batch. 1D tensor of [batch_size] size.
    tfrecords_name = "flower_photos_train_%d_%d.tfrecords" % (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    image_batch, labels = stp0_generate_batches_from_tfrecords.generate_batches_from_tfrecords(
        records_name=tfrecords_name, image_size=IMAGE_SIZE)

    # logits: the pre-softmax activations, a tensor of size [batch_size, num_classes]
    # end_points: a dictionary from components of the network to the corresponding activation.
    logits, end_points = stp2_inception_v3.inception_v3(inputs=image_batch, num_classes=NUM_CLASSES)
    final_prediction = end_points['Predictions']

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy_mean)

    # with tf.name_scope('evaluation'):
    #     correct_prediction = tf.equal(tf.argmax(final_prediction, 1), tf.argmax(label, 1))
    #     evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(MAX_STEPS):
            _, cross_entropy_m = sess.run([train_step, cross_entropy_mean])
            if i % 50 == 0:
                print("Step %d: cross_entropy_mean on random sampled 64 examples = %.1f" % (i, cross_entropy_m))


def main(unused):
    train_dog_cat_use_inception_v3()


if __name__ == '__main__':
    tf.app.run()
