from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def ldnet(inputs, num_classes=3, dropout_keep_prob=0.5, spatial_squeeze=True, scope="ldnet",
          print_current_tensor=False):
    """
    ldnet architecture:
        input: 32*32*3
                             input   depth   kenal   stride   padding
        conv0：   net = conv(input,   32,   [3, 3],    1,     "same")
                 --> 32*32*32
        conv1:    net = conv(net,    32,   [3, 3],    1,     "same")
                 --> 32*32*32
        conv2：   net = conv(net,    64,   [3, 3],    1,     "same")
                 --> 32*32*64
        maxpool1: net = pool(net,          [3, 3],    1,     "same")
                 --> 32*32*64
        conv3:    net = conv(net,    192,  [3, 3],    1,     "same")
                 --> 32*32*192
        maxpool2: net = pool(net,          [3, 3],    1,     "same")
                 --> 32*32*192

        ldnet blocks:
        mixed_1: 32 x 32 x 320 Feature extraction module
        mixed_2: 16 x 16 x 640 Dimension reduction module
        mixed_3: 16 x 16 x 640 Feature extraction module
        mixed_4: 8 x 8 x 1280 Dimension reduction module
        mixed_5: 4 x 4 x 1280 Dimension reduction module
        Final pooling and prediction -> 3

    :param inputs: the size of imputs is [batch_num, width, height, channel].
    :param num_classes: num of classes.
    :param dropout_keep_prob:
    :param spatial_squeeze:
    :param scope:
    :param print_current_tensor: whether print current tenser shape, name and type.

    :return:
        logits: [batch, num_classes]
    """

    with variable_scope.variable_scope(scope, "ldnet", [inputs]):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                kernel_size=[3, 3],
                stride=1,
                padding='SAME'):
            # input: 32 * 32 * 3

            net = layers.conv2d(inputs, 32, scope="conv0")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 32

            net = layers.conv2d(net, 32, scope="conv1")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 32

            net = layers.conv2d(net, 64, scope="conv2")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 64

            net = layers_lib.max_pool2d(net, kernel_size=[2, 2], scope="maxpool0")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 64

            net = layers.conv2d(net, 192, scope="conv3")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 192

            net = layers_lib.max_pool2d(net, kernel_size=[2, 2], scope="maxpool1")
            if print_current_tensor: print(net)
            # --> 32 * 32 * 192

        # ldnet blocks
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                stride=1,
                padding='SAME'):
            # mixed_1: 32 x 32 x 320 Feature extraction module
            with variable_scope.variable_scope("mixed_1"):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = layers.conv2d(
                        branch_0, 64, [3, 3], scope='Conv2d_0b_3x3')

                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                    branch_1 = layers.conv2d(
                        branch_1, 96, [5, 5], scope='Conv2d_0c_5x5')

                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, 64, [7, 7], scope='Conv2d_0b_7x7')

                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [5, 5], scope='AvgPool_0a_5x5')
                    branch_3 = layers.conv2d(
                        branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')

                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                if print_current_tensor: print(net)

            # mixed_2: 16 x 16 x 640 Dimension reduction module
            with variable_scope.variable_scope("mixed_2"):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net,
                        224, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = layers.conv2d(
                        branch_1,
                        96, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, scope='MaxPool_1a_3x3')

                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
                if print_current_tensor: print(net)

            # mixed_3: 16 x 16 x 640 Feature extraction module
            with variable_scope.variable_scope("mixed_3"):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = layers.conv2d(
                        branch_0, 128, [3, 3], scope='Conv2d_0b_3x3')

                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, 128, [5, 5], scope='Conv2d_0b_5x5')
                    branch_1 = layers.conv2d(
                        branch_1, 192, [5, 5], scope='Conv2d_0c_5x5')

                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers.conv2d(
                        net, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = layers.conv2d(
                        branch_2, 128, [7, 7], scope='Conv2d_0b_7x7')

                with variable_scope.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [5, 5], scope='AvgPool_0a_5x5')
                    branch_3 = layers.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
                if print_current_tensor: print(net)

            # mixed_4: 8 x 8 x 1280 Dimension reduction module
            with variable_scope.variable_scope("mixed_4"):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net,
                        448, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = layers.conv2d(
                        branch_1,
                        192, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, scope='MaxPool_1a_3x3')

                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
                if print_current_tensor: print(net)

            # mixed_5: 4 x 4 x 1280 Dimension reduction module
            with variable_scope.variable_scope("mixed_5"):
                with variable_scope.variable_scope('Branch_0'):
                    branch_0 = layers.conv2d(
                        net,
                        448, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_1'):
                    branch_1 = layers.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = layers.conv2d(
                        branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = layers.conv2d(
                        branch_1,
                        192, [3, 3],
                        stride=2,
                        scope='Conv2d_1a_1x1')

                with variable_scope.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(
                        net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                    branch_2 = layers.conv2d(
                        branch_2, 640, [1, 1], scope='Conv2d_0b_1x1')

                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
                if print_current_tensor: print(net)

            # Final pooling and prediction
            with variable_scope.variable_scope('Logits'):
                net = layers_lib.avg_pool2d(
                    net,
                    [4, 4],
                    padding='VALID',
                    scope='AvgPool_1a_4x4')
                # 1 x 1 x 1280

                # net = layers.conv2d(net, 640, [1, 1], scope='Conv2d_0b_1x1')
                # local1
                with variable_scope.variable_scope('local1') as scope:
                    # Move everything into depth so we can perform a single matrix multiply.
                    reshape = tf.reshape(net, [-1, 1280])
                    weights = _variable_with_weight_decay('weights', shape=[1280, 640],
                                                          stddev=0.04, wd=0.0001)
                    biases = _variable_on_cpu('biases', [640], tf.constant_initializer(0.1))
                    net = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
                # 1 x 1 x 640

                net = layers_lib.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_0c')

                # net = layers.conv2d(net, 320, [1, 1], scope='Conv2d_0d_1x1')
                # local2
                with variable_scope.variable_scope('local2') as scope:
                    weights = _variable_with_weight_decay('weights', shape=[640, 320],
                                                          stddev=0.04, wd=0.0001)
                    biases = _variable_on_cpu('biases', [320], tf.constant_initializer(0.1))
                    net = tf.nn.relu(tf.matmul(net, weights) + biases, name=scope.name)
                # 1 x 1 x 320

                net = layers_lib.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_0e')
                net = tf.expand_dims(net, 1)
                net = tf.expand_dims(net, 1)

                logits = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='Conv2d_0f_1x1')
                # 1 x 1 x 3
                if spatial_squeeze:
                    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    # 3

    return logits
