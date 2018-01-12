from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

import tensorflow as tf

from deform_conv.layers import ConvOffset2D


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


def _feature_extraction_residual(net, first_layer_depth=48, second_layer_depth=64, last_layer_depth=96,
                                 use_deform_conv=False, scope='feature_extraction_residual'):
    """
    Feature extraction module of ldnet-v1.
    :param net: the net input.
    :param first_layer_depth: first layer depth.
    :param second_layer_depth: second layer depth.
    :param last_layer_depth: last layer depth.
    :param scope: optional scope.
    :return:
        the size of returned net: [batch_size, height, width, channel], which
        channel = (second_layer_depth + last_layer_depth) * 2
    """
    with variable_scope.variable_scope(scope):
        with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(
                net, first_layer_depth, [1, 1], scope='Conv2d_0a_1x1')
            if use_deform_conv:
                branch_0 = ConvOffset2D(first_layer_depth, name='conv3_offset')(branch_0)  # net offset
            branch_0 = layers.conv2d(
                branch_0, second_layer_depth, [3, 3], scope='Conv2d_0b_3x3')

        with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(
                net, first_layer_depth, [1, 1], scope='Conv2d_0a_1x1')
            if use_deform_conv:
                branch_1 = ConvOffset2D(first_layer_depth, name='conv3_offset')(branch_1)  # net offset
            branch_1 = layers.conv2d(
                branch_1, second_layer_depth, [5, 5], scope='Conv2d_0b_5x5')
            if use_deform_conv:
                branch_1 = ConvOffset2D(second_layer_depth, name='conv3_offset')(branch_1)  # net offset
            branch_1 = layers.conv2d(
                branch_1, last_layer_depth, [5, 5], scope='Conv2d_0c_5x5')

        with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(
                net, first_layer_depth, [1, 1], scope='Conv2d_0a_1x1')
            if use_deform_conv:
                branch_2 = ConvOffset2D(first_layer_depth, name='conv3_offset')(branch_2)  # net offset
            branch_2 = layers.conv2d(
                branch_2, second_layer_depth, [7, 7], scope='Conv2d_0b_7x7')

        with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.avg_pool2d(net, [5, 5], scope='AvgPool_0a_5x5')
            branch_3 = layers.conv2d(
                branch_3, last_layer_depth, [1, 1], scope='Conv2d_0b_1x1')

        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    return net


def _dimension_reduction(net, branch_0_depth=224, branch_1_depth=96, use_deform_conv=False,
                         scope='dimension_reduction'):
    """
    Dimension reduction module of ldnet-v1.
    :param net: the net input.
    :param branch_0_depth: the depth of branch_0.
    :param branch_1_depth: the depth of branch_1.
    :param scope: optional scope.
    :return:
        the size of returned net: [batch_size, height, width, channel], which
        channel = (branch_0_depth + branch_1_depth) + last_net_depth
    """
    with variable_scope.variable_scope(scope):
        with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(
                net,
                branch_0_depth, [3, 3],
                stride=2,
                scope='Conv2d_1a_1x1')

        with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(
                net, 64, [1, 1], scope='Conv2d_0a_1x1')
            if use_deform_conv:
                branch_1 = ConvOffset2D(64, name='conv3_offset')(branch_1)  # net offset
            branch_1 = layers.conv2d(
                branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
            if use_deform_conv:
                branch_1 = ConvOffset2D(96, name='conv3_offset')(branch_1)  # net offset
            branch_1 = layers.conv2d(
                branch_1,
                branch_1_depth, [3, 3],
                stride=2,
                scope='Conv2d_1c_1x1')

        with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers_lib.max_pool2d(
                net, [3, 3], stride=2, scope='MaxPool_1a_3x3')

        net = array_ops.concat([branch_0, branch_1, branch_2], 3)

    return net


def _prev_net_linear_projection(net_shape, prev_net, scope='linear_projection'):
    """
    rf. Deep Residual Learning for Image Recognition

    The identity shortcuts (Eqn.(1)) can be directly used when the input and
    output are of the same dimensions(channels) (solid line shortcuts in Fig. 3).

    When the dimensions increase (dotted line shortcuts in Fig. 3), we consider:
    The projection shortcut in Eqn.(2) is used to match dimensions (done by 1x1 convolutions).

    When the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

    :param net_shape: the residual's shape [batch_size, height, width, channel].
    :param prev_net: the shortcut connection.
    :param scope: optional scope.
    :return:
        prev_net_projection: if the shortcut's shape equal to the residual's shape, return shortcut directly.
    """
    if net_shape[-3:] != prev_net.shape[-3:]:
        print(net_shape[-3:], prev_net.shape[-3:], (net_shape[-3:] == prev_net.shape[-3:]))
        out_stride = int(prev_net.shape[1]) // int(net_shape[1])
        if out_stride == 1:
            prev_net = layers.conv2d(prev_net, int(net_shape[3]), [1, 1], stride=out_stride,
                                     activation_fn=None, scope=scope)
        else:
            prev_net = layers.conv2d(prev_net, int(net_shape[3]), [3, 3], stride=out_stride,
                                     activation_fn=None, scope=scope)

    return prev_net


def _shortcuts_addition(net_shape, prev_net, prev_prev_net, scope="shortcuts_addition"):
    with variable_scope.variable_scope(scope):
        prev_net = _prev_net_linear_projection(net_shape, prev_net, scope='prev_net_linear_projection')
        prev_prev_net = _prev_net_linear_projection(net_shape, prev_prev_net,
                                                    scope='prev_prev_net_linear_projection')

    # plain residual.
    # return prev_net

    # enhanced residual.
    return prev_net + prev_prev_net


def ldnet_v1(inputs, num_classes=3, dropout_keep_prob=0.5, spatial_squeeze=True, scope="ldnet",
             use_deform_conv=True, print_current_tensor=False):
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
        mixed_2: 32 x 32 x 320 Feature extraction module
        mixed_res1: 32 x 32 x 320 Feature extraction module
        mixed_3: 16 x 16 x 640 Dimension reduction module
        mixed_4: 16 x 16 x 640 Feature extraction module
        mixed_res2: 16 x 16 x 640 Feature extraction module
        mixed_5: 8 x 8 x 1280 Dimension reduction module
        mixed_6: 8 x 8 x 1280 Feature extraction module
        mixed_res3: 8 x 8 x 1280 Feature extraction module
        Final pooling and prediction -> 3

    :param inputs: the size of imputs is [batch_num, width, height, channel].
    :param num_classes: num of classes predicted.
    :param dropout_keep_prob: dropout probability.
    :param spatial_squeeze: whether or not squeeze.
    :param scope: optional scope.
    :param use_deform_conv: whether to use deform conv.
    :param print_current_tensor: whether or not print current tenser shape, name and type.

    :return:
        logits: [batch_size, num_classes]
    """

    # end_points will collect relevant activations for the computation
    # of shortcuts.
    end_points = []

    with variable_scope.variable_scope(scope, "ldnet_v1", [inputs]):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d],
                kernel_size=[3, 3],
                stride=1,
                padding='SAME'):
            # input: 32 * 32 * 3
            net = inputs

            end_point = "conv0"
            # if use_deform_conv:
            #     net = ConvOffset2D(3, name='conv0_offset')(net)  # net offset
            net = layers.conv2d(net, 32, scope=end_point)
            if print_current_tensor:
                print(net)
            # --> 32 * 32 * 32

            end_point = "conv1"
            if use_deform_conv:
                net = ConvOffset2D(32, name='conv1_offset')(net)  # net offset
            net = layers.conv2d(net, 32, scope=end_point)
            if print_current_tensor: print(net)
            # --> 32 * 32 * 32

            end_point = "conv2"
            if use_deform_conv:
                net = ConvOffset2D(32, name='conv2_offset')(net)  # net offset
            net = layers.conv2d(net, 64, scope=end_point)
            if print_current_tensor: print(net)
            # --> 32 * 32 * 64

            end_point = "maxpool0"
            net = layers_lib.max_pool2d(net, kernel_size=[2, 2], scope=end_point)
            if print_current_tensor: print(net)
            # --> 32 * 32 * 64

            end_point = 'conv3'
            if use_deform_conv:
                net = ConvOffset2D(64, name='conv3_offset')(net)  # net offset
            net = layers.conv2d(net, 192, scope=end_point)
            if print_current_tensor: print(net)
            # end_points.append(net)
            # --> 32 * 32 * 192

            end_point = 'maxpool1'
            net = layers_lib.max_pool2d(net, kernel_size=[2, 2], scope=end_point)
            if print_current_tensor: print(net)
            # net.alias = end_point
            # end_points.append(net)
            # --> 32 * 32 * 192

        # ldnet blocks
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
                stride=1,
                padding='SAME'):
            # mixed_1: 32 x 32 x 320 Feature extraction module
            end_point = 'mixed_1'
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48, second_layer_depth=64,
                                                   last_layer_depth=96, scope='feature_extraction')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_2: 32 x 32 x 320 Feature extraction module
            end_point = 'mixed_2'
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48, second_layer_depth=64,
                                                   last_layer_depth=96, scope='feature_extraction_residual')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_res1: 32 x 32 x 320 Feature extraction module
            end_point = 'mixed_res1'
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48, second_layer_depth=64,
                                                   last_layer_depth=96, scope='feature_extraction_residual')
                net_linear = layers.conv2d(net, int(net.shape[3]), [1, 1], activation_fn=None,
                                           scope='net_linear_projection')
                shortcuts = _shortcuts_addition(net.shape, end_points[-1], end_points[-2],
                                                scope="shortcuts_addition")
                net = nn_ops.relu(net_linear + shortcuts)
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_3: 16 x 16 x 640 Dimension reduction module
            end_point = "mixed_3"
            with variable_scope.variable_scope(end_point):
                net = _dimension_reduction(net, branch_0_depth=224, branch_1_depth=96,
                                           scope='dimension_reduction')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_4: 16 x 16 x 640 Feature extraction module
            end_point = "mixed_4"
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48 * 2, second_layer_depth=64 * 2,
                                                   last_layer_depth=96 * 2, scope='feature_extraction_residual')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_res2: 16 x 16 x 640 Feature extraction module
            end_point = "mixed_res2"
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48 * 2, second_layer_depth=64 * 2,
                                                   last_layer_depth=96 * 2, scope='feature_extraction_residual')
                net_linear = layers.conv2d(net, int(net.shape[3]), [1, 1], activation_fn=None,
                                           scope='net_linear_projection')
                shortcuts = _shortcuts_addition(net.shape, end_points[-1], end_points[-2],
                                                scope="shortcuts_addition")
                net = nn_ops.relu(net_linear + shortcuts)
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_5: 8 x 8 x 1280 Dimension reduction module
            end_point = "mixed_5"
            with variable_scope.variable_scope(end_point):
                net = _dimension_reduction(net, branch_0_depth=224 * 2, branch_1_depth=96 * 2,
                                           scope='dimension_reduction')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_6: 8 x 8 x 1280 Feature extraction module
            end_point = "mixed_6"
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48 * 4, second_layer_depth=64 * 4,
                                                   last_layer_depth=96 * 4, scope='feature_extraction_residual')
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

            # mixed_res3: 8 x 8 x 1280 Feature extraction module
            end_point = "mixed_res3"
            with variable_scope.variable_scope(end_point):
                net = _feature_extraction_residual(net, first_layer_depth=48 * 4, second_layer_depth=64 * 4,
                                                   last_layer_depth=96 * 4, scope='feature_extraction_residual')
                net_linear = layers.conv2d(net, int(net.shape[3]), [1, 1], activation_fn=None,
                                           scope='net_linear_projection')
                shortcuts = _shortcuts_addition(net.shape, end_points[-1], end_points[-2],
                                                scope="shortcuts_addition")
                net = nn_ops.relu(net_linear + shortcuts)
                end_points.append(net)
                if print_current_tensor: print(net, len(end_points))

        # Final pooling and prediction
        with variable_scope.variable_scope('Logits'):
            with arg_scope([layers.conv2d], normalizer_fn=None, normalizer_params=None):
                net = layers.conv2d(net, int(net.shape[3]), [3, 3], stride=2, scope='conv2d_1a_3x3')
                # 4 x 4 x 1280
                net = layers_lib.avg_pool2d(
                    net,
                    [4, 4],
                    padding='VALID',
                    scope='AvgPool_1b_4x4')
                # 1 x 1 x 1280

                # net = layers.conv2d(net, 640, [1, 1], scope='Conv2d_0c_1x1')
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

                net = layers_lib.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_0d')
                net = tf.expand_dims(net, 1)
                net = tf.expand_dims(net, 1)

                logits = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='Conv2d_0e_1x1')
                # 1 x 1 x 3
                if spatial_squeeze:
                    logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    # 3

    return logits


def ldnet_v1_arg_scope(weight_decay=0.0004,
                       stddev=0.1,
                       batch_norm_var_collection='moving_vars'):
    """Defines the default ldnet_v1 arg scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      batch_norm_var_collection: The name of the collection for the batch norm
        variables.
    Returns:
      An `arg_scope` to use for the ldnet_v1 model.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # Set weight_decay for weights in Conv and FC layers.
    with arg_scope(
            [layers.conv2d, layers_lib.fully_connected],
            weights_regularizer=regularizers.l2_regularizer(weight_decay)):
        with arg_scope(
                [layers.conv2d],
                weights_initializer=init_ops.truncated_normal_initializer(
                    stddev=stddev),
                activation_fn=nn_ops.relu,
                normalizer_fn=layers_lib.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc
