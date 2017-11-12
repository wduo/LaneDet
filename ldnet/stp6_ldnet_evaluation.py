import tensorflow as tf

from stp5_ldnet_train import MODEL_SAVE_PATH

# constants describing the current file.
EVALUATION_PATH = "/tmp/ldnet/evaluation"


def evaluate(saver):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/train/ldnet_model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return


def main(_):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
