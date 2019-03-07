import os
import numpy as np
import datetime
import tensorflow as tf
from data_helper import *
from os.path import join as opj
# State which model to use here
from vdcnn import VDCNN

# Parameters settings
# Data loading params
tf.flags.DEFINE_string("database_path", "ag_news_csv/", "Path for the dataset to be used.")
tf.flags.DEFINE_string("model_name", "ag", "Path for the dataset to be used.")
tf.flags.DEFINE_boolean("use_title", False, "classification with only title")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 200, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "maxpool", "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", True, "Use optional shortcut (default: False)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")
tf.flags.DEFINE_integer("save_every", 2000, "save model after this many steps (default: 2000)")


FLAGS = tf.flags.FLAGS
# Data Preparation
# Load data
print("Loading data...")
data_helper = data_helper(
        sequence_max_length=FLAGS.sequence_max_length,
        use_title=FLAGS.use_title
        )
train_data, train_label, train_texts, test_data, test_label, test_texts = data_helper.load_dataset(FLAGS.database_path)
num_batches_per_epoch = int((len(train_data)-1)/FLAGS.batch_size) + 1
print("Loading data succees...")

# ConvNet
acc_list = [0]
sess = tf.Session()
cnn = VDCNN(num_classes=train_label.shape[1], 
	depth=FLAGS.depth,
	sequence_max_length=FLAGS.sequence_max_length, 
	downsampling_type=FLAGS.downsampling_type,
	use_he_uniform=FLAGS.use_he_uniform,
	optional_shortcut=FLAGS.optional_shortcut)

# Optimizer and LR Decay
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
	train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


# Initialize Graph
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=100)


# Train Step and Test Step
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {cnn.input_x: x_batch, 
                                cnn.input_y: y_batch, 
                                cnn.is_training: True}
    _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    
    if step % 100 == 0:
        print("{}: Step {}, Epoch {}, Loss {:g}, Acc {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, accuracy))
    #if step%FLAGS.evaluate_every == 0 and FLAGS.enable_tensorboard:
    #	summaries = sess.run(train_summary_op, feed_dict)
    #	train_summary_writer.add_summary(summaries, global_step=step)

def test_step(x_batch, y_batch):
	"""
	Evaluates model on a dev set
	"""
	feed_dict = {cnn.input_x: x_batch, 
				 cnn.input_y: y_batch, 
				 cnn.is_training: False}
	loss, preds = sess.run([cnn.loss, cnn.predictions], feed_dict)
	time_str = datetime.datetime.now().isoformat()
	return preds, loss

# Generate batches
train_batches = data_helper.batch_iter(list(zip(train_data, train_label)), train_texts, FLAGS.batch_size, FLAGS.num_epochs)

# Training loop. For each batch...
for train_batch in train_batches:
    x_batch, y_batch = zip(*train_batch[0])
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    # Testing loop
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        i = 0
        index = 0
        sum_loss = 0
        test_batches = data_helper.batch_iter(list(zip(test_data, test_label)), test_texts, FLAGS.batch_size, 1, shuffle=False)
        y_preds = np.ones(shape=len(test_label), dtype=np.int)
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch[0])
            preds, test_loss = test_step(x_test_batch, y_test_batch)
            sum_loss += test_loss
            res = np.absolute(preds - np.argmax(y_test_batch, axis=1))
            y_preds[index:index+len(res)] = res
            i += 1
            index += len(res)
        time_str = datetime.datetime.now().isoformat()
        acc = np.count_nonzero(y_preds==0)/len(y_preds)
        acc_list.append(acc)
        print("{}: Evaluation Summary, Loss {:g}, Acc {:g}".format(time_str, sum_loss/i, acc))
        print("{}: Current Max Acc {:g} at Iteration {}".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*FLAGS.evaluate_every)))
    if current_step > 0 and current_step % FLAGS.save_every == 0:
        save_path = opj('logs', FLAGS.model_name, 'model-step%05d.ckpt' % current_step)
        saver.save(sess, save_path)
