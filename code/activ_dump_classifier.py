import os
import numpy as np
import cPickle as pickle
import datetime
import tensorflow as tf
import math
from models.vdcnn.data_helper import *
from os.path import join as opj
from models.vdcnn.vdcnn import VDCNN
from IPython import embed

# Parameters settings
# Data loading params
tf.flags.DEFINE_string(
    "database_path", "yelp_review_polarity_csv/", "Path for the dataset to be used.")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 200,
                        "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "maxpool",
                       "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_integer(
    "depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True,
                        "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", True,
                        "Use optional shortcut (default: False)")
tf.flags.DEFINE_string("resume_model", None,
                       "Checkpoint file path to be restored")
tf.flags.DEFINE_integer("batch_size", 64, "Mini-batch size")
tf.flags.DEFINE_float("learning_rate", 0.02, "learning rate")
tf.flags.DEFINE_integer("num_epochs", 1, "epochs")
tf.flags.DEFINE_string("model_name", None, "[yelp, ag]")

FLAGS = tf.flags.FLAGS
# Data Preparation
# Load data
data_helper = data_helper(sequence_max_length=FLAGS.sequence_max_length)
train_data, train_label, train_texts, test_data, test_label, test_texts = data_helper.load_dataset(
    FLAGS.database_path)
num_batches_per_epoch = int((len(train_data)-1)/FLAGS.batch_size) + 1

# ConvNet
sess = tf.Session()
cnn = VDCNN(num_classes=train_label.shape[1],
            depth=FLAGS.depth,
            sequence_max_length=FLAGS.sequence_max_length,
            downsampling_type=FLAGS.downsampling_type,
            use_he_uniform=FLAGS.use_he_uniform,
            optional_shortcut=FLAGS.optional_shortcut)


# Initialize Graph
saver = tf.train.Saver()
saver.restore(sess, FLAGS.resume_model.replace('"', ''))

features = []
num_layers = 4
for i in range(num_layers):
    dim = int(math.pow(2, 6 + (i//2)))
    idx = 1 + (i % 2)

    if i == 0:
        layer_tensor_name = 'add:0'
    else:
        layer_tensor_name = 'add_%d:0' % (i * 2)
    layer_tensor = tf.reduce_mean(
        tf.get_default_graph().get_tensor_by_name(layer_tensor_name), axis=1)

    features.append(layer_tensor)


def forward(x_batch, y_batch):
    feed_dict = {cnn.input_x: x_batch,
                 cnn.input_y: y_batch,
                 cnn.is_training: False}
    features_run = sess.run(features, feed_dict)
    return features_run


# Generate batches
train_batches = data_helper.batch_iter(list(zip(
    train_data, train_label)), train_texts, FLAGS.batch_size, num_epochs=1, shuffle=False)

root_path = opj('results', 'activ_dump', FLAGS.model_name)
if not os.path.exists(root_path):
    os.makedirs(root_path)


batch_num = 0
budget = 500000

print '=' * 80
print 'Step 1. Feed each training sentence into classifier model,\n and save its activation to `%s`' % root_path
print '=' * 80


with tqdm(total=budget) as pbar:
# Training loop. For each batch...
    for train_batch in train_batches:
        x_batch, y_batch = zip(*train_batch[0])
        x_text = train_batch[1]

        # features_run = `list` of (batch, dim)
        features_run = forward(x_batch, y_batch)

        for batch_idx in range(len(features_run[0])):

            act_dict = {"sentence": x_text[batch_idx]}
            for layer_idx in range(len(features_run)):
                #dim = int(math.pow(2, 6 + (layer_idx // 2)))
                #name_idx = 1 + ( layer_idx % 2 )
                layer_name = 'conv_%d' % layer_idx

                act_dict[layer_name] = features_run[layer_idx][batch_idx]

            save_path = opj(root_path, '%06d.pkl' %
                            (batch_num * FLAGS.batch_size + batch_idx))

            with open(save_path, 'w') as f:
                pickle.dump(act_dict, f)

        batch_num += 1
        pbar.update(FLAGS.batch_size)

        if batch_num * FLAGS.batch_size > budget:
            break
