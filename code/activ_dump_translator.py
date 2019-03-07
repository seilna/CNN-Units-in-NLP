# coding: utf-8
import sys
sys.path.append('models/bytenet')

import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import translator
from pprint import pprint
from os.path import join as opj
from IPython import embed
from tqdm import tqdm
import utils
import shutil
import time
import math
import os
import cPickle as pickle

# disable TF debugging message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default=None)

parser.add_argument('--model_path', type=str, default='Data/Models/translation_model/en.blank-zh, d=512, e_w=5/model_epoch_0_250.ckpt',
                   help='restore model path')

parser.add_argument('--learning_rate', type=float, default=0.001,
                   help='Learning Rate')
parser.add_argument('--batch_size', type=int, default=8,
                   help='Learning Rate')
parser.add_argument('--bucket_quant', type=int, default=50,
                   help='Learning Rate')
parser.add_argument('--max_epochs', type=int, default=1000,
                   help='Max Epochs')
parser.add_argument('--beta1', type=float, default=0.5,
                   help='Momentum for Adam Update')
parser.add_argument('--resume_model', type=str, default=None,
                   help='Pre-Trained Model Path, to resume from')
parser.add_argument('--source_file', type=str, default='Data/translator_training_data/um.en.blank',
                   help='Source File')
parser.add_argument('--target_file', type=str, default='Data/translator_training_data/um.zh',
                   help='Target File')
parser.add_argument('--sample_every', type=int, default=200,
                   help='Sample generator output evry x steps')
parser.add_argument('--summary_every', type=int, default=50,
                   help='Sample generator output evry x steps')
parser.add_argument('--save_every', type=int, default=10000,
                   help='Saving model cycle')

parser.add_argument('--top_k', type=int, default=5,
                   help='Sample from top k predictions')
parser.add_argument('--bucket_size', type=int, default=350,
                   help='Sample bucket size')
parser.add_argument('--resume_from_bucket', type=int, default=0,
                   help='Resume From Bucket')

parser.add_argument('--tag', type=str, default='no-tag',
                   help='tag')

parser.add_argument('--split', type=str, default='train',
                    help='choose which split of data to use '
                         '(`train` or `valid`)')
parser.add_argument('--num_layers', type=int, default=15,
                   help='num of layers')

args = parser.parse_args()

model_path = args.model_path

data_loader_options = {
    'model_type' : 'translation',
    'source_file' : args.source_file,
    'target_file' : args.target_file,
    'bucket_quant' : args.bucket_quant,
}

dl = data_loader.Data_Loader(data_loader_options,
                             split=args.split,
                             vocab=None)
buckets, source_vocab, target_vocab = dl.load_translation_data()
config = model_config.translator_config

model_options = {
    'source_vocab_size' : len(source_vocab),
    'target_vocab_size' : len(target_vocab),
    'residual_channels' : config['residual_channels'],
    'decoder_dilations' : config['decoder_dilations'],
    'encoder_dilations' : config['encoder_dilations'],
    'decoder_filter_width' : config['decoder_filter_width'],
    'encoder_filter_width' : config['encoder_filter_width'],
    'layer_norm': config['layer_norm']
}

translator_model = translator.ByteNet_Translator( model_options )
translator_model.build_model()

translator_model.build_translator(reuse = True)


sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore(sess, model_path)


features = []

for layer in range(args.num_layers):
    dilation = int(math.pow(2, layer % 5))

    feature = tf.get_default_graph().get_tensor_by_name("bytenet_encoder_layer_{}_{}/add:0".format(layer, dilation))

    feature = tf.squeeze(tf.reduce_mean(feature, axis=1))
    features.append(feature)

root_path = opj('results', 'activ_dump', '%s' % args.model_path.split('/')[-2])
if not os.path.exists(root_path):
    os.makedirs(root_path)

batch_size = args.batch_size
bucket_size = args.bucket_size

print '=' * 80
print 'Step 1. Feed each training sentence into translator model,\n and save its activation to `%s/bucket%03d-*.pkl`' % (root_path, bucket_size)
print '=' * 80

# for lite version, set $budget to small value
budget = 100000
max_batch_idx = min(len(buckets[bucket_size]) - batch_size, budget)

for batch_num in tqdm(range(0, max_batch_idx, batch_size), dynamic_ncols=True):
    act_dict = {}

    source, target, _ = dl.get_batch_from_pairs( 
        buckets[bucket_size][batch_num : batch_num + batch_size] 
    )

    raw_source_texts = [ dl.inidices_to_string(sentence=source[i], vocab=source_vocab) for i in range(len(source)) ]

    features_ = sess.run(features, feed_dict = {
            translator_model.source_sentence : source,
        })

    for i in range(batch_size):
        sentence = raw_source_texts[i]
        act_dict = {'sentence': sentence}
        for j, feature in enumerate(features_):
            dilation =int(math.pow(2, j % 5))
            layer_name = 'bytenet_encoder_layer_{}_{}'.format(j, dilation)

            act_dict[layer_name] = features_[j][i]

        save_path = opj(root_path, 'bucket%03d-%06d.pkl' % (bucket_size, batch_num + i))
        with open(save_path, 'w') as w:
            pickle.dump(act_dict, w)
