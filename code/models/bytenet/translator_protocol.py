import tensorflow as tf
import numpy as np
import data_loader
import model_config
import math
import os
from ByteNet import translator
from os.path import join
from IPython import embed

class ByteNetCNN(object):
    def __init__(self, task):
        bucket_quant = 10

        current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        translator_root_path = join(current_path, 'pretrained_models')

        model_path = {
                'en-de-news': join(translator_root_path, 'en-de-news', 'model_epoch_4_145000.ckpt'),
                'en-fr-news': join(translator_root_path, 'en-fr-news', 'model_epoch_4_90000.ckpt'),
                'en-cs-news': join(translator_root_path, 'en-cs-news', 'model_epoch_4_70000.ckpt'),
                'en-de-europarl': join(translator_root_path, 'en-de-europarl', 'model_epoch_1_440000.ckpt')
                }

        data_root_path = join(current_path, 'Data', 'translator_training_data')
        source_file = {
                'en-de-europarl': join(data_root_path, 'europarl-v7.de-en.en'),
                'en-de-news': join(data_root_path, 'news-commentary-v12.de-en.en'),
                'en-fr-news': join(data_root_path, 'news-commentary-v9.fr-en.en'),
                'en-cs-news': join(data_root_path, 'news-commentary-v9.cs-en.en')
                }

        target_file = {
                'en-de-europarl': join(data_root_path, 'europarl-v7.de-en.de'),
                'en-de-news': join(data_root_path, 'news-commentary-v12.de-en.de'),
                'en-fr-news': join(data_root_path, 'news-commentary-v9.fr-en.fr'),
                'en-cs-news': join(data_root_path, 'news-commentary-v9.cs-en.cs')
                }

        data_loader_options = {
                'model_type': 'translation',
                'source_file': source_file[task],
                'target_file': target_file[task],
                'bucket_quant': bucket_quant
                }


        self.dl = data_loader.Data_Loader(data_loader_options)
        self.buckets, self.source_vocab, self.target_vocab = self.dl.load_translation_data()

        config = model_config.translator_config

        model_options = {
            'source_vocab_size' : len(self.source_vocab),
            'target_vocab_size' : len(self.target_vocab),
            'residual_channels' : config['residual_channels'],
            'decoder_dilations' : config['decoder_dilations'],
            'encoder_dilations' : config['encoder_dilations'],
            'decoder_filter_width' : config['decoder_filter_width'],
            'encoder_filter_width' : config['encoder_filter_width'],
            'layer_norm': config['layer_norm']
        }

        self.translator_model = translator.ByteNet_Translator( model_options )
        self.translator_model.build_model()
        self.translator_model.build_translator(reuse=True)

        self.sess = tf.Session()
        saver = tf.train.Saver()

        if model_path[task]:
            saver.restore(self.sess, model_path[task])

        self.features = {}
        for layer_index in range(15):
            dilation = int(math.pow(2, layer_index % 5))

            layer_tensor_name = "bytenet_encoder_layer_%d_%d/add:0" % (layer_index, dilation)
            layer_name = "bytenet_encoder_layer_%d_%d" % (layer_index, dilation)
            self.features[layer_name] = tf.get_default_graph().get_tensor_by_name(layer_tensor_name)

    def get_layer_name(self, layer_index):
        dilation = int(math.pow(2, layer_index % 5))
        layer_name = 'bytenet_encoder_layer_%d_%d' % (layer_index, dilation)
        return layer_name

    def make_feed(self, x):
        buckets = self.dl.create_buckets_only_src(
                [self.dl.string_to_indices(x, self.source_vocab)])

        bucket_size = buckets.keys()[0]
        feed_text = np.array(buckets[bucket_size], dtype='int32')

        return feed_text

    def forward(self, layer_name, x):
        feed_text = self.make_feed(x)

        activation = self.sess.run(
                self.features[layer_name],
                feed_dict={
                    self.translator_model.source_sentence: feed_text
                    }
                )

        # activation.shape = (len, # units)
        activation = activation[0]

        return activation

    def get_loss(self, src, tgt):

        src = np.reshape(src, [1, -1])
        tgt = np.reshape(tgt, [1, -1])

        src = np.array(src, dtype='int32')
        tgt = np.array(tgt, dtype='int32')


        loss = self.sess.run(
                self.translator_model.loss,
                feed_dict={
                    self.translator_model.source_sentence: src,
                    self.translator_model.target_sentence: tgt
                    }
                )
        
        return loss

    def get_grad(self, src, tgt):
        src = np.reshape(src, [1, -1])
        tgt = np.reshape(tgt, [1, -1])

        src = np.array(src, dtype='int32')
        tgt = np.array(tgt, dtype='int32')

        src_embed = [v for v in tf.global_variables() if 'source_embedding' in v.name][0]

        grad = self.sess.run(
                tf.gradients(self.translator_model.loss, src_embed),
                feed_dict={
                    self.translator_model.source_sentence: src,
                    self.translator_model.target_sentence: tgt
                    }
                )

        # grad.shape = (l, d)
        grad = grad[0].values

        # grad.shape = (l)
        grad = np.average(
                grad,
                axis=1
                )

        return grad

if __name__ == '__main__':
    print('Test')
    model = ByteNetCNN(
            task='en-de-europarl',
            preR=False
            )

    print('Activation:')
    print(model.forward(
        layer_name='bytenet_encoder_layer_14_16', 
        x='This is a example sentence.').shape)

    from IPython import embed; embed()




