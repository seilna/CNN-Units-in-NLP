import tensorflow as tf
import numpy as np
import sys
import os
from vdcnn import VDCNN
from data_helper import data_helper
from os.path import join as opj


class VeryDeepCNN(object):
    def __init__(self, task='ag'):
        sequence_max_length = 200
        downsampling_type = 'maxpool'
        depth = 9
        num_layers = 4
        use_he_uniform = True
        optional_shortcut = True

        current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        model_root = opj(current_path, 'pretrained_models')

        model_path = {
                'ag': opj(model_root, 'ag', 'model-step92000.ckpt'),
                'yelp': opj(model_root, 'yelp', 'model-step282000.ckpt'),
                'ag-cam': 'pretrained_models/ag-cam/model-step10000.ckpt',
                'yelp-cam': 'pretrained_models/yelp-cam/model-step54000.ckpt',
                'dbpedia': opj(model_root, 'dbpedia', 'model-step40000.ckpt')
                }

        num_classes = {
                'ag': 4,
                'yelp': 2,
                'ag-cam': 4,
                'yelp-cam': 2,
                'dbpedia': 14
                }

        self.cnn = VDCNN(
                num_classes=num_classes[task],
                depth=depth,
                sequence_max_length=sequence_max_length,
                use_he_uniform=use_he_uniform,
                optional_shortcut=optional_shortcut
                )

        self.data_helper = data_helper(
                sequence_max_length=sequence_max_length
                )

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path[task])

        self.features = {}
        for layer_index in range(num_layers):
            if layer_index == 0:
                layer_tensor_name = 'add:0'
            else:
                layer_tensor_name = 'add_%d:0' % (layer_index * 2)

            layer_name = 'conv_%d' % layer_index
            layer_tensor = tf.get_default_graph().get_tensor_by_name(layer_tensor_name)
            
            self.features[layer_name] = layer_tensor

    def make_feed(self, x):
        feed_text = self.data_helper.char2vec(x)
        feed_text = np.expand_dims(np.array(feed_text, dtype='int32'), axis=0)

        feed_dict = {
                self.cnn.input_x: feed_text,
                self.cnn.is_training: False
                }

        return feed_dict

    def forward(self, layer_name, x):
        feed = self.make_feed(x)

        activation = self.sess.run(
                self.features[layer_name],
                feed_dict=feed
                )

        activation = activation[0]
        
        #activation = np.average(activation[0, :len(x), :], )
        return activation

    def get_loss(self, x, label):
        x = np.expand_dims(x, axis=0)
        label = np.expand_dims(label, axis=0)
        loss = self.sess.run(
                self.cnn.loss,
                feed_dict={
                    self.cnn.input_x: x,
                    self.cnn.input_y: label,
                    self.cnn.is_training: False
                    }
                )

        return loss


    def get_grad(self, layer_name, x):
        feed = self.make_feed(x)

        # TODO: replace target to Ground-Truth
        prediction = self.sess.run(
                self.cnn.predictions,
                feed_dict=feed
                )[0]

        grad_tensor = tf.gradients(
                ys=self.cnn.fc3[0, prediction],
                xs=self.features[layer_name]
                )

        
        grad = self.sess.run(
                grad_tensor,
                feed_dict=feed)

        grad_per_units = np.average(grad[0][0], axis=0)

        return prediction, grad_per_units


if __name__ == '__main__':
    print('Test')
    model = VeryDeepCNN(
            task='ag'
            )

    print('Activation:')
    print(model.forward(
        layer_name='conv_3', 
        x='Hi!'))


