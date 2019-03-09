#-*- coding: utf-8 -*-
import cPickle as pickle
import argparse
import os
from os.path import join as opj
from IPython import embed as eb
from utils import html_header, html_per_unit, html_per_tas
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--top_k', type=int, required=True)
parser.add_argument('--num_align', type=int, required=True)
parser.add_argument('--num_units', type=int, required=True)


args = parser.parse_args()

def load_tas(task, layer):
    root = opj('results', 'top_activated_sentences', task, 'top-%d' % args.top_k)

    layer_name = 'bytenet_encoder_layer_%d_%d.pkl' % (layer, 2 ** (layer % 5) )\
            if 'en-' in task else\
            'conv_%d.pkl' % layer

    path = opj(root, layer_name)
    return pickle.load(open(path))

def load_alignment(task, layer, num_units):
    path = opj('results', 'concept_alignment', task, 'layer_%02d_top-%d_M=%02d_units=%04d.pkl' % (layer, args.top_k, args.num_align, num_units))

    assert layer < num_layers

    alignment = pickle.load(open(path))
    return alignment

def write_html(task, layer, alignment, num_units, num_align, tas):
    """write_html

    Args:
        task: one of ['en-de-news', 'en-fr-news', 'en-cs-news', 'en-de-europarl', 'ag', 'yelp', 'dbpedia']
        layer: 0~14 (translation model) or 0~3 (classification model)
        alignment: concept alignment results
        num_units: the number of units at givne `layer`
        num_align: the number of concepts to align
        tas: top-activated-sentences

    Return:
        write concept alignment result as html file to `../visualization` 
    """

    root = opj('../', 'visualization', '%s' % task)
    if not os.path.exists(root):
        os.makedirs(root)

    fname = opj(root, 'layer_%02d.html' % layer)

    header = html_header()
    with open(fname, 'w') as f:
        f.write(header)

        for unit in range(num_units):
            """write aligned `num_align` concepts to html file"""
            content = html_per_unit(task, layer, unit, alignment, num_align)
            f.write(content)

            """write top-activated-sentences"""
            content = html_per_tas(unit, tas, alignment, 5, 100)
            f.write(content)

    print 'Visualization result are saved at %s' % os.path.abspath(fname)


if __name__ == '__main__':

    num_layers = 15 if 'en-' in args.task else 4

    for layer in range(num_layers):
        if 'en-' in args.task:
            num_units = min([args.num_units, 1024])
        else:
            num_units = min([args.num_units, 2 ** (layer + 6)])

        alignment = load_alignment(args.task, layer, num_units)
        tas = load_tas(args.task, layer)


        write_html(args.task, layer, alignment, num_units, args.num_align, tas)
