import argparse
import numpy as np
import os
import json
import math
import cPickle as pickle
from glob import glob
from pprint import pprint
from tqdm import tqdm
from IPython import embed
from os.path import join as opj

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='en-de,d=512,e_w=3')
parser.add_argument('--layer_index', type=int, required=True)
parser.add_argument('--task', type=str, default='translation')
parser.add_argument('--top_k', type=int, required=True)

args = parser.parse_args()
dilation = int(math.pow(2, args.layer_index % 5))

root_path = opj('results', 'top_activated_sentences', args.model_name, 'top-%d' % args.top_k)

num_units = 1024 if args.task == 'translation' else 512
top_k = args.top_k

def duplicate_check(activations, sentence):
    for act, text in activations:
        if text == sentence: return True

    return False

def top_act_unit(layer_name, num_comb_units=1):
    """Given layer depth, save maximally activated sentences on each unit (combinations).

    Args:
        layer_name(str): 'bytenet_encoder_layer_X_X'  
        num_comb_units(int): the number of units to be combined

    Returns:
    """

    if args.task == 'translation':
        num_total_units = num_units
    elif args.task == 'classification':
        num_total_units = int(math.pow(2, (args.layer_index + 6)))
    else: raise


    top_act_queue = {}

    fnames = glob(opj('results', 'activ_dump', '{}'.format(args.model_name), '*.pkl'))[:]
    for fname in tqdm(fnames, dynamic_ncols=True):
        activ_dict = pickle.load(open(fname))

        for unit in range(num_total_units):
            
            if unit not in top_act_queue:
                top_act_queue[unit] = []  

            sentence = activ_dict['sentence']

            activ = round(activ_dict[layer_name][unit], 3)


            if duplicate_check(top_act_queue[unit], sentence):
                continue

            # preserve size of top_act_queue as `top_k`
            if len(top_act_queue[unit]) < top_k:
                top_act_queue[unit].append( (activ, sentence) )
            else:
                min_value = min(top_act_queue[unit])
                min_index = top_act_queue[unit].index(min_value)

                if activ > min_value[0]:
                    top_act_queue[unit][min_index] = (activ, sentence)

    for unit in range(num_total_units):
        # For each unit, sort its top-activated sentencs 
        # by decreasing order with its activation value.
        top_act_queue[unit] = sorted(top_act_queue[unit], reverse=True)
        assert len(top_act_queue[unit]) == top_k

    # Save results

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    save_path = opj(root_path, '{}.pkl'.format(layer_name))
    with open(save_path, 'w') as w:
        pickle.dump(top_act_queue, w)
        
           
def main():
    assert len(args.model_name) > 0, 'Wrong model path ->  %s' % args.model_name
   
    if args.task == 'translation':
        layer_name = 'bytenet_encoder_layer_%d_%d' % (args.layer_index, dilation)
    elif args.task == 'classification':
        layer_name = 'conv_%d' % args.layer_index
    else:
        raise

    print '=' * 120
    print 'Step 2. Save Top-Activated-Sentences for layer %s at %s ...' % (layer_name, root_path)
    top_act_unit(layer_name)

if __name__ == '__main__':
    main()
