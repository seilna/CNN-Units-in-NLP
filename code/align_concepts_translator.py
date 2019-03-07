from __future__ import division
from glob import glob
from os.path import join as opj
from IPython import embed
from tqdm import tqdm
from collections import Counter
import numpy as np
import os
import argparse
import math
from math import log, exp
import operator
import spacy
import time
from benepar.spacy_plugin import BeneparComponent
from polyglot.text import Text, Word
from models.bytenet.translator_protocol import ByteNetCNN
import cPickle as pickle

# disable TF debugging message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--layer_index', type=int, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--num_align', type=int, default=10)
parser.add_argument('--num_units', type=int, default=1024)


args = parser.parse_args()

model = ByteNetCNN(
    task=args.task
)

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent("benepar_en_small"))


def get_layer_name(layer_index):
    dilation = int(math.pow(2, layer_index % 5))
    return 'bytenet_encoder_layer_%d_%d' % (layer_index, dilation)


def load_tas(task, layer_index):

    layer_name = get_layer_name(layer_index)

    path = opj(
        'results',
        'top_activated_sentences',
        task,
        'top-%d' % args.top_k,
        layer_name + '.pkl'
    )

    tas = pickle.load(open(path))

    return tas


def explore_parse_tree(tokens, tree_nodes):

    # for word token, obtain morphemes
    if len(tokens) == 1:
        w = Word(tokens.text, language='en')
        for morph in w.morphemes:
            if morph == tokens.text:
                continue
            tree_nodes.append((morph, 'morpheme', 1))

        # add word itself
        tree_nodes.append((tokens, tokens._.labels, len(tokens)))

    if tokens._.labels == () or len(tokens) == 0:
        return 0

    tree_nodes.append((tokens, tokens._.labels, len(tokens)))

    # constituency parsed nodes
    for child in tokens._.children:
        explore_parse_tree(child, tree_nodes)


def get_concept_cands(tas, unit):

    concept_cands = []
    for activ, sent in tas[unit]:

        try:
            tokens = nlp(unicode(sent))
        except:
            continue

        # construct phrase candidates
        nodes = []
        if len(list(tokens.sents)) == 0:
            continue

        explore_parse_tree(list(tokens.sents)[0], tree_nodes=nodes)

        for tokens, constituency_label, num_words in nodes:
            try:
                # for words and morphemes
                normalized_cand = tokens.lemma_
            except:
                # for morphemes (type(tokens) = `str`)
                normalized_cand = tokens

            concept_cands.append((tokens, constituency_label, normalized_cand))

    return concept_cands


def generate_replicate_text(concepts, length=70):
    texts = []

    for concept, const_label, normalized_cand in concepts:

        text = ' '.join(['%s' % normalized_cand] * 1000)
        text = text[:length]
        texts.append(text)

    return texts


def compute_mu_replicate(layer_name, unit, concept_cands, replicated_texts):
    mu_replicate_per_unit = {}

    for (tokens, const_label, normalized_cand), replicated_text in zip(concept_cands, replicated_texts):
        # pass already computed candidates
        if normalized_cand in mu_replicate_per_unit:
            continue

        activ = model.forward(
            layer_name=layer_name,
            x=replicated_text
        )

        activ = np.average(activ[:len(replicated_text), unit])

        if const_label == 'morpheme':
            cand_key = 'MORPH_%s' % normalized_cand
        else:
            cand_key = normalized_cand

        mu_replicate_per_unit[cand_key] = activ

    return mu_replicate_per_unit


def save_result(concept_doa):

    wname = 'layer_%02d_top-%d_M=%02d_units=%04d.pkl' % \
            (args.layer_index, args.top_k, args.num_align, args.num_units)

    save_root = opj('results', 'concept_alignment', args.task)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    path = opj(save_root, wname)

    print 'Saving its results at %s ...' % path

    with open(path, 'w') as f:
        pickle.dump(concept_doa, f)


def main():
    layer_name = get_layer_name(args.layer_index)

    tas = load_tas(args.task, args.layer_index)

    concept_doa = {}

    num_units = args.num_units
    print '=' * 120
    print 'Step 3. Align concepts to each unit with replicatd sentences'

    for unit in tqdm(range(num_units)):
        concept_cands = get_concept_cands(tas, unit)

        average_length = {
            'en-de-europarl': 149,
            'en-de-news': 139,
            'en-fr-news': 140,
            'en-cs-news': 134,
        }

        replicate_length = int(average_length[args.task])

        replicated_texts = generate_replicate_text(
            concept_cands,
            length=replicate_length
        )

        assert len(concept_cands) == len(replicated_texts)

        mu_replicate_per_unit = compute_mu_replicate(
            layer_name,
            unit,
            concept_cands,
            replicated_texts)

        # align `M` concepts which have highest concept selectivity
        concept_doa[unit] = Counter(
            mu_replicate_per_unit).most_common(args.num_align)

    save_result(concept_doa)


if __name__ == '__main__':
    main()
