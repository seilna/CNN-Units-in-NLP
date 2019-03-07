from os.path import join as opj
from tqdm import tqdm
from collections import Counter
import numpy as np
import os
import argparse
import spacy
import cPickle as pickle
import tensorflow as tf
from models.vdcnn.classifier_protocol import VeryDeepCNN
from benepar.spacy_plugin import BeneparComponent
from polyglot.text import Text, Word


tf.flags.DEFINE_integer('layer_index', None, 'layer index')
tf.flags.DEFINE_integer('top_k', 10, '')
tf.flags.DEFINE_string('task', None, '')
tf.flags.DEFINE_integer('num_align', 10, 'number of concepts to be aligned')
tf.flags.DEFINE_integer('num_units', None, '')




FLAGS = tf.flags.FLAGS
model = VeryDeepCNN(
    task=FLAGS.task
)

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent("benepar_en_small"))


def lemma_custom(token):
    if token.lemma_ == '-PRON-':
        return token.text

    if token.lemma_ == 'be':
        return token.text

    return token.lemma_


def get_layer_name(layer_index):
    return 'conv_%d' % layer_index


def load_tas(task, layer_index):

    layer_name = get_layer_name(layer_index)

    path = opj('results', 'top_activated_sentences', task, 'top-%d' % FLAGS.top_k, layer_name + '.pkl')

    tas = pickle.load(open(path))

    return tas


def explore_parse_tree(tokens, tree_nodes):

    # for word token, obtain morphemes
    if len(tokens) == 1:
        w = Word(tokens.text, language='en')
        for morph in w.morphemes:
            if morph == tokens.text or morph == tokens.lemma_:
                continue
            tree_nodes.append((morph, 'morpheme', 1))

        # add word itself
        tree_nodes.append((tokens.text, tokens._.labels, len(tokens)))

    if tokens._.labels == () or len(tokens) == 0:
        return 0

    tree_nodes.append((tokens.text, tokens._.labels, len(tokens)))

    # constituency parsed nodes
    for child in tokens._.children:
        explore_parse_tree(child, tree_nodes)


def get_concept_cands(tas, unit):
    concept_cands = []
    for activ, sent in tas[unit]:
        sent = sent[:298] # max sentence length spacy suppoerted.
        tokens = nlp(sent.decode('utf-8', errors='ignore'))
        #tokens = nlp(sent)

        # construct phrase candidates
        nodes = []
        if len(list(tokens.sents)) == 0:
            continue

        explore_parse_tree(list(tokens.sents)[0], tree_nodes=nodes)

        for tokens, constituency_label, num_words in nodes:
            try:
                normalized_cand = tokens.lemma_
            except:
                normalized_cand = tokens

            concept_cands.append((tokens, constituency_label, normalized_cand))

    return concept_cands


def generate_replicate_text(concepts, length=70):
    texts = []
    for concept, const_label, normalized_cand in concepts:
        text = ' '.join(['%s' % normalized_cand] * 10)
        text = text[:length]
        texts.append(text)

    return texts


def compute_mu_replicate(layer_name, unit, concept_cands, replicated_texts):
    mu_replicate_per_unit = {}

    for (tokens, const_label, normalized_cand), replicated_text in zip(concept_cands, replicated_texts):
        if const_label == 'morpheme':
            cand_key = 'MORPH_%s' % normalized_cand
        else:
            cand_key = normalized_cand

        # pass already computed candidates
        if cand_key in mu_replicate_per_unit:
            continue

        activ = model.forward(
            layer_name=layer_name,
            x=replicated_text
        )

        activ = np.average(activ[:len(replicated_text), unit])

        mu_replicate_per_unit[cand_key] = activ

    return mu_replicate_per_unit


def save_result(concept_doa, num_units):


    wname = 'layer_%02d_top-%d_M=%02d_units=%04d.pkl' % (
        FLAGS.layer_index, FLAGS.top_k, FLAGS.num_align, num_units)

    save_root = opj('results', 'concept_alignment', FLAGS.task)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    path = opj(save_root, wname)

    print 'Saving its results at %s ...' % path

    with open(path, 'w') as f:
        pickle.dump(concept_doa, f)

def main():
    layer_name = get_layer_name(FLAGS.layer_index)

    tas = load_tas(FLAGS.task, FLAGS.layer_index)

    concept_doa = {}

    num_units = min([FLAGS.num_units, int(2 ** (FLAGS.layer_index + 6))])

    print '=' * 120
    print 'Step 3. Align concepts to each unit with replicatd sentences'

    for unit in tqdm(range(num_units)):
        concept_cands = get_concept_cands(tas, unit)


        length = 193

        replicated_texts = generate_replicate_text(
            concept_cands, length=length)
        assert len(concept_cands) == len(replicated_texts)

        mu_replicate_per_unit = compute_mu_replicate(
            layer_name,
            unit,
            concept_cands,
            replicated_texts)

        # align N=10 concepts which have highest concept selectivity
        concept_doa[unit] = Counter(
            mu_replicate_per_unit).most_common(FLAGS.num_align)

    # align `M` concepts which have highest concept selectivity
    save_result(concept_doa, num_units)


if __name__ == '__main__':
    main()
