# -*- coding: utf-8 -*-

from tqdm import tqdm
from IPython import embed
import csv
import numpy as np
import random
import os
from os.path import join as opj


# disable TF debugging message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class data_helper():
    def __init__(self, sequence_max_length=1024, use_title=False):
        self.alphabet = unicode('abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ', 'utf-8')
        #self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '

        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        for i, c in enumerate(self.alphabet):
            self.char_dict[c] = i+1
        
        self.use_title = use_title

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            if i >= self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                # unknown character set to be last index + 1
                data[i] = len(self.char_dict) + 1
        return data

    def load_csv_file(self, filename, num_classes):
        """
        Load CSV file, generate one-hot labels and process text data as Paper did.
        """
        all_data = []
        labels = []
        texts = []
        with open(filename) as f:
            reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
            for row in reader:
                # One-hot
                one_hot = np.zeros(num_classes)
                one_hot[int(row['class']) - 1] = 1
                labels.append(one_hot)
                # Char2vec
                data = np.ones(self.sequence_max_length)*68

                if self.use_title:
                    text = row['fields'][0].lower().replace('"', '')
                else:
                    text = row['fields'][-1].lower().replace('"', '')
                texts.append(text)
                all_data.append(self.char2vec(text))
        f.close()
        return np.array(all_data), np.array(labels), texts

    def load_dataset(self, dataset_path):
        # Read Classes Info
        with open(opj(dataset_path, "classes.txt")) as f:
            classes = []
            for line in f:
                classes.append(line.strip())
        f.close()
        num_classes = len(classes)
        # Read CSV Info
        train_data, train_label, train_texts = self.load_csv_file(
            opj(dataset_path, 'train.csv'), num_classes)
        test_data, test_label, test_texts = self.load_csv_file(
            opj(dataset_path, 'test.csv'), num_classes)
        return train_data, train_label, train_texts, test_data, test_label, test_texts

    def batch_iter(self, data, texts, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        assert data_size == len(texts), 'len(data) = %d, len(texts) = %d' % (
            data_size, len(texts))
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index], texts[start_index: end_index]
