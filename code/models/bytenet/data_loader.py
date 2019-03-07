import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import cPickle as pickle
import json

class Data_Loader:
    def __init__(self, options, split='train', vocab=None, cache=None):
        if options['model_type'] == 'translation':
            source_file = options['source_file'] + '.' + split
            target_file = options['target_file'] + '.' + split

            self.max_sentences = None
            if 'max_sentences' in options:
                self.max_sentences = options['max_sentences']

            with open(source_file) as f:
                self.source_lines = f.read().decode("utf-8", errors='ignore').split('\n')
                # for temporally covering error in inference_translation.py
            with open(target_file) as f:
                self.target_lines = f.read().decode("utf-8", errors='ignore').split('\n')


            if self.max_sentences:
                self.source_lines = self.source_lines[0:self.max_sentences]
                self.target_lines = self.target_lines[0:self.max_sentences]

            #print("Source Sentences", len(self.source_lines))
            #print("Target Sentences", len(self.target_lines))

            self.bucket_quant = options['bucket_quant']
            if split == "train":
                self.source_vocab = self.build_vocab(self.source_lines)
                self.target_vocab = self.build_vocab(self.target_lines)
            else:
                '''
                if vocab is None:
                    raise Exception("split={}: need vocab from training data"
                                    % split)
                with open(join(vocab, "source_vocab.pkl"), "rb") as f:
                    self.source_vocab = pickle.load(f)
                with open(join(vocab, "target_vocab.pkl"), "rb") as f:
                    self.target_vocab = pickle.load(f)
                '''
                pass


            #print("SOURCE VOCAB SIZE", len(self.source_vocab))
            #print("TARGET VOCAB SIZE", len(self.target_vocab))
        
        elif options['model_type'] == 'generator':
            dir_name = options['dir_name']
            files = [ join(dir_name, f) for f in listdir(dir_name) if  isfile(join(dir_name, f))   ]

            text = []
           
            # construct same vocab set both for train/valid data
            if vocab == None:
                print('There is no vocab file. Construct it from scratch.')
                for f in files:
                    text += list(open(f).read())

                vocab = {ch : True for ch in text}

            # If there is vocab cache, load it only!
            elif type(vocab) == str:
                print('Loading presaved vocab file | {}'.format(vocab))
                vocab = pickle.load(open(vocab))

            else:
                for f in files:
                    text += list(open(f).read())


            self.vocab = vocab
		
            print("Bool vocab", len(vocab))
            self.vocab_list = [ch for ch in vocab]
            print("vocab list", len(self.vocab_list))
            self.vocab_indexed = {ch : i for i, ch in enumerate(self.vocab_list)}
            print("vocab_indexed", len(self.vocab_indexed))

            if cache:
              print('text cache file [{}] is started to load...'.format(cache))
              self.text = np.load(cache)
              print('text cache file [{}] is loaded.'.format(cache))

            else:
              for index, item in enumerate(text):
                  text[index] = self.vocab_indexed[item]
              self.text = np.array(text, dtype='int32')
              
        elif options['model_type'] == 'classifier':
            text, rating = [], []

            fname = options['review_file'] + '.{}'.format(split)

            if not vocab: vocab_scratch = {'<p>': 0}
            
            with open(fname) as f:
                while True:
                    line = f.readline()
                    if not line: break
                    review = json.loads(line)

                    if not vocab:
                        for ch in review['text'].replace('\n', ''):
                            if ch in vocab_scratch: continue
                            else: vocab_scratch[ch] = len(vocab_scratch)

                    # make polarity
                    if int(review['stars']) > 3:
                        rating.append(1)
                    elif int(review['stars']) < 3:
                        rating.append(0)
                    else: continue

                    text.append( self.string_to_indices(review['text'].replace('\n', ''), vocab_scratch, pad=options['seq_len']) )

            self.text = np.array(text, dtype='int32')
            self.rating = np.array(rating, dtype='int32')
            self.vocab = vocab_scratch

    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def load_translation_data(self):
        source_lines = []
        target_lines = []
        for i in range(len(self.source_lines)):
            source_lines.append( self.string_to_indices(self.source_lines[i], self.source_vocab) )
            target_lines.append( self.string_to_indices(self.target_lines[i], self.target_vocab) )

        buckets = self.create_buckets(source_lines, target_lines)

        # frequent_keys = [ (-len(buckets[key]), key) for key in buckets ]
        # frequent_keys.sort()

        # print "Source", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][0], self.source_vocab)
        # print "Target", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][1], self.target_vocab)
        
        return buckets, self.source_vocab, self.target_vocab

    def load_classifier_data(self):
        return self.text, self.rating, self.vocab
        

    def create_buckets(self, source_lines, target_lines):
        
        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab
        target_vocab = self.target_vocab

        buckets = {}
        for i in xrange(len(source_lines)):
           
            # source = source + <EOL>
            # target = <BOL> + target + <EOL>
            source_lines[i] = np.concatenate( (source_lines[i], [source_vocab['eol']]) )
            target_lines[i] = np.concatenate( ([target_vocab['init']], target_lines[i], [target_vocab['eol']]) )
            
            sl = len(source_lines[i])
            tl = len(target_lines[i])

            new_length = max(sl, tl)

            # fitting new_length to neareast upperbound of bucket_quant
            # e.g. bucket_quant=50 -> new_length = 50, 100, 150, ...

            if new_length % bucket_quant > 0:
                new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant    
            
            s_padding = np.array( [source_vocab['padding'] for ctr in xrange(sl, new_length) ] )

            # NEED EXTRA PADDING FOR TRAINING.. 
            t_padding = np.array( [target_vocab['padding'] for ctr in xrange(tl, new_length + 1) ] )

            source_lines[i] = np.concatenate( [ source_lines[i], s_padding ] )
            target_lines[i] = np.concatenate( [ target_lines[i], t_padding ] )

            if new_length in buckets:
                buckets[new_length].append( (source_lines[i], target_lines[i]) )
            else:
                buckets[new_length] = [(source_lines[i], target_lines[i])]

            #if i%100000 == 0 and i > 0:
            #    print("Loading", i)
            
        return buckets


    def create_buckets_only_src(self, source_lines):
        
        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab

        buckets = {}
        for i in xrange(len(source_lines)):
           
            # source = source + <EOL>
            # target = <BOL> + target + <EOL>
            source_lines[i] = np.concatenate( (source_lines[i], [source_vocab['eol']]) )
            
            sl = len(source_lines[i])
            new_length = sl

            # fitting new_length to neareast upperbound of bucket_quant
            # e.g. bucket_quant=50 -> new_length = 50, 100, 150, ...
            if new_length % bucket_quant > 0:
                new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant    
            
            s_padding = np.array( [source_vocab['padding'] for ctr in xrange(sl, new_length) ] )

            source_lines[i] = np.concatenate( [ source_lines[i], s_padding ] )

            if new_length in buckets:
                buckets[new_length].append( source_lines[i] )
            else:
                buckets[new_length] = [ source_lines[i] ]

            
        return buckets


    def build_vocab(self, sentences):
        vocab = {}
        ctr = 0
        for st in sentences:
            for ch in st:
                if ch not in vocab:
                    vocab[ch] = ctr
                    ctr += 1

        # SOME SPECIAL CHARACTERS
        vocab['eol'] = ctr # end of line
        vocab['padding'] = ctr + 1  # padding
        vocab['init'] = ctr + 2 # init

        return vocab

    def string_to_indices(self, sentence, vocab, pad=-1):
        indices = []
        for s in sentence:
            try: indices.append(vocab[s])
            except: pass
        #indices = [ vocab[s] for s in sentence ]

        if pad > -1:
            if len(indices) > pad:
                indices = indices[:pad]
            else:
                padding = [ vocab['<p>'] for _ in xrange(len(indices), pad) ]
                indices.extend(padding)
         
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)

    def get_batch_from_pairs(self, pair_list):
        source_sentences = []
        target_sentences = []
        for s, t in pair_list:
            source_sentences.append(s)
            target_sentences.append(t)

        return np.array(source_sentences, dtype = 'int32'), np.array(target_sentences, dtype = 'int32'), pair_list


def main():
    # FOR TESTING ONLY
    trans_options = {
        'model_type' : 'translation',
        'source_file' : 'Data/translator_training_data/news-commentary-v9.fr-en.en',
        'target_file' : 'Data/translator_training_data/news-commentary-v9.fr-en.fr',
        'bucket_quant' : 25,
    }
    gen_options = {
        'model_type' : 'generator', 
        'dir_name' : 'Data',
    }

    dl = Data_Loader(trans_options)
    buckets, source_vocab, target_vocab = dl.load_translation_data()
    from IPython import embed; embed()

if __name__ == '__main__':
    main()
