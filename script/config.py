import json
from os.path import join as opj

tr_data_root = '../code/models/bytenet/Data/translator_training_data'
tr_pretrain_root = '../code/models/bytenet/pretrained_models'

cl_data_root = '../code/models/vdcnn/data'
cl_pretrain_root = '../code/models/vdcnn/pretrained_models'

config = {
    'en-de-news':{
            'src': opj(tr_data_root, 'news-commentary-v12.de-en.en.train'),
            'tgt': opj(tr_data_root, 'news-commentary-v12.de-en.de.train'),
            'pretrain': opj(tr_pretrain_root, 'model_epoch_4_145000.ckpt')
        },

    'en-fr-news':{
            'src': opj(tr_data_root, 'news-commentary-v9.fr-en.en.train'),
            'tgt': opj(tr_data_root, 'news-commentary-v9.fr-en.fr.train'),
            'pretrain': opj(tr_pretrain_root, 'model_epoch_4_90000.ckpt')
        },

    'en-cs-news':{ 
            'src': opj(tr_data_root, 'news-commentary-v9.cs-en.en.train'),
            'tgt': opj(tr_data_root, 'news-commentary-v9.cs-en.cs.train'),
            'pretrain': opj(tr_pretrain_root, 'model_epoch_4_70000.ckpt'),
        },

    'en-de-europarl':{
            'src': opj(tr_data_root, 'europarl-v7.de-en.en.train'),
            'tgt': opj(tr_data_root, 'europarl-v7.de-en.de.train'),
            'pretrain': opj(tr_pretrain_root, 'model_epoch_1_440000.ckpt')
        },

    
    'ag':{
        'src': opj(cl_data_root, 'ag_news_csv'),
        'pretrain': opj(cl_pretrain_root, 'ag', 'model-step92000.ckpt')
        },

    'yelp':{
        'src': opj(cl_data_root, 'yelp_review_polarity_csv'),
        'pretrain': opj(cl_pretrain_root, 'yelp', 'model-step282000.ckpt')
        },

    'dbpedia':{
        'src': opj(cl_data_root, 'dbpedia_csv'),
        'pretrain': opj(cl_pretrain_root, 'dbpedia', 'model-step40000.ckpt')
        },
}

with open('config.js', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)
