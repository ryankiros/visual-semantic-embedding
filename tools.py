"""
A selection of functions for encoding images and sentences
"""
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

from collections import OrderedDict, defaultdict
from scipy.linalg import norm

from utils import load_params, init_tparams
from model import init_params, build_sentence_encoder, build_image_encoder

#-----------------------------------------------------------------------------#
# Specify model location here
#-----------------------------------------------------------------------------#
default_model = '/ais/gobi3/u/rkiros/uvsmodels/coco.npz'
#-----------------------------------------------------------------------------#

def load_model(path_to_model=default_model):
    """
    Load all model components
    """
    print path_to_model

    # Load the worddict
    print 'Loading dictionary...'
    with open('%s.dictionary.pkl'%path_to_model, 'rb') as f:
        worddict = pkl.load(f)

    # Create inverted dictionary
    print 'Creating inverted dictionary...'
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Load model options
    print 'Loading model options...'
    with open('%s.pkl'%path_to_model, 'rb') as f:
        options = pkl.load(f)

    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)
    params = load_params(path_to_model, params)
    tparams = init_tparams(params)

    # Extractor functions
    print 'Compiling sentence encoder...'
    trng = RandomStreams(1234)
    trng, [x, x_mask], sentences = build_sentence_encoder(tparams, options)
    f_senc = theano.function([x, x_mask], sentences, name='f_senc')

    print 'Compiling image encoder...'
    trng, [im], images = build_image_encoder(tparams, options)
    f_ienc = theano.function([im], images, name='f_ienc')

    # Store everything we need in a dictionary
    print 'Packing up...'
    model = {}
    model['options'] = options
    model['worddict'] = worddict
    model['word_idict'] = word_idict
    model['f_senc'] = f_senc
    model['f_ienc'] = f_ienc
    return model

def encode_sentences(model, X, verbose=False, batch_size=128):
    """
    Encode sentences into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i,s in enumerate(captions):
        ds[len(s)].append(i)

    # quick check if a word is in the dictionary
    d = defaultdict(lambda : 0)
    for w in model['worddict'].keys():
        d[w] = 1

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict'][w] if d[w] > 0 and model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            x_mask = numpy.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.
            
            ff = model['f_senc'](x, x_mask)
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features

def encode_images(model, IM):
    """
    Encode images into the joint embedding space
    """
    images = model['f_ienc'](IM)
    return images


