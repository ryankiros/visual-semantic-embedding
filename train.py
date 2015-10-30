"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model, build_sentence_encoder, build_image_encoder
from vocab import build_dictionary
from evaluation import i2t, t2i
from tools import encode_sentences, encode_images
from datasets import load_dataset

# main trainer
def trainer(data='coco',  #f8k, f30k, coco
            margin=0.2,
            dim=1024,
            dim_image=4096,
            dim_word=300,
            encoder='gru',  # gru OR bow
            max_epochs=15,
            dispFreq=10,
            decay_c=0.,
            grad_clip=2.,
            maxlen_w=100,
            optimizer='adam',
            batch_size = 128,
            saveto='/ais/gobi3/u/rkiros/uvsmodels/coco.npz',
            validFreq=100,
            lrate=0.0002,
            reload_=False):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['encoder'] = encoder
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_

    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # Load training and development sets
    print 'Loading dataset'
    train, dev = load_dataset(data)[:2]

    # Create and save dictionary
    print 'Creating dictionary'
    worddict = build_dictionary(train[0]+dev[0])[0]
    n_words = len(worddict)
    model_options['n_words'] = n_words
    print 'Dictionary size: ' + str(n_words)
    with open('%s.dictionary.pkl'%saveto, 'wb') as f:
        pkl.dump(worddict, f)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, inps, cost = build_model(tparams, model_options)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building sentence encoder'
    trng, inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)

    print 'Building image encoder'
    trng, inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)

    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    train_iter = homogeneous_data.HomogeneousData([train[0], train[1]], batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    curr = 0.
    n_samples = 0
    
    for eidx in xrange(max_epochs):

        print 'Epoch ', eidx

        for x, im in train_iter:
            n_samples += len(x)
            uidx += 1

            x, mask, im = homogeneous_data.prepare_data(x, im, worddict, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            # Update
            ud_start = time.time()
            cost = f_grad_shared(x, mask, im)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                curr_model['word_idict'] = word_idict
                curr_model['f_senc'] = f_senc
                curr_model['f_ienc'] = f_ienc

                ls = encode_sentences(curr_model, dev[0])
                lim = encode_images(curr_model, dev[1])

                (r1, r5, r10, medr) = i2t(lim, ls)
                print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                (r1i, r5i, r10i, medri) = t2i(lim, ls)
                print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

                currscore = r1 + r5 + r10 + r1i + r5i + r10i
                if currscore > curr:
                    curr = currscore

                    # Save model
                    print 'Saving...',
                    params = unzip(tparams)
                    numpy.savez(saveto, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass

