"""
Model specification
"""
import theano
import theano.tensor as tensor
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Sentence encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Image encoder
    params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])

    return params

def contrastive_loss(margin, im, s, cim, cs):
    """
    Compute contrastive loss
    """
    cost_im = margin - (im * s).sum(axis=1) + (im * cs).sum(axis=1)
    cost_im = cost_im * (cost_im > 0.)
    cost_im = cost_im.sum(0)

    cost_s = margin - (s * im).sum(axis=1) + (s * cim).sum(axis=1)
    cost_s = cost_s * (cost_s > 0.)
    cost_s = cost_s.sum(0)

    cost = cost_im + cost_s
    return cost

def _step(cidx, totalcost, s, im, margin):
    """
    Step function for iterating over contrastive terms
    """
    cs = s[cidx]
    cim = im[cidx]
    cost = contrastive_loss(margin, im, s, cim, cs)
    return totalcost + cost

def build_model(tparams, options):                                                                                           
    """
    Computation graph for the model
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')
    con = tensor.matrix('con', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    sents = proj[0][-1]
    sents = l2norm(sents)

    # Encode images (source)
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')

    # Compute loss
    cost, updates = theano.scan(_step,
                                sequences=con,
                                outputs_info=tensor.alloc(0.),
                                non_sequences = [sents, images, options['margin']],
                                n_steps=con.shape[0],
                                profile=False,
                                strict=True)
    cost = cost[-1]
                               
    return trng, [x, mask, im, con], cost

def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    sents = proj[0][-1]
    sents = l2norm(sents)

    return trng, [x, mask], sents

def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # image features
    im = tensor.matrix('im', dtype='float32')

    # Encode images
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')
    images = l2norm(images)
    
    return trng, [im], images


