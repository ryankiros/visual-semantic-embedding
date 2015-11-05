"""
Embedding and captioning new images
"""
import os
import cPickle as pkl
import numpy
import skimage.transform

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

from scipy.linalg import norm
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tools

#-----------------------------------------------------------------------------#
# Specify VGG-19 convnet location here
#-----------------------------------------------------------------------------#
path_to_vgg = '/ais/gobi3/u/rkiros/vgg/vgg19.pkl'
#-----------------------------------------------------------------------------#

def retrieve_captions(model, net, captions, cvecs, file_name, k=5):
    """
    Retrieve captions for a given image

    model: Image-sentence embedding model
    net: VGG ConvNet
    captions: list of sentences to search over
    cvecs: the embeddings for the above sentences
    file_name: location of the image
    k: number of sentences to return
    """
    # Load the image
    im = load_image(file_name)

    # Run image through convnet
    feats = compute_features(net, im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = tools.encode_images(model, feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, cvecs.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [captions[a] for a in sorted_args[:k]]

    return sentences

def regularities(model, net, captions, imvecs, file_name, negword, posword, k=5, rerank=False):
    """
    This is an example of how the 'Multimodal Lingustic Regularities' was done.
    Returns nearest neighbours to 'image - negword + posword'

    model: the embedding model, with encoder='bow'
    net: VGG ConvNet
    captions: a list of sentences
    imvecs: the corresponding image embeddings to each sentence in 'captions'
    file_name: location of the query image
    negword: the word to subtract
    posword: the word to add
    k: number of results to return
    rerank: whether to rerank results based on their mean (to push down outliers)

    'captions' is used only as a reference, to avoid loading/displaying images.

    Returns:
    The top k closest sentences in captions
    The indices of the top k captions

    Note that in our paper we used the SBU dataset (not COCO)
    """
    # Load the image
    im = load_image(file_name)

    # Run image through convnet
    query = compute_features(net, im).flatten()
    query /= norm(query)

    # Embed words
    pos = tools.encode_sentences(model, [posword], verbose=False)
    neg = tools.encode_sentences(model, [negword], verbose=False)

    # Embed image
    query = tools.encode_images(model, query[None,:])

    # Transform
    feats = query - neg + pos
    feats /= norm(feats)

    # Compute nearest neighbours
    scores = numpy.dot(feats, imvecs.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [captions[a] for a in sorted_args[:k]]

    # Re-rank based on the mean of the returned results
    if rerank:
        nearest = imvecs[sorted_args[:k]]
        meanvec = numpy.mean(nearest, 0)[None,:]
        scores = numpy.dot(nearest, meanvec.T).flatten()
        sargs = numpy.argsort(scores)[::-1]
        sentences = [sentences[a] for a in sargs[:k]]
        sorted_args = [sorted_args[a] for a in sargs[:k]]

    return sentences, sorted_args[:k]    

def compute_fromfile(net, loc, base_path='/ais/gobi3/u/rkiros/coco/images/val2014/'):
    """
    Compute image features from a text file of locations
    """
    batchsize = 128
    imagelist = []
    with open(loc, 'rb') as f:
        for line in f:
            imagelist.append(base_path + line.strip())
    inds = numpy.arange(len(imagelist))
    numbatches = len(inds) / batchsize + 1
    feats = numpy.zeros((len(imagelist), 4096), dtype='float32')

    for minibatch in range(numbatches):
        print minibatch * batchsize
        idx = inds[minibatch::numbatches]
        batch = [imagelist[i] for i in idx]
        ims = numpy.zeros((len(idx), 3, 224, 224), dtype='float32')
        for j in range(len(idx)):
            ims[j] = load_image(batch[j])
        fc7 = compute_features(net, ims)
        feats[idx] = fc7
         
    return feats
    
def load_image(file_name):
    """
    Load and preprocess an image
    """
    MEAN_VALUE = numpy.array([103.939, 116.779, 123.68]).reshape((3,1,1))
    image = Image.open(file_name)
    im = numpy.array(image)

    # Resize so smallest dim = 256, preserving aspect ratio
    if len(im.shape) == 2:
        im = im[:, :, numpy.newaxis]
        im = numpy.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = numpy.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = numpy.swapaxes(numpy.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUE
    return floatX(im[numpy.newaxis])

def compute_features(net, im):
    """
    Compute fc7 features for im
    """
    fc7 = numpy.array(lasagne.layers.get_output(net['fc7'], im, deterministic=True).eval())
    return fc7

def build_convnet():
    """
    Construct VGG-19 convnet
    """
    print 'Building model...'
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    print 'Loading parameters...'
    output_layer = net['prob']
    model = pkl.load(open(path_to_vgg))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net

