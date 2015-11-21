# visual-semantic-embedding

Code for the image-sentence ranking methods from "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models" (Kiros, Salakhutdinov, Zemel. 2014).

Images and sentences are mapped into a common vector space, where the sentence representation is computed using LSTM. This project contains training code and pre-trained models for Flickr8K, Flickr30K and MS COCO.

New (Oct 19, 2015): Ability to embed and caption your own images.

If you're interested in generating image captions instead, see our follow up project [arctic-captions](https://github.com/kelvinxu/arctic-captions).

## Visualization

Here are [results](http://www.cs.toronto.edu/~rkiros/vse_coco_dev.html) on 1000 images from the MS COCO development set, using the pre-trained model available for download. For each image, we retrieve the highest scoring caption from the training set.

See below for details on how to use your own images.

## Results

Below is a table of results obtained using the code from this repository, comparing the numbers reported in our paper. aR@K is the Recall@K for image annotation (higher is better), while sR@K is the Recall@K for image search (higher is better). Medr is the median rank of the closest ground truth (lower is better). NOTE: these results use features from 1 image crop. Some papers report results using average features from 10 crops, which gives better results. 

**Flickr8K**

| Method | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| reported | 18.0 | 40.9 | 55.0 | 8 | 12.5 | 37.0 | 51.5 | 10 |
| this project | 22.3 | 48.7 | 59.8 | 6 | 14.9 | 38.3 | 51.6 | 10

**Flickr30K**

| Method | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| reported | 23.0 | 50.7 | 62.9 | 5 | 16.8 | 42.0 | 56.5 | 8
| this project | 29.8 | 58.4 | 70.5 | 4 | 22.0 | 47.9 | 59.3 | 6

**MS COCO**

| Method | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| this project | 43.4 | 75.7 | 85.8 | 2 | 31.0 | 66.7 | 79.9 | 3 

For a complete list of results on these tasks, see [this paper](http://arxiv.org/abs/1504.06063) by Lin Ma et al (ICCV 2015) which contains the the most up-to-date tables (as of September 2015).

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)

If you want to caption your own images, you will also need:

* [Lasagne](https://github.com/Lasagne/Lasagne)
* A version of Theano that Lasagne supports

## Getting started

You will first need to download the dataset files and pre-trained models. These can be obtained by running

    wget http://www.cs.toronto.edu/~rkiros/datasets/f8k.zip
    wget http://www.cs.toronto.edu/~rkiros/datasets/f30k.zip
    wget http://www.cs.toronto.edu/~rkiros/datasets/coco.zip
    wget http://www.cs.toronto.edu/~rkiros/models/vse.zip

Each of the dataset files contains the captions as well as [VGG features](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) from the 19-layer model. Flickr8K comes with a pre-defined train/dev/test split, while for Flickr30K and MS COCO we use the splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). Note that the original images are not included with the dataset. The full contents of each of the datasets can be obtained [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

Once the datasets are downloaded, open `datasets.py` and set the directory to where the datasets are.

**NOTE to Toronto users**: the unzipped files are available in my gobi3 directory under uvsdata and uvsmodels. Just link there instead of downloading.

## Evaluating pre-trained models

Lets use Flickr8K as an example. To reproduce the numbers in the table above, open `tools.py` and specify the path to the downloaded Flickr8K model. Then in IPython run the following: 

    import tools, evaluation
    model = tools.load_model()
    evaluation.evalrank(model, data='f8k', split='test')

This will evaluate the loaded model on the Flickr8K test set. You can also replace 'test' with 'dev' to evaluate on the development set. Alternatively, evaluate the Flickr30K and MS COCO models instead.

## Computing image and sentence vectors

Suppose you have a list of strings that you would like to embed into the learned vector space. To embed them, run the following:

    sentence_vectors = tools.encode_sentences(model, X, verbose=True)
    
Where 'X' is the list of strings. Note that the strings should already be pre-tokenized, so that split() returns the tokens.

As the vectors are being computed, it will print some numbers. The code works by extracting vectors in batches of sentences that have the same length - so the number corresponds to the current length being processed. If you want to turn this off, set verbose=False when calling encode.

To encode images, run the following instead:

    image_vectors = tools.encode_images(model, IM)
    
Where 'IM' is a NumPy array of VGG features. Note that the VGG features were scaled to unit norm prior to training the models.

## Training new models

Open `train.py` and specify the hyperparameters that you would like. Below we describe each of them in detail:

* data: The dataset to train on (f8k, f30k or coco).
* margin: The margin used for computing the pairwise ranking loss. Should be between 0 and 1.
* dim: The dimensionality of the learned embedding space (also the size of the RNN state).
* dim_image: The dimensionality of the image features. This will be 4096 for VGG.
* dim_word: The dimensionality of the learned word embeddings.
* ncon: The number of contrastive (negative) examples for computing the loss.
* encoder: Either 'gru' or 'bow'
* max_epochs: The number of epochs used for training.
* dispFreq: How often to display training progress.
* decay_c: The weight decay hyperparameter.
* grad_clip: When to clip the gradient.
* maxlen_w: Sentences longer then this value will be ignored.
* optimizer: The optimization method to use. Only supports 'adam' at the moment.
* batch_size: The size of a minibatch.
* saveto: The location to save the model.
* validFreq: How often to evaluate on the development set.
* reload_: Whether to reload a previously trained model.

Note that if encoder is 'bow', then dim and dim_word need to be the same dimension. Once you are happy, just run the following:

    import train
    train.trainer()
    
As the model trains, it will periodically evaluate on the development set (validFreq) and re-save the model each time performance on the development set increases. Generally you shouldn't need more than 15-20 epochs of training on any of the datasets. Once the models are saved, you can load and evaluate them in the same way as the pre-trained models.

## Using your own images

The script `demo.py` contains code for embedding and captioning your own images. First you need to download the model parameters for the VGG-19 ConvNet. You can download them by running:

    wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
    
Note that this model is for non-commercial use only. Next, open `demo.py` and set the location of where you saved the parameters. In `model.py` specify the location of the pre-trained MS COCO model. Then run the following:

    import demo, tools, datasets
    net = demo.build_convnet()
    model = tools.load_model()
    train = datasets.load_dataset('coco', load_train=True)[0]
    vectors = tools.encode_sentences(model, train[0], verbose=False)
    demo.retrieve_captions(model, net, train[0], vectors, 'image.jpg', k=5)
    
where image.jpg is some image. The above code will initialize the VGG ConvNet, load the pre-trained embedding model, load the MS COCO training set and then embed the training captions. For a new image, the last line will score the image embedding with each MS COCO training caption and retrieve the top-5 nearest captions. For example, with the first image [here](http://www.cs.toronto.edu/~rkiros/vse_coco_dev.html) I get the following output:

    ['The salad has many different types of vegetables in it .',
    'A salad is concocted with broccoli , potatoes and tomatoes .',
    'A table filled with greens and lots of vegetables .',
    'Pasta salad with tomatoes , broccoli , beans and sausage in a bowl..',
    'There is a lot if veggies that are in the tray']
    
Note that since this model was trained on MS COCO, if you use images that are much different than the training data you might get some funny results. If you want to do the reverse task (retrieve images for captions) it should be very straightforward to modify the code to do so.

## Using different datasets and features

If you want to use a different dataset, or use different image features, you will have to edit the paths in `datasets.py`. Each of (training/dev/test) contains 2 files: a .txt file of captions (one per line) and a .npy file containing a NumPy array of image features, where each row is the image features for the corresponding caption. If you put your dataset in the same format, then it can be used for training new models.

## Reference

If you found this code useful, please cite the following paper:

Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel. **"Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models."** *arXiv preprint arXiv:1411.2539 (2014).*

    @article{kiros2014unifying,
      title={Unifying visual-semantic embeddings with multimodal neural language models},
      author={Kiros, Ryan and Salakhutdinov, Ruslan and Zemel, Richard S},
      journal={arXiv preprint arXiv:1411.2539},
      year={2014}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)




