import sys
import argparse
from itertools import islice

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from corpus import TwitterCorpus, YelpCorpus, SentimentCorpus
from tokenizer import Tokenizer
from feature_extractor import BOWFeatureExtractor, EasyAdaptFeatureExtractor, \
        LexiconFeatureExtractor, InstanceWeightFeatureExtractor


def get_minibatch(datastream, batch_size):
    data = [sample for sample in islice(datastream, batch_size)]
    return data

def iter_minibatches(datastream, batch_size):
    data = get_minibatch(datastream, batch_size)
    while data:
        yield data
        data = get_minibatch(datastream, batch_size)

def show_most_informative_features(word2idx, clf, n=50, easyadapt=False):
    # adapted from http://stackoverflow.com/a/11140887/5818736
    feature_names = sorted(((k,v) for (k,v) in word2idx.items()),
                           key=lambda x: x[1])
    feature_names = [x[0] for x in feature_names]
    if easyadapt:
        vocab_size = len(feature_names)
        source_feature_names = ["%s_source" % f for f in feature_names]
        target_feature_names = ["%s_target" % f for f in feature_names]
        feature_names = feature_names + source_feature_names + \
            target_feature_names
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def test_lr(train=['yelp'], test=['twitter_test'], n_train=[100000, 10000],
            feature_extractor='bow', combine_train=True, combine_test=False):

    print [('train',train), ('test',test), ('n_train',n_train),
           ('feature_extractor', feature_extractor)]

    if len(train) == 1:
        n_train = n_train[0]
        if train[0] == 'yelp':
            train_corpus = YelpCorpus('../data/yelp', label_max=(n_train / 2))
        elif train[0] == 'twitter':
            train_corpus = TwitterCorpus('../data/sentiment140',
                                         label_max=(n_train / 2))
    elif len(train) == 2:
        yelp_corpus = YelpCorpus('../data/yelp', label_max=(n_train[0] / 2))
        twitter_corpus = TwitterCorpus('../data/sentiment140', train_only=True,
                                       label_max=(n_train[1] / 2))
        train_corpus = SentimentCorpus([yelp_corpus, twitter_corpus])

    if len(test) == 1:
        if test[0] == 'twitter_test':
            test_corpus = TwitterCorpus('../data/sentiment140', test_only=True)

    print "Fitting feature extractor..."
    # fit the feature extractor on the training data
    if feature_extractor == 'bow':
        fe = BOWFeatureExtractor()
    elif feature_extractor == 'easy':
        fe = EasyAdaptFeatureExtractor()
        print "source: %s, target: %s" % (fe.source_domain, fe.target_domain)
    elif feature_extractor == 'weighted':
        fe = InstanceWeightFeatureExtractor()
    fe = fe.fit(train_corpus.processed_samples())

    # create train and test data streams
    train_data_stream = train_corpus.processed_samples()
    test_data_stream = test_corpus.processed_samples()

    # create test set and exclude val set from training stream
    if feature_extractor == 'weighted':
        X_test, y_test, weights_test = fe.transform(test_data_stream)
        X_val, y_val, weights_val = fe.transform(
            get_minibatch(train_data_stream, 1000))
    else:
        X_test, y_test = fe.transform(test_data_stream)
        X_val, y_val = fe.transform(get_minibatch(train_data_stream, 1000))
    print "vocab size:", len(fe.vocab)+1
    print "feature space dim:", X_val.shape[1]

    print "Training classifier..."
    minibatch_size = 512
    clf = SGDClassifier(loss='log')
    classes = np.unique(y_test)
    for i, minibatch in enumerate(iter_minibatches(train_data_stream, 512)):
        if feature_extractor == 'weighted':
            X_train, y_train, weights_train = fe.transform(minibatch)
            clf.partial_fit(X_train, y_train, classes=classes,
                            sample_weight=weights_train)
        else:
            X_train, y_train = fe.transform(minibatch)
            clf.partial_fit(X_train, y_train, classes=classes)
        if i % 100  == 0 and i != 0:
            val_acc = clf.score(X_val, y_val)
            print "val acc after %i samples: %.2f" % \
                    (i * minibatch_size, val_acc)
    test_acc = clf.score(X_val, y_val)
    val_acc = clf.score(X_val, y_val)
    print "Test accuracy on %s: %.2f" % (test, clf.score(X_test, y_test))
    print "Val accuracy: %.2f" % (val_acc)
    easyadapt = True if feature_extractor == 'easy' else False
    show_most_informative_features(fe.word2idx, clf, n=20, easyadapt=easyadapt)
    return clf, fe

def test_lexicon(mode='vector'):
    test_corpus = TwitterCorpus('../data/sentiment140', test_only=True)
    if mode == 'score':
        fe = LexiconFeatureExtractor('../data/opinion-lexicon-English',
                                     mode=mode)
        X_test, y_test = fe.fit_transform(test_corpus.processed_samples())
        def decision_fn(x):
            if x >= 0:
                return fe.l_enc['pos']
            else:
                return fe.l_enc['neg']
        vdf = np.vectorize(decision_fn)
        pred = vdf(X_test)
        correct = np.where(pred == y_test)
        acc = correct[0].shape[0] / float(y_test.shape[0])
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='*', default=['yelp'])
    parser.add_argument('--test', nargs='*', default=['twitter_test'])
    parser.add_argument('-f', nargs=1, type=str, default='bow')
    parser.add_argument('-n', nargs='*', type=int, default=[100000, 10000])
    args = parser.parse_args()
    train = args.train
    test = args.test
    n = args.n
    f = args.f[0]
    test_lr(train, test, n, f)
