import numpy as np
import glob
import itertools
from collections import Counter
from os.path import join as pathjoin
from os.path import basename
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from abc import ABCMeta, abstractmethod

"""
Classes for creating datasets to feed to sklearn classifiers.
"""

class BOWFeatureExtractor(object):
    """
    Featurize corpus samples into one-hot bag of words representations.
    The BOW arrays are stored in a scipy sparse matrix.
    """
    def __init__(self):
        self.vocab = set()
        self.word2idx = {}
        self.l_enc = {'pos':1, 'neg':0}

    def fit(self, datastream):
        """
        Populate word->index dictionary, vocabulary, and fit a label encoder.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        self
        """
        self.fit_transform(datastream)
        return self

    def fit_transform(self, datastream):
        """
        Creates a word to index mapping and a label
        to int label mapping and returns a dataset.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse.csr_matrix of shape [nb_samples, nb_features]
        y: array of shape [nb_samples,]
        """
        labels = []
        idx_counter = 0
        indices = []
        indptr = [0]
        values = []

        # create word2idx and vocab
        for sample, tokens in datastream:
            for token in tokens:
                self.vocab.add(token)
                if token not in self.word2idx:
                    self.word2idx[token] = idx_counter
                    idx_counter += 1
                idx = self.word2idx[token]
                indices.append(idx)
                values.append(1)
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')
        self.word2idx['unk'] = len(self.vocab)

        # create X matrix
        nb_samples = len(labels)
        nb_features = len(self.vocab) + 1  # unk
        shape = (nb_samples, nb_features)
        matrix = sparse.csr_matrix((values, indices, indptr),
                                   shape=shape, dtype='int32')

        # encode labels
        y = np.array([self.l_enc[l] for l in labels], dtype=np.int32)
        return matrix, y

    def transform(self, datastream):
        """
        Maps corpus samples to bag of words representations
        using `self.word2idx` and returns dataset.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse.csr_matrix of shape [nb_samples, nb_features]
        y: array of shape [nb_samples,]
        """
        labels = []
        sample_counter = 0
        indices = []
        indptr = [0]
        values = []

        # gather word indices
        for sample, tokens in datastream:
            sample_counter += 1
            for token in tokens:
                if token in self.vocab:
                    idx = self.word2idx[token]
                else:
                    idx = self.word2idx['unk']
                indices.append(idx)
                values.append(1)
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')

        # create X matrix
        nb_samples = sample_counter
        nb_features = len(self.vocab) + 1  # unk
        shape = (nb_samples, nb_features)
        matrix = sparse.csr_matrix((values, indices, indptr),
                                   shape=shape, dtype='int32')

        # encode labels
        y = np.array([self.l_enc[l] for l in labels], dtype=np.int32)

        y = y.astype(np.int32)
        return matrix, y

class EasyAdaptFeatureExtractor(BOWFeatureExtractor):
    """
    Implements the Easy Adapt feature augmentation method
    of (Daume III, 2009) [http://arxiv.org/pdf/0907.1815v1.pdf].

    Define to mapping functions, phi_s and phi_t that take as
    input feature vectors in R^F space extracted from the source
    data and target data respectively, and return augmented
    feature vectors in R^F3:
        phi_s(x) = [x, x, 0],   phi_t(x) = [x, 0, x],
            where x is the original feature vector and
            0 is the zeros vector in R^F.

    Parameters
    ----------
    source_domain: string identifying which dataset is being treated as the
        source domain
    target_domain: string identifying which dataset is being treated as the
        target domain
    """
    def __init__(self, source_domain='yelp', target_domain='twitter'):
        self.source_domain = source_domain
        self.target_domain = target_domain
        super(EasyAdaptFeatureExtractor, self).__init__()

    def fit(self, datastream):
        """
        Populate word->index dictionary, vocabulary, and fit a label encoder.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        self
        """
        super(EasyAdaptFeatureExtractor, self).fit_transform(datastream)
        return self

    def fit_transform(self, datastream):
        """
        Populate word2idx dict and vocab set and then
        Extract bag of words feature representations for each
        sample and apply the Easy Adapt technique on them.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse.csr_matrix of shape [nb_samples, new_nb_features],
            where new_nb_features is 3 * nb_features from the original feature
            space
        y: array of shape [nb_samples,]
        """
        # need to make the generator into a list so we can pass
        # over it multiple times
        self.fit(datastream)

        labels = []
        sample_counter = 0
        indices = []
        indptr = [0]
        values = []

        orig_feature_dim = len(self.vocab) + 1
        new_feature_dim = orig_feature_dim * 3
        for sample, tokens in datastream:
            sample_counter += 1
            if sample.dataset == self.source_domain: # [x] -> [x,x,0]
                copy_offset = orig_feature_dim
            elif sample.dataset == self.target_domain: # [x] -> [x,0,x]
                copy_offset = orig_feature_dim * 2
            # first pass: original feature vector
            for token in tokens:
                if token in self.vocab:
                    idx = self.word2idx[token]
                else:
                    idx = self.word2idx['unk']
                indices.append(idx)
                values.append(1)
            # second pass: copy
            for token in tokens:
                if token in self.vocab:
                    idx = self.word2idx[token] + copy_offset
                else:
                    idx = self.word2idx['unk'] + copy_offset
                indices.append(idx)
                values.append(1)
            # no third pass necessary, vector of zeros can be added
            # by changing final matrix shape
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')

        # create X matrix
        nb_samples = sample_counter
        nb_features = new_feature_dim
        shape = (nb_samples, nb_features)
        matrix = sparse.csr_matrix((values, indices, indptr),
                                   shape=shape, dtype='int32')

        # encode labels
        y = np.array([self.l_enc[l] for l in labels], dtype=np.int32)
        return matrix, y

    def transform(self, datastream):
        """
        Extract bag of words feature representations for each
        sample and apply the Easy Adapt technique on them.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse.csr_matrix of shape [nb_samples, new_nb_features],
            where new_nb_features is 3 * nb_features from the original feature
            space
        y: array of shape [nb_samples,]
        """
        labels = []
        sample_counter = 0
        indices = []
        indptr = [0]
        values = []

        orig_feature_dim = len(self.vocab) + 1
        new_feature_dim = orig_feature_dim * 3
        for sample, tokens in datastream:
            sample_counter += 1
            if sample.dataset == self.source_domain: # [x] -> [x,x,0]
                copy_offset = orig_feature_dim
            elif sample.dataset == self.target_domain: # [x] -> [x,0,x]
                copy_offset = orig_feature_dim * 2
            # first pass: original feature vector
            for token in tokens:
                if token in self.vocab:
                    idx = self.word2idx[token]
                else:
                    idx = self.word2idx['unk']
                indices.append(idx)
                values.append(1)
            # second pass: copy
            for token in tokens:
                if token in self.vocab:
                    idx = self.word2idx[token] + copy_offset
                else:
                    idx = self.word2idx['unk'] + copy_offset
                indices.append(idx)
                values.append(1)
            # no third pass necessary, vector of zeros can be added
            # by changing final matrix shape
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')

        # create X matrix
        nb_samples = sample_counter
        nb_features = new_feature_dim
        shape = (nb_samples, nb_features)
        matrix = sparse.csr_matrix((values, indices, indptr),
                                   shape=shape, dtype='int32')

        # encode labels
        y = np.array([self.l_enc[l] for l in labels], dtype=np.int32)
        return matrix, y

class InstanceWeightFeatureExtractor(BOWFeatureExtractor):
    """
    Extracts unigram features from samples and weights for each instance.
    The weights are penalties for how representative of the source samples
    are.

    Parameters
    ----------
    source_domain: the dataset string identifier for the source domain
    target_domain: the dataset string identifier for the target domain
    """

    def __init__(self, source_domain='yelp', target_domain='twitter'):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.pos_target_counter = Counter()
        self.neg_target_counter = Counter()
        self.pos_source_counter = Counter()
        self.neg_source_counter = Counter()
        self.target_lexicon = set()
        self.source_lexicon = set()
        super(InstanceWeightFeatureExtractor, self).__init__()

    def fit(self, datastream):
        """
        Create word2idx dictionary, vocab, and target and source lexicons.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        self
        """

        labels = []
        idx_counter = 0
        indices = []
        indptr = [0]
        values = []

        # create source and target lexicons, word2idx, and vocab
        for sample, tokens in datastream:
            for token in tokens:
                self.vocab.add(token)
                if token not in self.word2idx:
                    self.word2idx[token] = idx_counter
                    idx_counter += 1
                idx = self.word2idx[token]
                indices.append(idx)
                values.append(1)
                if sample.dataset == self.source_domain:
                    if sample.label == 'pos':
                        self.pos_source_counter[idx] += 1
                    else:
                        self.neg_source_counter[idx] += 1
                elif sample.dataset == self.target_domain:
                    if sample.label == 'pos':
                        self.pos_target_counter[idx] += 1
                    else:
                        self.neg_target_counter[idx] += 1
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')
        self.word2idx['unk'] = len(self.vocab)

        # create source and target lexicons
        self.create_lexicons()

        return self

    def fit_transform(self, datastream):
        """
        Create word2idx dictionary, vocab, and target and source lexicons
        and return dataset.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse matrix of size [nb_samples, nb_features]
        y: array of size [nb_samples,]
        instance_weights: array of shape [nb_samples,]
        """
        labels = []
        idx_counter = 0
        indices = []
        indptr = [0]
        values = []

        # create source and target lexicons, word2idx, and vocab
        for sample, tokens in datastream:
            for token in tokens:
                self.vocab.add(token)
                if token not in self.word2idx:
                    self.word2idx[token] = idx_counter
                    idx_counter += 1
                idx = self.word2idx[token]
                indices.append(idx)
                values.append(1)
                if sample.dataset == self.source_domain:
                    if sample.label == 'pos':
                        self.pos_source_counter[idx] += 1
                    else:
                        self.neg_source_counter[idx] += 1
                elif sample.dataset == self.target_domain:
                    if sample.label == 'pos':
                        self.pos_target_counter[idx] += 1
                    else:
                        self.neg_target_counter[idx] += 1
            indptr.append(len(indices))
            labels.append(sample.label)
        indices = np.array(indices, dtype='int32')
        indptr = np.array(indptr, dtype='int32')
        self.word2idx['unk'] = len(self.vocab)

        # create source and target lexicons
        self.create_lexicons()

        # create X matrix
        nb_samples = len(labels)
        nb_features = len(self.vocab) + 1  # unk
        shape = (nb_samples, nb_features)
        matrix = sparse.csr_matrix((values, indices, indptr),
                                   shape=shape, dtype='int32')

        # assign instance weights
        X = matrix.toarray()
        instance_weights = np.apply_along_axis(self.assign_weight,
                                               axis=1, arr=X)

        # encode labels
        y = np.array([self.l_enc[l] for l in labels], dtype=np.int32)
        return matrix, y, instance_weights

    def create_lexicons(self):
        most_pos_source = set([t[0] for t in
                           self.pos_source_counter.most_common(3000)])
        most_neg_source = set([t[0] for t in
                          self.neg_source_counter.most_common(3000)])
        most_pos_target = set([t[0] for t in
                           self.pos_target_counter.most_common(3000)])
        most_neg_target = set([t[0] for t in
                           self.neg_target_counter.most_common(3000)])
        source_lexicon = most_pos_source | most_neg_source
        target_lexicon = most_pos_target | most_neg_target
        source_only_lexicon = source_lexicon - target_lexicon
        target_only_lexicon = target_lexicon - source_lexicon
        assert len(source_only_lexicon.intersection(target_only_lexicon)) == 0
        self.source_lexicon = source_only_lexicon
        self.target_lexicon = target_only_lexicon

    def assign_weight(self, sample_vec):
        idxs = np.nonzero(sample_vec)[0]
        source_only = [idx for idx in idxs if idx in self.source_lexicon]
        target_only = [idx for idx in idxs if idx in self.target_lexicon]
        source_score = max(1, (len(source_only) - len(target_only)))
        w = 1. / source_score
        return w

    def transform(self, datastream):
        """
        Create dataset using lexicons, vocab, and word2idx.

        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        matrix: scipy.sparse matrix of size [nb_samples, nb_features]
        y: array of size [nb_samples,]
        instance_weights: array of shape [nb_samples,]
        """
        matrix, y = super(InstanceWeightFeatureExtractor,
                          self).transform(datastream)

        # assign instance weights
        X = matrix.toarray()
        instance_weights = np.apply_along_axis(self.assign_weight,
                                               axis=1, arr=X)

        return matrix, y, instance_weights


class LexiconFeatureExtractor(object):
    """
    Use a precompiled sentiment lexicon to featurize samples.
    Either computes a single score for each sample or a feature vector for
    each sample.

    Parameters:
    -----------
    lexicon_dir: directory containing the lexicon .txt files
    mode:
        `vector`: create a feature vector for each sample
            where each element of the vector is an indicator
            function for a word in the lexicon
        `score`: return the total number of words from the
            lexicon found in the sample
    """
    def __init__(self, lexicon_dir, mode='vector'):
        self.mode = mode
        self.lexicon_dir = lexicon_dir
        self.feature2idx = {}
        self.negfeatures = set()
        self.posfeatures = set()
        self.l_enc = {'pos':1, 'neg':0}

        for lexicon_file in glob.glob(pathjoin(lexicon_dir, "*.txt")):
            with open(lexicon_file) as f:
                polarity = basename(lexicon_file)
                for line in f:
                    idx_counter = 0
                    if not line.startswith(';'):
                        feature = line.strip()
                        if self.mode == 'vector':
                            self.feature2idx[feature] = idx_counter
                            idx_counter += 1
                        else:
                            if polarity == 'positive-words':
                                self.posfeatures.add(feature)
                            else:
                                self.negfeatures.add(feature)

    def fit(self, datastream):
        """
        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator
        """
        # everything is initialized in __init__
        return

    def fit_transform(self, datastream):
        """
        Create the X,y dataset for a datastream.


        Parameters
        ----------
        datasteam: iterable of data, i.e., a corpus's
            `processed_samples` generator

        Returns
        -------
        X: array of sentiment scores of shape [nb_samples,] (if mode is score)
            or array of feature vectors of shape [nb_sample, nb_features] (if
            mode is vector).
        y: array of shape [nb_samples,]
        """
        # control for negation (opposite sentiment)
        vectors = []
        scores = []
        labels = []
        for sample, tokens in datastream:
            score = 0
            vector = []
            for token in tokens:
                if self.mode == 'vector':
                    if token in self.feature2idx:
                        vector.append(self.feature2idx[token])
                elif self.mode == 'score':
                    if token.endswith('_NOT'):
                        token = token[:-4]
                        if token in self.posfeatures:
                            score -= 1
                        elif token in self.negfeatures:
                            score += 1
                    if token in self.posfeatures:
                        score += 1
                    elif token in self.negfeatures:
                        score -= 1
            if self.mode == 'vector':
                vectors.append(vector)
            elif self.mode == 'score':
                scores.append(score)
            labels.append(sample.label)
        if self.mode == 'vector':
            X = np.array(vectors)
        elif self.mode == 'score':
            X = np.array(scores)
        y = np.array([self.l_enc[label] for label in labels])
        return X, y

    def transform(self, datastream, mode='vector'):
        """
        Create the X,y dataset for a datastream.

        See documentation for `fit_transform` method.
        """
        return self.fit_transform(datastream)

