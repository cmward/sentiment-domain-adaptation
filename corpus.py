import csv
import glob
import re
import gzip
from copy import copy
from itertools import chain
from collections import Counter
from abc import ABCMeta, abstractmethod
from random import shuffle, seed

from os.path import dirname, basename, splitext, abspath
from os.path import join as pathjoin
from os.path import split as pathsplit

import pandas as pd
import numpy as np

from tokenizer import Tokenizer

"""
Classes for loading and sentiment analysis data from
various datasets.
"""

TWITTER_TRAIN_CSV = 'training.1600000.processed.noemoticon_shuffled.csv'
TWITTER_TEST_CSV = 'testdata.manual.2009.06.14.csv'
YELP_CSV = 'yelp_review_pruned_balanced_shuffled.csv'
AMAZON_CLOTHING_GZ = 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
AMAZON_HOME_GZ = 'reviews_Home_and_Kitchen_5.json.gz'

class Sample(object):
    """
    Class for storing a data sample.

    Parameters
    ----------
    data: the data sample's text
    label: label, one of ['pos', 'neut', 'neg']
    text: original sample text
    dataset: the dataset from which the sample was taken,
        one of ['imdb', 'yelp', 'twitter', 'amazon_clothes',
        'amazon_home']
    split: which split from the original dataset the
        sample was taken from, one of ['test', 'train', 'NA']
    """
    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label, dataset, split='NA'):
        self.data = data
        self.label = label
        self.dataset = dataset
        self.split = split

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def process(self, tokenizer):
        """Tokenize and normalize the sample's text,
        returning a list of tokens.

        Lowercases all words that aren't all uppercase.
        Attaches _NOT to any words in the scope of negation.

        Parameters
        ----------
        tokenizer: used to split the sample's text. Accessed from
            a corpus's class attribute `corpus.tokenizer`.

        Returns
        -------
        tokens: list of processed tokens
        """
        text = copy(self.data)
        tokens = tokenizer.tokenize(text)
        tokens = [t.lower() if not t.isupper() else t for t in tokens]

        # add _NOT to any words occurring in the scope of negation
        negations = ("never no nothing nowhere noone none not haven't \
                     haven't hasn't hadn't can't couldn't shouldn't \
                     won't wouldn't don't doesn't didn't isn't aren't \
                     aint").split()
        clause_punctutation = ['.',':',';','!','?']
        for i, token in enumerate(tokens):
            if token in negations:
                # don't add negation marker after 'not only'
                try:
                    if token == 'not' and tokens[i+1] == 'only':
                        continue
                except IndexError:
                    continue
                for j, inscope_token in enumerate(tokens[i+1:]):
                    if inscope_token in clause_punctutation:
                        break
                    else:
                        # add negation marker
                        tokens[i+j+1] = "%s_NOT" % tokens[i+j+1]
        return tokens

class CorpusStream(object):
    """
    Collection of labeled texts. Instead of storing all data samples
    in a field, the class implements a `samples` generator which
    streams samples from an original source data file on disk.

    Parameters
    ----------
    datafiles: list of file names from which data is read
    sample_class: the class used to construct sample objects
    """

    __metaclass__ = ABCMeta

    tokenizer = Tokenizer()

    def __init__(self, datafiles, sample_class=Sample):
        self.datafiles = datafiles
        self.sample_class = sample_class

    @abstractmethod
    def samples(self):
        """Generator that streams the samples from the corpus."""
        pass

    def processed_samples(self):
        """Generator that streams processed samples"""
        for sample in self.samples():
            yield (sample, sample.process(CorpusStream.tokenizer))

class SentimentCorpus(object):
    def __init__(self, sample_gens):
        self.sample_gens = sample_gens

    def samples(self):
        seed(hash('sentiment'))
        all_samples = [s for sample_gen in self.sample_gens
                       for s in list(sample_gen.samples())]
        shuffle(all_samples)
        for sample in all_samples:
            yield sample

    def processed_samples(self):
        """Generator that streams processed samples"""
        for sample in self.samples():
            yield (sample, sample.process(CorpusStream.tokenizer))

class IMDBCorpus(CorpusStream):
    """
    Corpus of short form movie reviews from IMDB. Each sample
    is approximately a paragraph long and labeled as `pos` or
    `neg`.

    Parameters
    ----------
    imdb_data_dir: top-level directory containing data files
    """
    def __init__(self, imdb_data_dir, sample_class=Sample):
        self.dataset = 'imdb'
        pos_samples = glob.glob(pathjoin(imdb_data_dir,'*/pos/*.txt'))
        neg_samples = glob.glob(pathjoin(imdb_data_dir,'*/neg/*.txt'))
        self.datafiles = pos_samples + neg_samples
        super(IMDBCorpus, self).__init__(
            self.datafiles, sample_class=sample_class)

    def samples(self):
        """
        Yield labeled samples from txt files.
        Labels are the parent directory of the file.
        """
        # this works in this case because we have one sample
        # per file
        n = 0
        n_samples = len(self.datafiles)
        for datafile in self.datafiles:
            label = pathsplit(dirname(datafile))[1]
            split = pathsplit(datafile)[0].split('/')[-2]
            with open(datafile, 'r') as f:
                sample = f.read()
                yield self.sample_class(sample, label, self.dataset, split)

class CSVCorpusStream(CorpusStream):
    """
    Corpus of samples read from CSV file(s).

    Parameters
    ----------
    datafiles: list of csv files from which the data is read
    lookup: dictionary used to map class labels in the data source
        to the target labels {'pos','neg','neut'}
    label_max: the maximum number of samples from each class to stream
    """

    def __init__(self, datafiles, sample_class,
                 lookup={}, label_max=50000):
        self.lookup = lookup
        self.label_max = label_max
        super(CSVCorpusStream, self).__init__(datafiles, sample_class=sample_class)

    def shuffle_and_write_csv(self):
        """
        Read the csv datafiles into dataframes, shuffle them,
        and write the resulting dataframes to new csv files.
        """
        for datafile in self.datafiles:
            df = pd.read_csv(datafile)
            df = df.sample(frac=1, random_state=1234)
            outfile = "%s_shuffled.csv" % splitext(basename(datafile))[0]
            outfile_path = pathjoin(dirname(abspath(datafile)), outfile)
            df.to_csv(outfile_path, index=False)

    @abstractmethod
    def data_from_row(self, csvrow):
        """
        What fields of the csv file rows should be added into samples.

        Parameters
        ----------
        csvrow: list representing a row of a csv file

        Returns
        -------
        (label, data): tuple of class label and text data, both strings

        """
        return NotImplemented

    def samples(self):
        """
        Generator that returns the first `self.label_max` instances
        for each label.
        """
        badrows = 0
        self.labelcounts = Counter()
        self.labelcounts['pos'], self.labelcounts['neg'] = 0, 0
        for datafile in self.datafiles:
            csvfile = open(datafile, 'rU')
            with open(datafile, 'rU') as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # skip header
                try:
                    for row in reader:
                        if all(n == self.label_max for n in
                               self.labelcounts.values()):
                            raise StopIteration
                        label, data = self.data_from_row(row)
                        if label in self.lookup:
                            label = self.lookup[label]
                            if self.labelcounts[label] < self.label_max:
                                self.labelcounts[label] += 1
                                yield self.sample_class(
                                    data, label, self.dataset, split='NA')
                except UnicodeDecodeError:
                    badrows += 1
            print 'bad rows:',badrows

class YelpCorpus(CSVCorpusStream):
    """
    Corpus of reviews from Yelp dataset. Scores are mapped
    as follows:
        1 stars: neg
        5 stars: pos

    Assumes that the original CSV data file has been pruned to
    contain only reviews with 1 or 5 star scores.

    See https://www.yelp.com/dataset_challenge for further description.
    """
    def __init__(self, yelp_data_dir, sample_class=Sample,
                 lookup={'neg':'neg','pos':'pos'}, label_max=260000):
        self.dataset = 'yelp'
        datafiles = [pathjoin(yelp_data_dir, YELP_CSV)]
        super(YelpCorpus, self).__init__(datafiles, sample_class=Sample,
                                        lookup=lookup, label_max=label_max)

    def prune_csv(self, outfile):
        csvfile = '../data/yelp/yelp_academic_dataset_review.csv'
        df = pd.read_csv(csvfile, usecols=['stars', 'text'],
                         dtype={'stars':np.int32, 'text': str})
        df = df[df['stars'].isin([1,5])]
        df.to_csv(outfile, columns=['stars', 'text'], index=False, na_rep='NA')

    def data_from_row(self, row):
        """
        csv file has already been pruned, so each row
        is just [label, data]
        """
        if len(row) != 2: # bad data
            return None, None
        return tuple(row)

    def samples(self):
        return super(YelpCorpus, self).samples()

class TwitterCorpus(CSVCorpusStream):
    """
    Corpus of tweets from sentiment140 dataset. Labels are converted
    as follows:
        0: neg
        4: pos

    The original test data contains neutral tweets, but these are ignored
    since the training data is comprised solely of positive and negative
    tweets.

    Original corpus is a single CSV file.
    See http://help.sentiment140.com/for-students for further description.
    """
    def __init__(self, twitter_data_dir, sample_class=Sample,
                 lookup={'0':'neg','4':'pos'}, label_max=50000,
                 train_only=False, test_only=False):
        self.dataset = 'twitter'
        train_csvfile = pathjoin(twitter_data_dir, TWITTER_TRAIN_CSV)
        test_csvfile = pathjoin(twitter_data_dir, TWITTER_TEST_CSV)
        if test_only:
            self.test_only = True
            datafiles = [test_csvfile]
        elif train_only:
            self.train_only = True
            datafiles = [train_csvfile]
        else:
            datafiles = [train_csvfile, test_csvfile]
        super(TwitterCorpus, self).__init__(
            datafiles, sample_class=sample_class,
            lookup=lookup, label_max=label_max
        )

    def data_from_row(self, row):
        return row[0], row[-1]

    def samples(self):
        return super(TwitterCorpus, self).samples()

class AmazonCorpus(CorpusStream):
    """
    Corpus of reviews from Amazon.
    Scores are mapped as follows:
        5.0: pos
        1.0: neg

    Original file is a single gzipped JSON file.
    See http://jmcauley.ucsd.edu/data/amazon/ for further description.
    """
    def __init__(self, amazon_data_dir, domain='clothing', sample_class=Sample):
        if domain == 'clothing':
            self.dataset = 'amazon_clothing'
            datafiles = [pathjoin(amazon_data_dir, AMAZON_CLOTHING_GZ)]
        elif domain == 'home':
            self.dataset = 'amazon_home'
            datafiles = [pathjoin(amazon_data_dir, AMAZON_CLOTHING_GZ)]
            datafiles = [pathjoin(amazon_data_dir, AMAZON_HOME_GZ)]
        super(AmazonCorpus, self).__init__(datafiles, sample_class=Sample)

    def samples(self):
        for datafile in self.datafiles:
            lookup = {5.0: 'pos', 1.0: 'neg'}
            def parse(d):
                g = gzip.open(d)
                for l in g:
                    yield eval(l)
            for datadict in parse(datafile):
                rating = datadict['overall']
                if rating in lookup:
                    label = lookup[rating]
                    sample = datadict['reviewText']
                    yield self.sample_class(sample, label, self.dataset, split='NA')
