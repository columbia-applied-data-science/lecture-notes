import nltk
from os import listdir
from os.path import isfile, join
import sys

import itertools
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import IMapUnorderedIterator, IMapIterator



def main():

    basepath = '/home/langmore/jrl/enron/data/raw/enron-spam/all'
    allfiles = [f for f in listdir(basepath) if isfile(join(basepath, f))][:1000]

    # Number of slave processes to start
    n_procs = 4

    # The size chunk to send between slave and master
    chunksize = 1

    # The part of speech type that we will keep
    pos_type = 'NN'

    # Construct a function of one variable by fixing all but the last argument
    # f(filename) = process_file(..., filename)
    f = partial(process_file, pos_type, basepath)

    # Construct an iterator that is equivalent to
    # (f(filename) for filename in allfiles)

    # If we are using 1 processor, just use the normal itertools.imap function
    # Otherwise, use the worker_pool
    if n_procs == 1:
        results_iter = itertools.imap(f, allfiles)
    else:
        worker_pool = Pool(n_procs)
        results_iter = worker_pool.imap_unordered(f, allfiles, chunksize=chunksize)

    for result in results_iter:
        sys.stdout.write(result + '\n')


def process_file(pos_type, basepath, filename):
    """
    Read one file at a time, extract non stop words that whose part of speech
    is pos_type, return a count.

    Parameters
    ----------
    pos_type : String
        Some nltk part of speech type, e.g. 'NN'
    basepath : String
        Path to the base directory holding files
    filename : String
        Name of the file

    Returns
    -------
    counts : String
        word1:n1 word2:n2 word3:n3
    """
    path = join(basepath, filename)

    with open(path, 'r') as f:
        text = f.read()
        tokens = nltk.tokenize.word_tokenize(text)
        good_words = [t for t in tokens if t.isalpha() and not is_stopword(t)]
        word_pos_tuples = nltk.pos_tag(good_words)
        typed = [wt[0] for wt in word_pos_tuples if wt[1] == pos_type]
        freq_dist = nltk.FreqDist(typed)

        # Format the output string
        outstr = filename + '| '
        for word, count in freq_dist.iteritems():
            outstr += word + ':' + str(count) + ' '

        return outstr


def imap_wrap(func):
    """
    Wrapper for Pool.imap_unordered that allows exit upon Ctrl-C.  This is a fix
    of the known python bug  bugs.python.org/issue8296 given by 
    https://gist.github.com/aljungberg/626518
    """
    def wrap(self, timeout=None):
        return func(self, timeout=timeout if timeout is not None else 1e100)
    return wrap

# Redefine IMapUnorderedIterator so we can exit with Ctrl-C
IMapUnorderedIterator.next = imap_wrap(IMapUnorderedIterator.next)
IMapIterator.next = imap_wrap(IMapIterator.next)


def is_stopword(string):
    return string.lower() in nltk.corpus.stopwords.words('english')


if __name__ == '__main__':
    main()
