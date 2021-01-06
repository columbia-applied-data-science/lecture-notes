import nltk
from os import listdir
from os.path import isfile, join
import sys


def main():
    basepath = '/home/langmore/jrl/enron/data/raw/enron-spam/all'
    allfiles = [f for f in listdir(basepath) if isfile(join(basepath, f))]

    # The part of speech that we will keep
    pos_type = 'NN'

    for filename in allfiles:
        result = process_file(pos_type, basepath, filename)
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


def is_stopword(string):
    return string.lower() in nltk.corpus.stopwords.words('english')


if __name__ == '__main__':
    main()
