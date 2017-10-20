import re
import glob
import enchant
from gensim.utils import lemmatize
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words


class IterDocs(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for filename in glob.glob(self.fname):
            with open(filename, 'r') as ofile:
                content = ofile.read()
                yield process_doc(content)


class MyCorpus(object):
    def __init__(self, fname, dictionary):
        self.fname = fname
        self.dictionary = dictionary

    def __iter__(self):
        for filename in glob.glob(self.fname):
            with open(filename, 'r') as ofile:
                content = ofile.read()
                yield self.dictionary.doc2bow(content.lower().split())


# TODO: remove tickers from doc, disable enchant dict
# Process every single doc in iterator
def process_doc(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # lemmtizer
    # lmtzr = WordNetLemmatizer()

    # dict_en = enchant.Dict("en_US")
    # dict_en = enchant.DictWithPWL("en_US", "inv_token.txt")

    # ticker list
    # f = open('ticker.txt', 'r')
    # ticker = [line.strip().lower() for line in f.readlines()]
    # f.close()

    # investor dictionary
    f = open("inv_token.txt", "r")
    inv_dict = [line.strip().lower() for line in f.readlines()]
    f.close()

    raw = doc.decode('utf-8').lower()
    tokens = tokenizer.tokenize(raw)


    stopped_tokens = [token for token in tokens if token not in en_stop]
    # stemmed_tokens = [lmtzr.lemmatize(i) for i in stopped_tokens]
    stemmed_tokens = [word.split('/')[0]
                      for word in lemmatize(' '.join(stopped_tokens))]
    # stemmed_tokens = [word.split('/')[0]
    #                   for word in lemmatize(' '.join(stopped_tokens),
    #                                         allowed_tags=re.compile('(NN)'),
    #                                         min_length=3)]

    words_tokens = [word for word in stemmed_tokens if word in inv_dict]
    # words_tokens = [word for word in stemmed_tokens if word not in ticker]
    # words_tokens = [word for word in stemmed_tokens if dict_en.check(word)]
    return words_tokens
