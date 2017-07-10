import os
import json
import glob
import gensim
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import enchant
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer


def process_doc(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # lemmtizer
    lmtzr = WordNetLemmatizer()

    raw = doc.decode('utf-8').lower()
    tokens = tokenizer.tokenize(raw)
    dict_en = enchant.Dict("en_US")

    stopped_tokens = [token for token in tokens if token not in en_stop]
    stemmed_tokens = [lmtzr.lemmatize(i) for i in stopped_tokens]
    words_tokens = [word for word in stemmed_tokens if dict_en.check(word)]
    return words_tokens


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


def ldamodel(dir_pattern, num_tops=3):
    f = open("low_tfidf.txt", "w")
    text_iter = IterDocs(dir_pattern)
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(line for line in text_iter)

    # convert tokenized documents into a document-term matrix
    corpus = MyCorpus(dir_pattern, dictionary)

    # initialize a tfidf transformation
    tfidf = models.TfidfModel(corpus)

    # filter low tf-idf
    threshold = 0.05
    low_value_words = []
    for bow in corpus:
        low_value_words += [id for id, value in tfidf[bow] if value < threshold]
        # output words w/ low tf-idf
        # TODO: filter out and collect stopwords
        json.dump([(id, dictionary[id], value) for id, value in tfidf[bow] if value < threshold], f)
    f.close()
    dictionary.filter_tokens(low_value_words)
    dictionary.compactify()
    new_corpus = MyCorpus(dir_pattern, dictionary)

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    ldamodel = gensim.models.ldamodel.LdaModel(new_corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    return dictionary, ldamodel
    # return dictionary, texts, ldamodel


def format_result(dist):
    res = dict()
    for topic, words in dist:
        total = 0.0
        words = words.split(' + ')
        res[topic] = list()
        for tp in words:
            prob, word = tp.split('*"')
            word = word.rstrip('"')
            length = len(prob.split('.')[1])
            for count in range(int(float(prob) * pow(10, length))):
                res[topic].append(word)
            total += float(prob)
    return res


def visualize(res):
    for topic in res:
        # Generate a word cloud image
        text = ' '.join(res[topic])
        wordcloud = WordCloud(collocations=False).generate(text)
        image = wordcloud.to_image()
        image.show()

if __name__ == "__main__":
    # dictionary, texts, LDAMODEL = ldamodel("*.txt")
    dictionary, LDAMODEL = ldamodel("*.txt", 2)
    # doc_lda = LDAMODEL[dictionary.doc2bow(texts[3])]
    dist = LDAMODEL.show_topics()
    final_res = format_result(dist)
    print final_res
    visualize(final_res)