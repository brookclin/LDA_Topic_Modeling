import os
# remove_chars = len(os.linesep)
import re
import json
import glob
import gensim
import time
import pyLDAvis.gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import enchant
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.utils import lemmatize
from nltk.stem.wordnet import WordNetLemmatizer

cur_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
if not os.path.exists(cur_time):
    os.makedirs(cur_time)


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
    # stemmed_tokens = [lmtzr.lemmatize(i) for i in stopped_tokens]
    stemmed_tokens = [word.split('/')[0]
                      for word in lemmatize(' '.join(stopped_tokens),
                                            allowed_tags=re.compile('(NN)'),
                                            min_length=3)]
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
    f = open(cur_time + "/low_tfidf.txt", "w")
    f2 = open(cur_time + "/low_tfidf_dict.txt", "w")
    text_iter = IterDocs(dir_pattern)

    # bigram = gensim.models.Phrases(text_iter)
    # bigram = gensim.models.Phrases(text_iter, min_count=1, threshold=2)
    # turn our tokenized documents into a id <-> term dictionary
    # dictionary = corpora.Dictionary(bigram[line] for line in text_iter)

    dictionary = corpora.Dictionary(line for line in text_iter)

    # convert tokenized documents into a document-term matrix
    corpus = MyCorpus(dir_pattern, dictionary)

    # initialize a tfidf transformation
    tfidf = models.TfidfModel(corpus)

    # filter low tf-idf
    threshold = 0.05
    # low_value_words = []
    low_value_words = set()
    f.write('[')
    for bow in corpus:
        # low_value_words += [id for id, value in tfidf[bow] if value < threshold]
        json_list = []
        for id, value in tfidf[bow]:
            if value < threshold:
                low_value_words.add(id)
                json_list.append((dictionary[id], value))
        # output words w/ low tf-idf
        # TODO: filter out and collect stopwords
        json.dump(json_list, f)
        # json.dump([(dictionary[id], value) for id, value in tfidf[bow] if value < threshold], f)
        f.write(',')
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write(']')
    f.close()
    low_value_list = [dictionary[id] for id in low_value_words]
    low_value_list.sort()
    for item in low_value_list:
        f2.write("%s\n" % item)
    f2.close()
    dictionary.filter_tokens(low_value_words)
    dictionary.compactify()
    new_corpus = MyCorpus(dir_pattern, dictionary)

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    ldamodel = gensim.models.ldamodel.LdaModel(new_corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    return new_corpus, dictionary, ldamodel
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
    count = 1
    for topic in res:
        # Generate a word cloud image
        text = ' '.join(res[topic])
        wordcloud = WordCloud(collocations=False).generate(text)
        filename = './' + cur_time + '/' + str(count) + '.png'
        wordcloud.to_file(filename)
        count += 1
        # image = wordcloud.to_image()
        # image.show()

if __name__ == "__main__":
    corpus, dictionary, LDAMODEL = ldamodel("*.txt", 3)
    pyLDAvis.gensim.prepare(LDAMODEL, corpus, dictionary)
    # corpus, dictionary, LDAMODEL = ldamodel("../pdfextractor/results/*.txt", 10)
    dist = LDAMODEL.show_topics()
    final_res = format_result(dist)
    print final_res
    visualize(final_res)