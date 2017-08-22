import os
import re
import glob
import gensim
import time
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import enchant
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models
from gensim.utils import lemmatize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO # ipython sometimes messes up the logging setup; restore

cur_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
path = './experiment/' + cur_time
if not os.path.exists(path):
    os.makedirs(path)


def process_doc(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # lemmtizer
    # lmtzr = WordNetLemmatizer()

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
    text_iter = IterDocs(dir_pattern)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(line for line in text_iter)

    # convert tokenized documents into a document-term matrix
    corpus = MyCorpus(dir_pattern, dictionary)

    # initialize a tfidf transformation
    tfidf = models.TfidfModel(corpus)

    # filter low tf-idf
    threshold = 0.05
    # low_value_words = []
    low_value_words = set()
    for bow in corpus:
        # low_value_words += [id for id, value in tfidf[bow] if value < threshold]
        for id, value in tfidf[bow]:
            if value < threshold:
                low_value_words.add(id)
    dictionary.filter_tokens(low_value_words)
    dictionary.compactify()
    dictionary.save(path+"/dictionary")
    new_corpus = MyCorpus(dir_pattern, dictionary)
    corpora.MmCorpus.serialize(path + '/SerializedCorpus.mm', new_corpus)
    serialize_corpus = corpora.MmCorpus(path + '/SerializedCorpus.mm')
    serialize_dict = corpora.Dictionary.load(path+"/dictionary")

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(serialize_corpus, num_topics=num_tops, id2word=dictionary, passes=20)

    # ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=dictionary, passes=20, workers=3)
    ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20, workers=3)
    for bow in corpus:
        print ldamodel.get_document_topics(bow)
    return serialize_corpus, dictionary, ldamodel, text_iter


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
        filename = path + '/' + str(count) + '.png'
        wordcloud.to_file(filename)
        count += 1
        # image = wordcloud.to_image()
        # image.show()


def evaluate_graph(dictionary, corpus, texts, limit):
    # need dictionary before filtering
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    # for num_topics in range(1, limit):
    for num_topics in limit:
        lm = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = gensim.models.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    # x = range(1, limit)
    x = limit
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v

if __name__ == "__main__":
    num_topics = 3
    corpus, dictionary, LDAMODEL, text = ldamodel("sample/*.txt", num_topics)
    # corpus, dictionary, LDAMODEL, text = ldamodel("../pdfextractor/results/*.txt", num_topics)
    dist = LDAMODEL.show_topics(num_topics)
    f = open(path+'/topics.txt', 'w')
    f.write(str(dist))
    f.close()
    final_res = format_result(dist)
    # print final_res
    visualize(final_res)
