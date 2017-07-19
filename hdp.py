import os
import glob
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import enchant
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer

def get_docs():
    doc_list = []
    for file in glob.glob("*.txt"):
        with open(file,'r') as ofile:
            content = ofile.read()
            doc_list.append(content)
    return doc_list

def ldamodel():
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # lemmtizer
    lmtzr = WordNetLemmatizer()

    # compile sample documents into a list
    doc_set = get_docs()

    # list for tokenized documents in loop
    texts = []

    for doc in doc_set:
        raw = doc.decode('utf-8').lower()
        tokens = tokenizer.tokenize(raw)
        dictionary = enchant.Dict("en_US")

        stopped_tokens = [token for token in tokens if token not in en_stop]
        stemmed_tokens = [lmtzr.lemmatize(i) for i in stopped_tokens]
        words_tokens = [word for word in stemmed_tokens if dictionary.check(word)]
        texts.append(words_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # initialize a tfidf transformation
    tfidf = models.TfidfModel(corpus)
    #print list(tfidf[corpus])

    # filter low tf-idf
    threshold = 0.05
    high_value_words = []
    for bow in corpus:
        high_value_words += [id for id, value in tfidf[bow] if value >= threshold]

    dictionary.filter_tokens(good_ids=high_value_words)
    new_corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(list(tfidf[corpus]), num_topics=3, id2word = dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(new_corpus, num_topics=3, id2word = dictionary, passes=20)
    ldamodel = gensim.models.HdpModel(new_corpus, dictionary)
    return dictionary, texts, ldamodel


def format_result(dist):
    res = dict()
    for topic, words in dist:
        total = 0.0
        words = words.split(' + ')
        res[topic] = list()
        for tp in words:
            prob, word = tp.split('*')
            # HDP output format
            # prob, word = tp.split('*"')
            # word = word.rstrip('"')
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
    dictionary, texts, LDAMODEL = ldamodel()
    doc_lda = LDAMODEL[dictionary.doc2bow(texts[3])]
    dist = LDAMODEL.show_topics()
    final_res = format_result(dist)
    print final_res
    visualize(final_res)