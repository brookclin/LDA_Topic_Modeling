from wordcloud import WordCloud
from gensim import corpora
import pandas as pd
import glob
import gensim
import os

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


def visualize(res, output_path):
    count = 1
    for topic in res:
        # Generate a word cloud image
        text = ' '.join(res[topic])
        wordcloud = WordCloud(collocations=False).generate(text)
        filename = output_path + '/' + str(count) + '.png'
        wordcloud.to_file(filename)
        count += 1
        # image = wordcloud.to_image()
        # image.show()


def doc_topic_distribution(ldamodel, input_path, output_path):
    """
    Create csv file of document-topic distribution
    """
    # read corpus from file
    # new_corpus = corpora.MmCorpus(output_path + '/new_SerializedCorpus.mm')
    new_corpus = corpora.MmCorpus(output_path + '/SerializedCorpus.mm')

    # column of file names for csv
    fnames = [filename.split('/')[-1] for filename in glob.glob(input_path)]

    # doc-topics distribution to csv
    doc_topics_weights = []
    idx = 0
    for bow in new_corpus:
        row = ldamodel.get_document_topics(bow)
        for tup in row:
            new_tup = [fnames[idx]] + list(tup)
            doc_topics_weights.append(new_tup)
        idx += 1
    df = pd.DataFrame(doc_topics_weights)
    df = df.pivot(index=0, columns=1, values=2)
    df.to_csv(output_path + "/doc_topics.csv")


def corpus_words_output(input_path, output_path):
    corpus = corpora.MmCorpus(output_path + '/SerializedCorpus.mm')
    dictionary = corpora.Dictionary.load(output_path + "/dictionary")
    tfidf = gensim.models.TfidfModel.load(output_path + "/tfidf_model")
    csv_dir = output_path + "/word_tfidf"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # column of file names for csv
    fnames = [filename.split('/')[-1] for filename in glob.glob(input_path)]

    idx = 0
    for bow in corpus:
        # output csv's on (word, tfidf_value) of each article's corpus
        # word_list = [(dictionary[id], value) for id, value in tfidf[bow]]
        bow = map(lambda (id, value): (dictionary[id], value), tfidf[bow])
        df = pd.DataFrame.from_records(bow, columns=['word', 'tfidf'])
        df = df.sort_values(by='tfidf')
        df.to_csv(csv_dir + "/" + fnames[idx] + ".csv")
        idx += 1





