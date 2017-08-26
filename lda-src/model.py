import glob
import gensim
import pandas as pd
from gensim import corpora, models
from iter_class import IterDocs, MyCorpus
from output import doc_topic_distribution


def ldamodel(input_path, output_path, num_tops=3):
    # TODO: begin of individual function
    text_iter = IterDocs(input_path)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(line for line in text_iter)

    # convert tokenized documents into a document-term matrix
    corpus = MyCorpus(input_path, dictionary)

    # initialize a tfidf transformation
    tfidf = models.TfidfModel(corpus)

    # TODO: end, serialize tfidf as well as original corpus and dictionary

    # filter low tf-idf
    threshold = 0.05
    # TODO: create a function to set threshold then output word lists, to compare with output files
    # TODO: list words with low tf-idf value
    low_value_words = set()
    for bow in corpus:
        # low_value_words += [id for id, value in tfidf[bow] if value < threshold]
        for id, value in tfidf[bow]:
            if value < threshold:
                low_value_words.add(id)
    # f = open(output_path + "/low_tfidf_dict.txt", "w")
    # low_value_list = [dictionary[id] for id in low_value_words]
    # low_value_list.sort()
    # for item in low_value_list:
    #     f.write("%s\n" % item)
    # f.close()
    dictionary.filter_tokens(low_value_words)
    dictionary.compactify()
    dictionary.save(output_path + "/dictionary")
    new_corpus = MyCorpus(input_path, dictionary)
    corpora.MmCorpus.serialize(output_path + '/SerializedCorpus.mm', new_corpus)
    serialize_corpus = corpora.MmCorpus(output_path + '/SerializedCorpus.mm')
    serialize_dict = corpora.Dictionary.load(output_path + "/dictionary")

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_tops, id2word=dictionary, passes=20)
    # ldamodel = gensim.models.ldamodel.LdaModel(serialize_corpus, num_topics=num_tops, id2word=dictionary, passes=20)

    # ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=dictionary, passes=20, workers=3)
    ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20, workers=3)

    doc_topic_distribution(corpus, ldamodel, input_path, output_path)

    return serialize_corpus, dictionary, ldamodel, text_iter
