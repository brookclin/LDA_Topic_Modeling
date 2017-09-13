import gensim
from gensim import corpora, models


def ldamodel(output_path, num_tops=3):
    # serialize_corpus = corpora.MmCorpus(output_path + '/new_SerializedCorpus.mm')
    serialize_corpus = corpora.MmCorpus(output_path + '/SerializedCorpus.mm')
    # serialize_dict = corpora.Dictionary.load(output_path + "/new_dictionary")
    serialize_dict = corpora.Dictionary.load(output_path + "/dictionary")

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20)
    ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20, workers=3)
    ldamodel.save(output_path + "/ldamodel")
    return ldamodel
