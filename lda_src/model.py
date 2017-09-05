import gensim
from gensim import corpora, models


def ldamodel(input_path, output_path, num_tops=3):
    serialize_corpus = corpora.MmCorpus(output_path + '/new_SerializedCorpus.mm')
    serialize_dict = corpora.Dictionary.load(output_path + "/new_dictionary")

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20)
    ldamodel = gensim.models.LdaMulticore(serialize_corpus, num_topics=num_tops, id2word=serialize_dict, passes=20, workers=3)

    return ldamodel