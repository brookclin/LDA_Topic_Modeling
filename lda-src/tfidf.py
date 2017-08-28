import gensim
from gensim import corpora


def low_tfidf_terms(output_path, threshold=0.05):
    # TODO: create a function to set threshold then output word lists, to compare with output files
    # TODO: list words with low tf-idf value
    corpus = corpora.MmCorpus(output_path + '/SerializedCorpus.mm')
    dictionary = corpora.Dictionary.load(output_path + "/dictionary")
    tfidf = gensim.models.TfidfModel.load(output_path + "/tfidf_model")
    low_value_words = set()
    for bow in corpus:
        # low_value_words += [id for id, value in tfidf[bow] if value < threshold]
        for id, value in tfidf[bow]:
            if value < threshold:
                low_value_words.add(id)
    f = open(output_path + "/low_tfidf_dict_" + str(threshold) + ".txt", "w")
    low_value_list = [dictionary[id] for id in low_value_words]
    low_value_list.sort()
    for item in low_value_list:
        f.write("%s\n" % item)
    f.close()
    return low_value_words
