from gensim import corpora, models
from iter_class import IterDocs, MyCorpus


def ldamodel(dir_pattern, path, num_tops=3):
    text_iter = IterDocs(dir_pattern)
    # column of file names for csv
    fnames = [filename.split('/')[-1] for filename in glob.glob(dir_pattern)]

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

    # doc-topics distribution to csv
    doc_topics_weights = []
    idx = 0
    for bow in corpus:
        row = ldamodel.get_document_topics(bow)
        for tup in row:
            new_tup = [fnames[idx]] + list(tup)
            doc_topics_weights.append(new_tup)
        idx += 1
    df = pd.DataFrame(doc_topics_weights)
    df = df.pivot(index=0, columns=1, values=2)
    df.to_csv(path + "/doc_topics.csv")

    return serialize_corpus, dictionary, ldamodel, text_iter