import re
import enchant
from gensim.utils import lemmatize
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words


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
                      for word in lemmatize(' '.join(stopped_tokens))]
    # stemmed_tokens = [word.split('/')[0]
    #                   for word in lemmatize(' '.join(stopped_tokens),
    #                                         allowed_tags=re.compile('(NN)'),
    #                                         min_length=3)]
    words_tokens = [word for word in stemmed_tokens if dict_en.check(word)]
    return words_tokens