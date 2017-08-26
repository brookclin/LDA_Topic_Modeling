import os
import glob
import gensim
import time
import logging

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from model import ldamodel
from output import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO # ipython sometimes messes up the logging setup; restore

# os.chdir("C:/Users/John/Desktop/LDA-chunlin")
cur_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
path = './experiment/' + cur_time
if not os.path.exists(path):
    os.makedirs(path)


if __name__ == "__main__":
    num_topics = 3  # 100
    corpus, dictionary, LDAMODEL, text = ldamodel("sample/*.txt", path, num_topics)
    # corpus, dictionary, LDAMODEL, text = ldamodel("../cp_extracted/*.txt", num_topics)
    dist = LDAMODEL.show_topics(num_topics)
    f = open(path+'/topics.txt', 'w')
    f.write(str(dist))
    f.close()
    final_res = format_result(dist)
    # print final_res
    visualize(final_res, path)
