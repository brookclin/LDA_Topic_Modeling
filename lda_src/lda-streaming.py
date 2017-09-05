import os
import glob
import gensim
import time
import logging

import matplotlib.pyplot as plt
import pandas as pd
from model import ldamodel
from output import *
from preprocess import load_serialize, filter_tfidf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO # ipython sometimes messes up the logging setup; restore

# os.chdir("C:/Users/John/Desktop/LDA-chunlin")
cur_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
output_path = './experiment/' + cur_time
if not os.path.exists(output_path):
    os.makedirs(output_path)


if __name__ == "__main__":
    num_topics = 3  # 100
    input_path = "sample/*.txt"
    # input_path = "../cp_extracted/*.txt"
    load_serialize(input_path, output_path)
    # filter_tfidf(input_path, output_path, 0.05)
    LDAMODEL = ldamodel(input_path, output_path, num_topics)
    doc_topic_distribution(LDAMODEL, input_path, output_path)
    dist = LDAMODEL.show_topics(num_topics)
    f = open(output_path + '/topics.txt', 'w')
    f.write(str(dist))
    f.close()
    final_res = format_result(dist)
    # print final_res
    visualize(final_res, output_path)
