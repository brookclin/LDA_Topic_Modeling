from wordcloud import WordCloud
import pandas as pd
import glob

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


def doc_topic_distribution(corpus, ldamodel, input_path, output_path):
    # column of file names for csv
    fnames = [filename.split('/')[-1] for filename in glob.glob(input_path)]

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
    df.to_csv(output_path + "/doc_topics.csv")
