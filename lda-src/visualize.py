from wordcloud import WordCloud


def visualize(res, path):
    count = 1
    for topic in res:
        # Generate a word cloud image
        text = ' '.join(res[topic])
        wordcloud = WordCloud(collocations=False).generate(text)
        filename = path + '/' + str(count) + '.png'
        wordcloud.to_file(filename)
        count += 1
        # image = wordcloud.to_image()
        # image.show()