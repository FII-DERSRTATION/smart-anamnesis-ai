import ast
import json
from collections import Counter

from keras.preprocessing.text import one_hot
from nltk import FreqDist, word_tokenize
from nltk import tokenize
import gensim
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import pandas as pd


def write_corpus_to_file():
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    corpus_out = open("corpus_out.txt", "w")

    # Iterating through the json
    # list
    for i in data['hits']['hits']:
        print(i['_source']['anamnesis'])
        corpus_out.write("\n")
        corpus_out.write(i['_source']['anamnesis'])
        corpus_out.write("\n")

    corpus_out.close()


def get_vocabulary():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_out.txt', 'r') as fin:
        uniq = set(word_tokenize(fin.read()))
        return uniq


def get_vocabulary_lists():
    vocab_sentences = []
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    for i in data['hits']['hits']:
        vocab_sentences.append(i['_source']['anamnesis'])


def get_vocabulary_length():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_out.txt', 'r') as fin:
        uniq = set(word_tokenize(fin.read()))
        return len(uniq)


def training_set_one_hot_encodings():
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    vocab_size = get_vocabulary_length()
    one_hot_encodings = []

    k = 750
    for i in data['hits']['hits']:
        print(one_hot(i['_source']['anamnesis'], vocab_size))
        one_hot_encodings.append(one_hot(i['_source']['anamnesis'], vocab_size))
        k -= 1
        if k <= 0:
            break

    return one_hot_encodings


def get_context_words_dataframe():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_out.txt', 'r') as fin:
        raw_corpus = fin.read()
        tok = tokenize.sent_tokenize(raw_corpus)
        df = pd.DataFrame({"texts": tok})

        tfidf_vectorizer = TfidfVectorizer(ngram_range=[1, 1], stop_words='english')
        tfidf_separate = tfidf_vectorizer.fit_transform(df["texts"])

        word_lst = tfidf_vectorizer.get_feature_names()
        count_lst = tfidf_separate.toarray().sum(axis=0)

        vocab_df = pd.DataFrame((zip(word_lst, count_lst)),
                                columns=["vocab", "tfidf_value"])

        vocab_df.sort_values(by="tfidf_value", ascending=False)

        return vocab_df


def get_context_words(text):
    df = get_context_words_dataframe()
    words = word_tokenize(text)
    words_tf_idf = {}

    for word in words:
        word = word.lower()

        try:
            word_tfidf = df.query("vocab == '%s'" % (word))
            score = word_tfidf['tfidf_value'].values[0]
            words_tf_idf[word] = score
        except:
            continue


    if len(words_tf_idf) <= 0:
        return None

    sorted_words = sorted(words_tf_idf.items(), key=lambda x: x[1])

    # return only the first 5 words

    return [w[0] for w in sorted_words[-5:]]


def cluster_texts():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_out.txt', 'r') as fin:
        raw_corpus = fin.read()
        documents = tokenize.sent_tokenize(raw_corpus)

    # vectorizer = TfidfVectorizer(stop_words='english')

    cv = CountVectorizer(analyzer='word', max_features=500, lowercase=True, preprocessor=None, tokenizer=None,
                         stop_words='english')

    vectors = cv.fit_transform(documents)
    kmeans = KMeans(n_clusters=22, init='k-means++', random_state=0)
    kmean_indices = kmeans.fit_predict(vectors)

    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(vectors.toarray())

    colors = ['sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']



    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]
    fig, ax = plt.subplots(figsize=(150, 30))

    ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

    for i, txt in enumerate(documents):
        ax.annotate(txt, (x_axis[i], y_axis[i]))

    plt.show()



if __name__ == "__main__":
    # vocab_sentences = []
    # f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json')
    #
    # # returns JSON object as
    # # a dictionary
    # data = ast.literal_eval(f.read())
    #
    # f.close()
    #
    # for i in data['hits']['hits']:
    #     anamnesis = i['_source']['anamnesis']
    #     ctx_wds = get_context_words(anamnesis)
    #
    #     print(50 * "=")
    #     print(anamnesis)
    #     print(ctx_wds)
    #     print(50 * "=")

    cluster_texts()