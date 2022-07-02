import ast
import json
from collections import Counter

from fuzzywuzzy import fuzz
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
import statistics


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


def write_berteilung_corpus_to_file():
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    corpus_out = open("/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_berteilung.txt", "w")

    # Iterating through the json
    # list
    count = 0

    for i in data['hits']['hits']:
        if count >= 6000:
            break

        print(i['_source']['berteleug'])
        corpus_out.write("\n")
        corpus_out.write(i['_source']['berteleug'])
        corpus_out.write("\n")

        count += 1

    corpus_out.close()


def get_vocabulary():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_out.txt', 'r') as fin:
        uniq = set(word_tokenize(fin.read()))
        return uniq


def get_vocabulary_lists():
    vocab_sentences = []
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    a = 0
    b = 0
    f.close()

    for i in data['hits']['hits']:
        a = max(a, len(i['_source']['anamnesis']))

        if len(i['_source']['anamnesis']) <= 256:
            b += 1

        if len(i['_source']['anamnesis']) == 3594:
            print("================================")
            print( "[" + i['_source']['anamnesis'] + "]")
            print("===============================")

        vocab_sentences.append(i['_source']['anamnesis'])

    print(a)
    print(b)


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


context_words_df = None

def get_context_words(text):

    if context_words_df != None:
        df = context_words_df
    else:
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

context_words_df_berteilung = None

def get_context_words_berteilung(text):

    if context_words_df != None:
        df = context_words_df_berteilung
    else:
        df = get_context_words_berteilung_dataframe()

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



def get_context_words_berteilung_dataframe():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_berteilung.txt', 'r') as fin:
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


        context_words_df_berteilung = vocab_df

        return vocab_df

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


        context_words_df = vocab_df

        return vocab_df


def cluster_texts():
    with open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/corpus_berteilung.txt', 'r') as fin:
        raw_corpus = fin.read()
        documents = tokenize.sent_tokenize(raw_corpus)

    print(documents)

    # vectorizer = TfidfVectorizer(stop_words='english')

    cv = CountVectorizer(analyzer='word', max_features=500, lowercase=True, preprocessor=None, tokenizer=None,
                         stop_words='english')

    vectors = cv.fit_transform(documents)
    kmeans = KMeans(n_clusters=12, init='k-means++', random_state=0)
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


def get_stats():
    vocab_sentences = []
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    len_list = []
    words_num = []

    for i in data['hits']['hits']:
        len_list.append(len(i['_source']['anamnesis']))
        words_num.append(len(i['_source']['anamnesis'].split()))
        vocab_sentences.append(i['_source']['anamnesis'])

    print(statistics.mean(len_list))
    print(statistics.mean(words_num))

def find_bertelung_for_most_similar_anamnesis(anamnesis):
    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')

    data = ast.literal_eval(f.read())

    f.close()

    max_ratio = 0
    target_entry = None

    for i in data['hits']['hits']:
        ratio = fuzz.ratio(i['_source']['anamnesis'], anamnesis)

        if ratio > max_ratio:
            target_entry = i
            max_ratio = ratio

    return target_entry['_source']['berteleug']

if __name__ == "__main__":
    # vocab_sentences = []
    # f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')
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

    # get_stats()

    # get_vocabulary_lists()
    # write_berteilung_corpus_to_file()
    cluster_texts()