
import json
from flask import Flask, request

from t5_test import generate_for_anamnesis, generate_for_berteilung
from trining_set_provider.corpus_utils import find_bertelung_for_most_similar_anamnesis, get_context_words_berteilung

app = Flask(__name__)

@app.route('/anamnesis')
def anamnesis_generate():
    args = request.args
    context_words = args['key1'].split()

    return generate_for_anamnesis(context_words)


@app.route('/berteilung')
def berteilung_generate():
    args = request.args
    anamnesis = args['key1']
    berteleug = find_bertelung_for_most_similar_anamnesis(anamnesis)

    context_words = get_context_words_berteilung(berteleug)

    return generate_for_berteilung(context_words)


if __name__ == "__main__":
    app.run(port=5001)