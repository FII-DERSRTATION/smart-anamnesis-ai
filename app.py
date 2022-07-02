import json

import requests
from flask import Flask, render_template, request
from flask import send_file

from docs_generator.docs_generator import generate_dummy_docx_file

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html', title='test', userName='kentei')

@app.route('/generate')
def generate():
    firstName = request.args['firstName']
    lastName = request.args['lastName']
    SID = request.args['SID']
    anamnesis = request.args['anamnesis']
    statsMap = json.loads(request.args['statsMap'])


    anamnesisGen = requests.get("http://127.0.0.1:5001/anamnesis", params={'key1': anamnesis}).content
    berteilung = requests.get("http://127.0.0.1:5001/berteilung", params={'key1': anamnesisGen}).content

    generate_dummy_docx_file(firstName, lastName, SID, anamnesisGen, berteilung)

    return ''


@app.route('/download') # this is a job for GET, not POST
def plot_csv():
    return send_file('/Users/danilamarius-cristian/PycharmProjects/pythonProject2/report.docx',
                     mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                     attachment_filename='report.docx',
                     as_attachment=True)


if __name__ == '__main__':
    app.run()
