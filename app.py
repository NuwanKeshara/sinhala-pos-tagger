from flask import Flask, render_template, url_for, request, jsonify 
from tagger import tagger

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/page1')
def page1():
    return render_template('page1.html')


@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text1']
    sentence = text.strip().split(' ')
    # print(text)
    obj = tagger()
    result = obj.result(sentence)
    print(result)
    return jsonify(result=result)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')