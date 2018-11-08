from flask import Flask, request, jsonify, Blueprint
from main import AUTORE
import traceback

app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route("/")
def index():
    return 'hello, this is test'

@app.route("/<username>")
def hello(username):
    return 'hello '+ username

@app.route("/relation_ex", methods=['POST'])
def relation_extract():
    try:
        re = AUTORE(config_file='parameter.json')
        tuples = re.bootstrap(sentence_file='sentences.txt')
        return jsonify(
            {
                'ret_code': 0,
                'task': "relation_ex",
                'result': tuples
            }
        )
    except Exception as e:
        return jsonify(
            {
                'ret_code': -1,
                'task': "relation_ex",
                'err_info': traceback.format_exc()
            }
        )

if __name__ == '__main__':
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
    # manage = Manager(app)
    # manage.run()
