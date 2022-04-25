from pickle import GET
from flask import Flask, render_template
import process_data
import supervised_models
import semi_supervised_models
import unsupervised

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello_world():
	return 'Hello World'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/supervised', methods=['GET'])
def supervised():
    try:
        return supervised_models.train_with_data()
    except:
        return "Error - Unable to train models"

@app.route('/semiSupervised', methods=['GET'])
def semisup():
    try:
        return semi_supervised_models.semisup()
    except:
        return "Error - Unable to train models"

@app.route('/unsupervised', methods=['GET'])
def unsupervisedAPI():
    try:
        # return unsupervised.unsupervised_cluster()
        return unsupervised.demo()
    except:
        return "Error in training unsupervised models"

if __name__ == '__main__':
	app.run()
