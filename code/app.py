from pickle import GET
from flask import Flask, render_template
import process_data
import supervised_models
import semi_supervised_models

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello_world():
	return 'Hello World'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/splitData', methods=['GET'])
def splitData():
    try:
        process_data.split_data()
        return "Split Successful"
    except:
        return "Split Failed"

@app.route('/supervised', methods=['GET'])
def supervised():
    try:
        return supervised_models.train_with_data()
    except:
        return "Error - Unable to train models"

@app.route('/semisup', methods=['GET'])
def semisup():
    try:
        return semi_supervised_models.semisup()
    except:
        return "Error - Unable to train models"

if __name__ == '__main__':
	app.run()
