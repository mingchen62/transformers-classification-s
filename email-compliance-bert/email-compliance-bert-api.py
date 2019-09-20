#!/usr/bin/env python
"""
    API entrance
"""
from __future__ import division

import argparse
import os, glob
from pathlib import Path

import time
import json

from train import _train
from inference import model_fn, predict_fn

from flask import Flask, render_template, request,Response, jsonify
app = Flask(__name__)
from flask_cors import CORS
CORS(app)


# globals initialization

print(" model initialized")
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='bert')
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--task_name', type=str, default='binary')
parser.add_argument('--output_mode', type=str, default='classification')
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--model_dir', type=str, default='/home/mc00/transformers-classification-sm/model')
args= vars(parser.parse_args())
print(parser.parse_args())

model = model_fn(args['model_dir'])
print(" model loaded")



@app.route('/predict', methods=['POST'])
def translate():
    if request.method == 'POST':
        # user inputs
        input_data = request.json
        print(input_data)

        # api call
        res={'prediction':0}
        if len(input_data['txt']) >0:
            print("input_data txt "+str(input_data['txt']))

            try:
                start_t=time.time()
                res['prediction'] = predict_fn(input_data, model)
                print("completed {} ".format(time.time-start_t))
            except:
                print("process error, return empty str")

        return jsonify(res)


    return render_template('index.html')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8088))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
    app.run(debug=True)