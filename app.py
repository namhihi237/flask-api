
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from tree import Tree_node, Decision_Tree
import pickle as p
import json
n_layers = 12
n_neurons = 32
n_inputs = 3
n_outputs = 1

app = Flask(__name__)
object = Decision_Tree()


@app.route('/')
def hi():
    print('hi')
    return 'hello'


@app.route('/api/v1/rain', methods=['POST'])
def api():

    data = request.get_json()  # [{},{},{}]
    data = data['input']
    # handle data
    input_data = np.empty((0, 4))
    for item in data:
        list_values = [v for v in item.values()]
        x = np.array(list_values, dtype=np.float32).reshape(-1, 4)
        input_data = np.concatenate((input_data, x), axis=0)

    input_data.reshape(-1, 12, 4)
    input_data = input_data.min(axis=0).reshape(-1, 4)
    print("load")
    modelfile = 'model_tree/tree3.pickle'
    with open(modelfile, "rb") as f:
        object = p.load(f)

    print("load ok")
    a = object.predict(input_data)
    print(a)
    prediction = a[0]
    return {"result": prediction}


@app.route('/api/v1/temp', methods=['POST'])
def apiv1():
    # get data from request
    data = request.get_json()
    data = data['input']
    # handle data
    # print(data)
    input_data = np.empty((0, 3))
    for item in data:
        list_values = [v for v in item.values()]
        x = np.array(list_values, dtype=np.float32).reshape(-1, 3)
        input_data = np.concatenate((input_data, x), axis=0)

    X0 = input_data
    print(X0)
    # try:
    with tf.compat.v1.Session() as sess:
        print("a")
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.import_meta_graph("models/model.ckpt.meta")
        save_path = saver.restore(sess, "models/model.ckpt")
        graph = tf.compat.v1.get_default_graph()
        get = graph.get_tensor_by_name('Reshape_1:0')
        X = graph.get_tensor_by_name('Placeholder:0')
        X_batches = X0.reshape(-1, n_layers, n_inputs)

        y_pred = sess.run(get, feed_dict={X: X_batches})
        print("ok")

        print(y_pred)
        return {"result": str(y_pred[-1][-1][-1])}


if __name__ == '__main__':
    # load model and run server

    app.run(debug=True)
