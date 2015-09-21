import numpy as np
from flask import Flask, jsonify, request
import cPickle as pickle



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load Model
    my_rfc_model = pickle.load(open("iris_rfc.pkl", "rb"))

    # more error checking should go here
    data = request.get_json(force=True)

    #convert our json to a numpy array
    predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
    predict_request = np.array(predict_request)

    #np array goes into random forest, prediction comes out
    y_hat = my_rfc_model.predict(predict_request)

    #return our prediction
    output = [y_hat[0]]
    return jsonify(results=output)

if __name__=="__main__":
    app.run(port=9000, debug=True)