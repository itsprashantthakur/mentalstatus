from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('logit_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("final.html")


@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)
    if output>str(0.5):
        return render_template('final.html', pred='You might consider relaxing.\n Probability that you may feel depressed is {}'.format(output))
    else:
        return render_template('final.html', pred="You're safe.\n Probability that you may feel depressed is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
