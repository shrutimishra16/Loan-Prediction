from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    t = request.form["education"]
    if t=='Graduate':
        t = 0
    else:
        t = 1
    education = int(t)
    income = int(request.form["income"])
    prediction = model.predict([[education, income]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text=f'A person with {income}$ income can get a home loan of ${output}')

if __name__ == "__main__":
    app.run()