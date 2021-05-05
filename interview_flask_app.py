import os
import pickle
import mysklearn.myutils as myutils

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index_page():
    prediction = ""
    if request.method == "POST":
        Danceability = request.form["danceability"]
        Energy = request.form["energy"]
        Valence = request.form['valence']
        prediction = predict_popularity([Danceability, Energy, Valence])
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction) 

@app.route("/", methods=["GET"])
def predict():
    Danceability = request.args.get("danceability")
    Energy = request.args.get("energy")
    Valence = request.args.get('valence')

    prediction = predict_popularity([Danceability, Energy, Valence])
    if prediction is not None:
        # success!
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400
"""
Danceability = tracks.get_column('danceability')
Energy = tracks.get_column('energy')
Speechiness = tracks.get_column('speechiness')
Acousticness = tracks.get_column('acousticness')
Instrumentals = tracks.get_column('instrumentalness')
Liveliness = tracks.get_column('liveness')
Valence = tracks.get_column('valence')
Tempo = tracks.get_column('tempo')
Duration = tracks.get_column('duration_ms')
"""
# recursive


def predict_popularity(unseen_instance):
    # deserialize to object (unpickle)
    infile = open("trees.p", "rb")
    trees = pickle.load(infile)
    infile.close()
    try:
        return myutils.random_forest_predict([unseen_instance], trees)
    except:
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='127.0.0.1', port=port, debug=False)