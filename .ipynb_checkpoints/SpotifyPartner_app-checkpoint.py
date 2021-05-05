from flask import Flask, jsonify 

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():

    return "<h1> Welcome to our Web App by Charles and Toby:</h1>", 200

@app.route("/", methods=["GET"])
def predict():
    Danceability = request.args.get("danceability","")
    Energy = request.args.get("energy","")
    Speechiness = request.args.get('speechiness',"")
    Acousticness = request.args.get('acousticness',"")
    Instrumentals = request.args.get('instrumentalness',"")
    Liveliness = request.args.get('liveness',"")
    Valence = request.args.get('valence',"")
    Tempo = request.args.get('tempo',"")
    Duration = request.args.get('duration_ms',"")
    print("popularity:", Danceability, Energy, Speechiness, Acousticness, Instrumentals, Liveliness, Valence, Tempo, Duration)

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
    result = {"prediction": "True"}
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)