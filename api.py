from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils
import itertools
# using the "spotipy" library which is a python library that utilizes the Spotify API
import spotipy 
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials 
# https://opendatascience.com/a-machine-learning-deep-dive-into-my-spotify-data/ similar project with interesting info on dealing with the same data
# https://developer.spotify.com/dashboard/applications/09c0ca42f3c54683ad63963988c3de9f my spotify dev site, gmail login
#the client id and secret  needed to make Spotify api calls


# def api_call(): 
cid = "09c0ca42f3c54683ad63963988c3de9f" 
secret = "30fed791c2ed465d9a1e430468a88c37"

auth_manager = SpotifyClientCredentials(client_id= cid, client_secret=secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
sp.trace=False
playlist_id = "6FKDzNYZ8IW1pvYVF4zUN2" # this playlist has 10000 songs on it called "The Longest PlayList On Spotify" https://open.spotify.com/playlist/6FKDzNYZ8IW1pvYVF4zUN2
offs = 0
#grab first 100 track objects
results = sp.playlist_items(playlist_id, offset=offs) 
tracks = results["items"]
#loop through until 400 tracks
while offs < 5000: 
    results = sp.playlist_items(playlist_id, offset=offs)
    tracks.extend(results["items"])
    offs += 100

track_data_objs = []
#build the track data objects and append
# first = True
# header = ["Track_name"]
for track in tracks:
    track_data = []
    popularity = track["track"]["popularity"]
    if popularity == 0:
        continue #skip any track with 0 popularity, because I'm unsure if this is a default value 
    else:
        #name = track["track"]["name"]
        #track_data.append(name) # will be ignored but could but each track_data_obj should be identifiable
        features_dict = sp.audio_features(track["track"]["id"]) #this  returns a features dictonary
        for key in features_dict[0]: #loop through and add only the attributes we want
            if key != "type" and key != "id" and key != "uri" and key != "track_href" and key != "analysis_url" and key != "time_signature" and key != "mode" and key != "key" and key != "loudness":
                val = features_dict[0][key]
                if key != "tempo" and key != "duration_ms":
                    val = myutils.percent_to_rating(val)
                track_data.append(val)
                # if first == True:
                #     header.append(key)
        # first = False
        pop_class = myutils.pop_rating(popularity)
        track_data.append(pop_class) # popularity will be the y_train
        track_data_objs.append(track_data)
# header.append("popularity")
# now we can turn this into an xtrain and ytrain or keep it stitched together 
# when dealing with the data we can delete the first col, which is the name identifier

print(len(track_data_objs))

header = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity']

tracks_mypy = MyPyTable(header, track_data_objs)
tracks_mypy.save_to_file("tracks_data.txt")
  