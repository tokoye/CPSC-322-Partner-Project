import pickle
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
import os

fname = os.path.join("input_data", "tracks_data.txt")
tracks = MyPyTable().load_from_file(fname)

Danceability = tracks.get_column('danceability')
Energy = tracks.get_column('energy')
Acousticness = tracks.get_column('acousticness')
Valence = tracks.get_column('valence')

y_train = Acousticness
x_train = [[Danceability[i],Energy[i],Valence[i]] for i in range(len(y_train))]

rf = MyRandomForestClassifier()
rf.fit(x_train, y_train, 20, 7, 2)
rf = MyRandomForestClassifier()
rf.fit(x_train, y_train, 30, 4, 2)
# serialize to file (pickle)
outfile = open("trees.p", "wb")
pickle.dump(rf.trees, outfile)
outfile.close()

# deserialize to object (unpickle)
infile = open("trees.p", "rb")
trees2 = pickle.load(infile)
infile.close()
