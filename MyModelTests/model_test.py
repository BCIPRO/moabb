from WDataset import WenData
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from moabb.paradigms import MotorImagery

dataset = WenData()
paradigm = MotorImagery()

X, y, metadata = paradigm.get_data(dataset)
shape = X.shape
X = X.reshape((shape[0], shape[1]*shape[2]))

le = LabelEncoder()
y = le.fit_transform(y)

groups = metadata.session.values
cv = LeaveOneGroupOut()

model = SVC()

for train, test in cv.split(X, y, groups):
    model.fit(X[train], y[train])
        
    print(f"predicted: {model.predict(X[test])}, expected {y[test]}, score: {model.score(X[test], y[test])}")

