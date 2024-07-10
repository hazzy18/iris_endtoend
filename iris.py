import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('iris.csv')
X=np.array(df.iloc[:,0:4])
Y=np.array(df['Species'])
le = LabelEncoder()
Y=le.fit_transform(Y.reshape(-1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)
clf = SVC(kernel='linear')
clf.fit(X_train,Y_train)
pickle.dump(clf,open('iris.pkl','wb'))

