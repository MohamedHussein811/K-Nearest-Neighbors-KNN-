import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

data = load_digits()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
dataset

X = dataset.copy()
Y = data['target']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.70)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier =classifier.fit(X_train,Y_train)
classifier.get_params()

predictions = classifier.predict(X_test)
predictions

accuracy=accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')

confusion_matrix(Y_test,predictions,labels=[0,1])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)

x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y_train, edgecolors='k', marker='o', s=60, linewidth=1, cmap=plt.cm.Paired)
plt.title('k-Nearest Neighbors Decision Boundaries')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()