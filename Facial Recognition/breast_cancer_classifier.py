'''A binary classifier using a support vector machine.

Completed with guidance from the book 'Python Machine Learning by Example'
This method only works for linearly separable problems.
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42)
clf = SVC(kernel = "linear", C = 1.0, random_state = 42)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy*100:.1f}%")