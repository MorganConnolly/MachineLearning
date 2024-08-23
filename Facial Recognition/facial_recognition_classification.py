'''A multiclass classifier using as support vector machine.

Completed with guidance from the book 'Python Machine Learning by Example'
Uses a non-linear radius-basis function kernel enabling classification of
non-linear problems
'''
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

face_data = fetch_lfw_people(min_faces_per_person = 60)
X = face_data.data
Y = face_data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42)

pca = PCA(n_components = 100, whiten = True, random_state = 42)
svc = SVC(class_weight = "balanced", kernel = "rbf", random_state = 42)
model = Pipeline([("pca", pca), ("svc", svc)])
parameters_pipeline = {"svc__C": [1, 3, 10], "svc__gamma": [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)
print("The best model:\n", grid_search.best_params_)
print(f"The accuracy is: {best_model.score(X_test, Y_test)*100:.2f}%")
print(classification_report(Y_test, pred, target_names = face_data.target_names))

# fig, ax = plt.subplots(3, 4)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(face_data.images[i], cmap = "bone")
#     axi.set(xticks = [], yticks = [],
#         xlabel = face_data.target_names[face_data.target[i]])

# plt.show()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42)
# clf = SVC(class_weight = "balanced", random_state = 42)
# parameters = {"C": [0.1, 1, 10], "gamma": [1e-07, 1e-08, 1e-06], "kernel": ["rbf", "linear"]}
# grid_search = GridSearchCV(clf, parameters, n_jobs = -1, cv = 5)
# grid_search.fit(X_train, Y_train)
# # print("The best model:\n", grid_search.best_params_)

# clf_best = grid_search.best_estimator_
# pred = clf_best.predict(X_test)
# print(f"The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%")
# print(classification_report(Y_test, pred, target_names = face_data.target_names))