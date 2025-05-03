import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.title("Damodar Pokhrel app")
st.write("Explore different classifiers and dataset which one is best for you")
dataset = st.selectbox("Select dataset", ["Iris", "Wine", "Breast Cancer"])
st.write(f"## {dataset} dataset")
classifier_name = st.sidebar.selectbox('Select classifier', ['KNN', 'SVM', 'Random Forest'])

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'KNN':
        K = st.slider("K", 1, 15)
        params['K'] = K
    elif classifier_name == 'SVM':
        C = st.slider("C", 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.slider("max_depth", 2, 15)
        n_estimators = st.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                   max_depth=params['max_depth'], 
                                   random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write("Classifier = ", classifier_name)
st.write("Accuracy = ", acc)
st.write("Parameters = ", params)

# PCA Visualization
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, c=y, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset')
st.pyplot(fig)




