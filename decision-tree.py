import pandas as pd
import numpy
from sklearn import tree
import graphviz

filename = "./data/creditcard.csv"

data = pd.read_csv(filename)
print("file read successfuly")

X = data[["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]]
Y = data["Class"]

treeClassifier = tree.DecisionTreeClassifier()

treeClassifier = treeClassifier.fit(X, Y)

dot_data = tree.export_graphviz (treeClassifier, out_file = None, filled=True, rounded=True, special_characters = True)
graph = graphviz.Source(dot_data)
graph.view()
