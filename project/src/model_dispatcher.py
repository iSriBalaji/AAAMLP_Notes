# This code dispatch our models to our training scripts
# key: value - name: ml_model

from sklearn import tree
from sklearn import ensemble

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
}