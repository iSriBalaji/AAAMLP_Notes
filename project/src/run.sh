#!/bin/sh

echo "Training Model"
python train.py -fold 0 -model decision_tree_gini
python train.py -fold 1 -model decision_tree_entropy
python train.py -fold 2 -model rf
python train.py -fold 3 -model decision_tree_entropy
python train.py -fold 4 -model decision_tree_entropy
python train.py -fold 5 -model decision_tree_gini
python train.py -fold 6 -model rf
python train.py -fold 7 -model rf
python train.py -fold 8 -model rf
python train.py -fold 9 -model rf
