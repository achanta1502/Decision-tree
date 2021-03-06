from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import random

header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv', header=None, names=['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.3)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
nodesList = []
for i in innerNodes:
    nodesList.append(i.id)
for j in leaves:
    nodesList.append(j.id)

arrayOfNodes = []
for i in range(0, 7):
    a = random.choice(nodesList)
    if a not in arrayOfNodes:
        arrayOfNodes.append(a)

print("*************Nodes used for pruning*******")
print(arrayOfNodes)
t_pruned = prune_tree(t, arrayOfNodes)
print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
