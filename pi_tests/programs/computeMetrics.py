import numpy as np
import os.path
from sklearn import metrics

if not os.path.exists('test.npy'):
    print("no array found")
    exit
with open('C:/Users/Maurice/Desktop/test.npy', 'rb') as f:
    a = np.load(f)
print(a)
labels = []
pred = []
prob = []
file = []
time = []
for row in a:
    if not row[0] == "detectiontime [s]":
        time.append(float(row[0]))
        file.append(row[1])
        pred.append(row[3])
        prob.append(row[4])
        labels.append(row[5])
#labels = a[:][0]
#print(labels)
#print(pred)
#print(prob)
#print(file)
#print(time)

#conditions = [(labels =='okay'),
#              (labels =='peac'),
#              (labels =='RECOVERING')]
#choices = [int(1), int(0), int(2)]
#data['operation'] = np.select(conditions, choices, default=0)
print("unique predictions:")
unique, counts = np.unique(pred, return_counts=True)
print(dict(zip(unique, counts)))
print("unique true labels:")
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
print("Accuracy:",metrics.accuracy_score(labels, pred))
print(metrics.confusion_matrix(labels, pred))
print(metrics.classification_report(labels, pred))

print(np.mean(time))