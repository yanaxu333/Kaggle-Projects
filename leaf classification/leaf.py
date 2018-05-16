import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args,**kwarges): pass

import warnings 
warnings.warn = warn
from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit





train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')


# data preparation 
#swiss army knife function to orginize the data
def encode(train, test ):
	le = LabelEncoder().fit(train.species)

	labels = le.transform(train.species) #encode species strings
	classes = list(le.classes_)
	test_ids = test.id

	train = train.drop(['species','id'], axis=1)


	test = test.drop(['id'], axis=1)

	return train, labels, test, test_ids, classes


train, labels, test, test_ids, classes = encode(train, test )
print train.head(1)

# Stratified Train and test split 

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2,random_state = 23)


for train_index, test_index in sss:

	x_train, x_test = train.values[train_index], train.values[test_index]

	y_train, y_test = labels[train_index],labels[test_index]

print x_train.shape 
print y_train.shape
print x_test.shape
print y_test.shape
# sklearn classfication 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers =[
	KNeighborsClassifier(2),
	SVC(kernel = 'rbf',C = 0.025, probability = True),
	NuSVC(probability=True),
	DecisionTreeClassifier(),
	RandomForestClassifier(),
	AdaBoostClassifier(),
	GradientBoostingClassifier(),
	GaussianNB(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis()]

#logging for visual Comparison 
log_cols = ['Classifier', 'Accuracy','Log loss']
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
	clf.fit(x_train, y_train)
	name = clf.__class__.__name__
	print('='*30)
	print(name)

	print('...result...')
	train_predictions = clf.predict(x_test)
	acc=accuracy_score(y_test, train_predictions)
	print('accuracy: {:.4%}'.format(acc))


	train_predictions = clf.predict_proba(x_test)
	ll = log_loss(y_test, train_predictions)
	print ('log loss:{}'.format(ll))

	log_entry = pd.DataFrame([[name, acc*100,ll]], columns = log_cols)
	log = log.append(log_entry)

print("="*30)


print log


sns.set_color_codes('muted')
sns.barplot(x ='Accuracy', y = 'Classifier', data=log, color='b')


plt.xlabel('Accuracy %')
plt.title('classifier accuracy')
plt.show()

sns.set_color_codes('muted')
sns.barplot(x='Log loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

#predict test set 
favorite_clf = LinearDiscriminantAnalysis()
favorite_clf.fit(x_train, y_train)
test_predictions  = favorite_clf.predict_proba(test)

#format dataframe 
submission =  pd.DataFrame(test_predictions, columns = classes)
print submission.tail(1)

submission.insert(0,'id',test_ids)
print submission.tail(1)

submission.reset_index()
print submission.tail(1)

