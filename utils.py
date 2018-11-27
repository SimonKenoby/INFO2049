import os
import pandas as pd
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold



def load_data():
	pos_files = []
	for root, dirs, files in os.walk("review_polarity/txt_sentoken/pos"):  
	    for filename in files:
	        if '.txt' in filename:
	            pos_files.append(os.path.join(root, filename))
	neg_files = []
	for root, dirs, files in os.walk("review_polarity/txt_sentoken/neg"):  
	    for filename in files:
	        if '.txt' in filename:
	            neg_files.append(os.path.join(root, filename))

	data = pd.DataFrame(columns=['text', 'type'])
	for filename in pos_files:
	    with open(filename, encoding='utf-8') as f:
	        data= data.append({'text' : f.read().lower().translate(str.maketrans('','',string.punctuation)), 'type' : 'pos'}, ignore_index=True)
	for filename in neg_files:
	    with open(filename, encoding='utf-8') as f:
	        data= data.append({'text' : f.read().lower().translate(str.maketrans('','',string.punctuation)), 'type' : 'neg'}, ignore_index=True)

	return data


def kfolds_tfidf(model):
	data = load_data()
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(data['text'])
	y = data['type'].as_matrix()
	train_accuracy = []
	test_accuracy = []
	kf = KFold(n_splits=10, shuffle = True)
	for train_index, test_index in kf.split(X):    
	    X_train = X[train_index]
	    X_test = X[test_index]
	    y_train = y[train_index]
	    y_test = y[test_index]
	    model.fit(X_train, y_train)
	    train_accuracy.append(model.score(X_train, y_train))
	    test_accuracy.append(model.score(X_test, y_test))
	    #print("Training accuracy : {}".format(model.score(X_train, y_train)))
	    #print("Test accuracy : {}".format(model.score(X_test, y_test)))
	    #print("Classification report for test set")
	    #print(classification_report(y_test, model.predict(X_test)))
	print("Train accuracy : {}".format(np.mean(train_accuracy)))
	print("Test accuracy : {}".format(np.mean(test_accuracy)))
	print(test_accuracy)