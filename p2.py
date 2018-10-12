from pickle import load,dump
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import numpy as np
import string, sklearn, os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import pandas as pd
from time import time
import matplotlib
matplotlib.use('Qt5Agg') # just to use another plotter, take out of pycharm
import matplotlib.pyplot as plt


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
# print(stop_words)
# print(stopwords.words("english"))
punct = set(string.punctuation)

# print(punct)

pos_files = []
for root, dirs, files in os.walk("review_polarity/txt_sentoken/pos"):
    for filename in files:
        if '.txt' in filename:
            pos_files.append(filename)
neg_files = []
for root, dirs, files in os.walk("review_polarity/txt_sentoken/neg"):
    for filename in files:
        if '.txt' in filename:
            neg_files.append(filename)


reviews = []
y = []
for filename in pos_files:
    with open('review_polarity/txt_sentoken/pos/'+filename, encoding='utf-8') as f:
        reviews.append(f.read().lower())
        y.append(1)
for filename in neg_files:
    with open('review_polarity/txt_sentoken/neg/'+filename, encoding='utf-8') as f:
        reviews.append(f.read().lower())
        y.append(0)
# Positive reviews [0:1000] - Negative reviews [1000:2000]

# reviews, re_test, y, y_test = train_test_split(reviews, y, shuffle=True, test_size=0.01,  random_state=13)
# print(np.shape(reviews), np.shape(y))
# TODO: remove this

# example_words = ["python.","pythoner#$%","pythoning.'","pythoned","pythonly?", "you're", "can't", "_election_"]
# str2 = ' '.join(example_words)
# print(str2)
# # str3 = str2.translate(str.maketrans({key: None for key in string.punctuation}))
# str3 = str2.translate(str.maketrans('','', string.punctuation))
# print(str3)
# for w in example_words:
#     print(ps.stem(w))
print("\nCleaning & Preprocessing the data..")

def cleanANDPreprocess(word_Reviews):
    word_revs = []
    word_revsTFid = []
    # # N_reviews = len(reviews) #500
    # N_reviews = len(word_Reviews) #500
    # for i, rev in enumerate(reviews[:N_reviews]):
    for i, rev in enumerate(word_Reviews[:N_reviews]):
        tmp = wordpunct_tokenize(rev)
        tmpList = []
        for w in tmp:
            if w not in stop_words and w not in punct: # punct it s not enough
                again = w.translate(str.maketrans('','', string.punctuation))
                tmpList.append(again)
        word_revs.append(tmpList)
        word_revsTFid.append(' '.join(tmpList))
    # TODO: implement a Stemmer
    return word_revsTFid

# print(np.shape(word_revs))
# to Count how many words are repeat and understand why the Vectorizer deletes other
# s = []
# count = 0
# for w in word_revs[0]:
#     if w not in s:
#         s.append(w)
#     else: count = count +1
# print(count)
# b = np.unique(word_revs[0], return_counts=True)
# print(b)
# -----------------------------------
# a = set(word_revs[0])
# print(len(a))



print("\nImplementing Feature Selection..")

''' smooth_idf:
    Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. 
    Prevents zero divisions. 
 '''

# cvect = CountVectorizer()
# counts = cvect.fit_transform(word_revsTFid)
# normCounts = normalize(counts, norm='l1', axis=1)
# new_X = normCounts.multiply(vectorizer.idf_)

N_reviews = len(reviews) #500
# N_reviews = len(word_Reviews) #500
word_revsTFid = cleanANDPreprocess(reviews)
vectorizer = TfidfVectorizer(min_df=4, smooth_idf=False) # remember better if you apply this little threshold to positive and to the neg separately
X = vectorizer.fit_transform(word_revsTFid)

# sum tfidf frequency of each term through documents
sums = X.sum(axis=0)
# print(sums.sum())
wPer = [] # over 100 %

for s in sums:
    w = s * 100 / sums.sum()
    wPer = np.append(wPer, w)
sums = wPer.reshape(1, -1)
# print(sums.sum())

# print(np.shape(X))
# print((sums))
# connecting term to its sums frequency
data = []
for col, term in enumerate(vectorizer.get_feature_names()):
    data.append((term, sums[0, col]))

ranking = pd.DataFrame(data, columns=['Feature', 'Score over 100%'])
ranking.sort_values('Score over 100%', inplace=True, ascending=True)
print(ranking)


# threshold = 0.79 # TODO: work more on this...
threshold = 0.04
print("\n\tApplying {} as Threshold..".format(threshold))
# # print(np.shape(data))
# data = [(term, score) for (term, score) in data if score > threshold]
# # print(np.shape(data))
#
# # ranking = pd.DataFrame(data, columns=['Feature', 'Score over 100%'])
# # ranking.sort_values('Score over 100%', inplace=True, ascending=True)
# # print(ranking)
# # print(np.shape(X))
# data = np.array(data)
# vecThresH = TfidfVectorizer(min_df=3, smooth_idf=False, vocabulary=data[:, 0])
# X = vecThresH.fit_transform(word_revsTFid)
print("\tDone!")
# print(np.shape(y[:N_reviews]))
# print(np.shape(X))

# ----------------------------------------
# printing a beautiful barGraph
# score = lambda data: data[1]
# data.sort(key=score,reverse=True)
# plt.figure(1)
# plt.tight_layout()
# for i in range(len(data)):
#     plt.bar(i, data[i][1])
#
# plt.title("Relevance of the Words according to Tf-idf")
# plt.ylabel("% of Relevance in the corpus")
# plt.legend([ v[0] for v in data])
# plt.xticks([])
# plt.grid()
# plt.show()
# ------------------------------------
# corpus_index = [n for n, doc in enumerate(word_revsTFid)]
# df = pd.DataFrame(new_X.T.todense(), index=vectorizer.get_feature_names(), columns=corpus_index)

# df = pd.DataFrame(X.T.todense(), index=vectorizer.get_feature_names()) # this was working..
# df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
# print(df.head(15))

# tfidf = pd.DataFrame(index=vectorizer.get_feature_names(), data=vectorizer.idf_ , columns=['Score']).rename_axis('Words', axis='columns').sort_values('Score',ascending=False)

print("\nTraining & Predictions..")
X_train, X_test, y_train, y_test = train_test_split(X, y[:N_reviews], shuffle=True, random_state=15, test_size=0.3)
print("Data was split like that -> ", np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

train = True

t0 = time()
mnb = MultinomialNB()
# mnb = BernoulliNB()
# rndForest = RandomForestClassifier(random_state=12, n_estimators=100, n_jobs=4,
#                                              min_samples_split=23)
rndForest = RandomForestClassifier(n_estimators=100, n_jobs=4)
if train == True:
    mnb.fit(X_train, y_train)
    rndForest.fit(X_train, y_train)

    # classifier = nltk.NaiveBayesClassifier.train(X_train)
    # # to Save
    # save_classifier = open("naivebayes.pickle", "wb")
    # pickle.dump(classifier, save_classifier)
    # save_classifier.close()
else: # LOAD it
    pass
    # classifier_f = open("naivebayes.pickle", "rb")
    # classifier = pickle.load(classifier_f)
    # classifier_f.close()

predictions = mnb.predict(X_test)

# a = np.array(y_test)
# for i in range(len(predictions)):
#     print(predictions[i], " : ", y_test[i])

print("\n\tTraining time was %.3f" % (time() - t0))
scores = cross_val_score(mnb, X_test, y_test, cv=5)
print("\n\tBayesNaives Result")
print("Accuracy: %0.3f (+/- %0.3f)\n" % (scores.mean(), scores.std() * 2))
scores2 = cross_val_score(rndForest, X_test, y_test, cv=5)
print("\tRandomForest Result")
print("Accuracy: %0.3f (+/- %0.3f)\n" % (scores2.mean(), scores2.std() * 2))

newReview = ["Is it a comedy? Is it an action film? [Venom] makes no pretense of being anything more than a superhero film,... trading the thematic material present in most major superhero ventures for biting off people's heads."]

newReview = cleanANDPreprocess(newReview)

newReview = vectorizer.transform(newReview)
newReview = newReview.toarray()
print(mnb.predict(newReview))
print(rndForest.predict(newReview))