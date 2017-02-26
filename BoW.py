import pandas as pd

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

# List 2 .csv
import csv
with open("temp.csv",'w') as file:
    fw = csv.writer(file) #csv.writer(myfile, quoting=csv.QUOTE_ALL)?
    fw.writerows(sentences)
with open("temp.csv",'r') as file:
	y = file.read().replace(","," ")
with open("sentences.csv", "w") as file:
    file.write(y)

"""
# cleaning stopwords in sentences to compress .csv size
%cd O:\pythonTest\IPython
import csv
from gensim.parsing.preprocessing import STOPWORDS

with open("sentences3.csv",'r') as file:
    y = file.read().replace(" the ","")
    for stopword in STOPWORDS:
        y = y.replace(" "+stopword+" "," ")


with open("sentences4.csv", "w") as file:
    file.write(y)
"""


import fasttext

model = fasttext.cbow("sentences.csv",'model',min_count=30,dim=60)
"""
#http://qiita.com/HirofumiYashima/items/be94421837b733ea1da2
most_similar()
similarity is determined by inner product
import numpy as np
car = m['car']
truck = m['truck']
car_n = car/np.linalg.norm(car)
truck_n = truck/np.linalg.norm(truck)

print np.dot(car_n,truck_n)
print m.most_similar['car']
"""

"""
%cd O:\pythonTest\IPython
from gensim.models import word2vec
from gensim.parsing import PorterStemmer
model = word2vec.Word2Vec.load_word2vec_format("modelSkipgram_100dim_40minCount_10ws.vec")
#model = word2vec.Word2Vec.load_word2vec_format("modelCBOW_400minCount_100dim_10ws.vec")
#model = word2vec.Word2Vec.load("300features_40minwords_10context.gensim")
modelCBOW_400minCount_100dim_10ws.vec
#model.most_similar("car")

#
einstein = model['einstein']
genius = model['genius']
#model.similar_by_vector(model['einstein'] - model['genius'])
model.similar_by_vector(einstein - genius)
stemmer = PorterStemmer()
stemmer.stem()


#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
x = model[model.vocab]
x_pca = PCA().fit_transform(x)

similarList = model.most_similar("example")
words = []
X=[]
Y=[]
Z=[]
words.append("example")
index = model.vocab.keys().index("example")
X.append(x_pca[index,0])
Y.append(x_pca[index,1])
for word,similarity in similarList:
    index = model.vocab.keys().index(word)
    words.append(word)
    X.append(x_pca[index,0])
    Y.append(x_pca[index,1])



import matplotlib.pyplot as plt
fig,p = plt.subplots()
p.scatter(X, Y)
for i,txt in enumerate(words):
    p.annotate(txt,(X[i],Y[i]))

fig.tight_layout()
fig.show()
fig.savefig('temp.png', dpi=fig.dpi)

similarList = model.most_similar("example")
words = []
x = []
words.append("example")
index = model.vocab.keys().index("example")
x.append(model["example"])
for word,similarity in similarList:
    words.append(word)
    x.append(model[word])
    
for i,txt in enumerate(words):
    p.annotate(txt,x_tsne_n[i])

x_tsne = TSNE(n_components=2).fit_transform(x)
import matplotlib.pyplot as plt

"""

