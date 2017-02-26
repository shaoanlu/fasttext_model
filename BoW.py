# From https://www.kaggle.com/c/word2vec-nlp-tutorial#part-2-word-vectors
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

# Create a .csv file for model trinaing
import csv
with open("temp.csv",'w') as file:
    fw = csv.writer(file) #csv.writer(myfile, quoting=csv.QUOTE_ALL)?
    fw.writerows(sentences)
with open("temp.csv",'r') as file:
	y = file.read().replace(","," ")
with open("sentences.csv", "w") as file:
    file.write(y)


import fasttext
# fasttext package provides 2 models, CBOW and Skipgram
#model = fasttext.cbow("sentences3.csv",'model', min_count=40, dim=100, ws=10)
model = fasttext.skipgram("sentences3.csv",'model', min_count=40, dim=100, ws=10)
"""
# import gensim to load movel.vec
from gensim.models import word2vec
model = word2vec.Word2Vec.load_word2vec_format("model.vec")
model.most_similar("car")
"""
