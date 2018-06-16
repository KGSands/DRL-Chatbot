"""
# 	Class: 
#		create_model
# 	Description: 
#		Load the conversations to learn from and use these
#		to build a word2vec model (change the words to vectors)
#		so that they can be input to the neural network.
#		These are changed into sentence vectors in the main file
#		when they are being input
#	Imports:
#		os: change directory
#		nltk: natural language toolkit - tokenization
#		pandas: reading csv file
"""
import os
import nltk
import gensim
import pandas as pd

# Move to the conversations directory and read the file
os.chdir("C:/Users/Keelan/Desktop/Project/Conversations")
textdata = pd.read_csv("Conversations.csv")

# Create word corpus
greetings = textdata['Greetings'].values.tolist()
place = textdata['Place'].values.tolist()
cost = textdata['Cost'].values.tolist()
corpus = greetings + place + cost

# Tokenize the words in the corpus
tokenise = [nltk.word_tokenize(str(sent).lower()) for sent in corpus]

# Create a word2vec model for the tokenized words
model = gensim.models.Word2Vec(tokenise, min_count=1, size=1)

# Save the model so it can be loaded in the main program
os.chdir("C:/Users/Keelan/Desktop/Project/Vec_Models")
model.save('conversations')