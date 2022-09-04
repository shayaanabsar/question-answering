from os import listdir, path
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation
from math import log

STOPWORDS = set(stopwords.words('english'))

NUM_DOCUMENTS = len(listdir('corpus'))

files = {}
tokenized_files = {}
idfs = {}

def load_files():
	file_names = listdir('corpus') 
	# Get the names of all files in the corpus

	for file in file_names:
		with open(path.join('corpus', file), encoding='utf8') as f:
			# Place the contents of the file into the files dict
			files[file] = f.read()
			# Place the filtered version into tokenized_files dict
			tokenized_files[file] = filter_text(files[file])

def filter_text(text):
	# Remove punctuation
	for i in punctuation:
		text = text.replace(i, '')
	# Use nltk word tokenizer to tokenize
	text = word_tokenize(text.lower()) 
	# Use our lemmatize function to lemmatize the text
	text = lemmatize(text) 
	# Remove stopwords
	text = [word for word in text if word not in STOPWORDS] 

	return text

def lemmatize(text):
	text = pos_tag(text) 
	# Use nltk's pos function to find the Part of Speech
	for i, val in enumerate(text):
		word, tag = val
		# The pos tags returned by nltk's pos function 
		# are differented to the ones used by the lemmatizer
		# So the tags need to be converted

		match tag[0]:
			case 'J':
				tag = wordnet.ADJ
			case 'V':
				tag = wordnet.VERB
			case 'N':
				tag = wordnet.NOUN
			case 'R':
				tag = wordnet.ADV
			case _:
				tag = None

		# Use the WordNet lemmatizer to lemmatize the word
		# using the word and the pos tag (if one was found)

		if tag is None:
			word =  WordNetLemmatizer().lemmatize(word)
		else:
			word = WordNetLemmatizer().lemmatize(word, tag)
		
		text[i] = word

	return text




def calculate_idfs():
	# Calculate idf values for all of the words.

	for file in tokenized_files:
		# Create a set of each word in the file (sets remove duplicates)
		words = set(tokenized_files[file])

		# Loop through the set and add 1 to the words count in the idf dict
		for word in words:
			if word in idfs:
				idfs[word] += 1
			else:
				idfs[word] = 1
	
	# idfs[i] is the number of documents containing the word
	#print(idfs)
	for i in idfs:
		idfs[i] = log(NUM_DOCUMENTS / idfs[i] + 1)

def top_score(relevancy_scores):
	return max(relevancy_scores, key=lambda x: relevancy_scores[x])

def top_file(query):
	relevancy_scores = {}

	for file in tokenized_files:
		relevancy_scores[file] = 0
		# Find the total tf-idf value for each file
		for word in query:
			try:
				relevancy_scores[file] += (idfs[word] * tokenized_files[file].count(word))
			except KeyError:
				pass
	# Return file with highest value
	return top_score(relevancy_scores)

def top_sentence(query, top_file):
	relevancy_scores = {}
	#Tokenize the most relevant file into sentences
	sentences = sent_tokenize(files[top_file])

	for sentence in sentences:
		relevancy_scores[sentence] = 0
		words = filter_text(sentence)
		# Find total idf value for each sentence
		for word in query:
			if word in words:
				try:
					relevancy_scores[sentence] += idfs[word]
				except KeyError:
					pass
	# Return sentence with the highest value
	return top_score(relevancy_scores)



def main():
	load_files()
	calculate_idfs()
	
	while True:
		query = filter_text(input(">>> "))
		print(top_sentence(query, top_file(query)))
	

main()
