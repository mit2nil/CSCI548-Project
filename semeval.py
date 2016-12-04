import sys
sys.path.append('monolingual-word-aligner')
from glob import glob
from requests import get

from aligner import *
from collections import Counter
import re
import math
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

def read_sentence_file(filename):
	with open(filename) as f:
		return [x.strip().split("\t") for x in f.readlines()]

def read_gs_file(filename):
	with open(filename) as f:
		return [float(x) for x in f.readlines()]

def get_umbc_score(sentence_pair):
	endpoint = "http://swoogle.umbc.edu/StsService/GetStsSim"
	try:
		response = get(endpoint, params={'operation':'api','phrase1':sentence_pair[0],'phrase2':sentence_pair[1]})
		return float(response.text.strip())
	except:
		print 'Error in getting similarity for %s: %s' % ((sentence_pair[0],sentence_pair[1]), response)
		return 0.0

def get_alignment_score(sentence_pair):
	try:
		alignment = align(sentence_pair[0].decode("utf-8").encode("ascii","ignore"), sentence_pair[1].decode("utf-8").encode("ascii","ignore"))
	except:
		alignment = align(re.sub("[#(){}:]+","",sentence_pair[0]).decode("utf-8").encode("ascii","ignore"), 
			re.sub("[#(){}:]+","",sentence_pair[1]).decode("utf-8").encode("ascii","ignore"))
	align_count1 = len([x[0] for x in alignment[1] if x[0] not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	align_count2 = len([x[1] for x in alignment[1] if x[1] not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	content_count1 = len([x for x in alignment[2] if x not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	content_count2 = len([x for x in alignment[3] if x not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	prop1 = align_count1/float(content_count1)
	prop2 = align_count2/float(content_count2)
	if prop1 and prop2:
		return (2*prop1*prop2)/(prop1 + prop2)
	else:
		return 0

def get_cosine_similarity(sentence_pair):
	sentence1 = Counter(word_token_expr.findall(sentence_pair[0]))
	sentence2 = Counter(word_token_expr.findall(sentence_pair[1]))
	common_words = set(sentence1.keys()) & set(sentence2.keys())
    	common_count = sum([sentence1[x] * sentence2[x] for x in common_words])
    	sum1 = sum([sentence1[x]**2 for x in sentence1.keys()])
    	sum2 = sum([sentence2[x]**2 for x in sentence2.keys()])
	union_count = math.sqrt(sum1) * math.sqrt(sum2)
    
    	if not union_count:
        	return 0
	else:
		return float(common_count)/union_count

def read_data(category, dataset_type):
	files = [f for f in glob("data/*") if dataset_type in f]
	sentence_files = [f for train_dir in files for f in glob(train_dir+"/data/*")  if category in f]
	gs_files = [f for train_dir in files for f in glob(train_dir+"/gs/*") if category in f]
	sentences = [x for f in sentence_files for x in read_sentence_file(f)]
	gold_standard_scores = [x for f in gs_files for x in read_gs_file(f)]
	return sentences, gold_standard_scores
	
if __name__ == "__main__": 
	for category in ["c1","c2","c3","c4","c5"]:
		
		print "\n# Running Semantic Textual Similarity for SemEval category : "+category+"\n"
		test_sentences, gold_standard_scores = read_data(category,"test")
		
		print "# Method 1"
		print "# SemEval 2015 rank 5 - DLS@CU-U"
		predicted_alignment_scores = map(get_alignment_score, test_sentences)
		print "Pearson coefficient : "+pearsonr(gold_standard_scores, predicted_alignment_scores)[0]
		print "\n"

		print "# Method 2"
		print "# SemEval 2013 rank 1 - UMBC EBIQUITY-CORE"
		predicted_umbc_scores = map(get_umbc_score, test_sentences)
		print "Pearson coefficient : "+pearsonr(gold_standard_scores, predicted_umbc_scores)[0]
		print "\n"

		print "# Method 3"
		print "# SemEval 2015 rank 1+3 - DLS@CU-S1 and DLS@CU-S2"
		word_token_expr = re.compile(r'\w+')
		predicted_cosine_similarity_scores = map(get_cosine_similarity, test_sentences)

		# Get the training data
		ridge_training_sentences, ridge_gold_standard_scores = read_data(category, "train")

		# Feature 1 - DLS@CU-U  
		ridge_predicted_alignment_scores = map(get_alignment_score, ridge_training_sentences)
		# Feature 2 - Cosine similarity
		ridge_predicted_cosine_similarity_scores = map(get_cosine_similarity, ridge_training_sentences)
		
		# Ridge training
		X_train = zip(ridge_predicted_alignment_scores,ridge_predicted_cosine_similarity_scores)
		ridge_classifier = Ridge(alpha = 1.0)
		ridge_classifier.fit(X_train,ridge_gold_standard_scores)

		# Ridge Prediction
		X_test = zip(predicted_alignment_scores, predicted_cosine_similarity_scores)
		predicted_ridge_scores = ridge_classifier.predict(X_test)
		print "Pearson coefficient : "+pearsonr(gold_standard_scores, predicted_ridge_scores)[0]
		print "\n"
