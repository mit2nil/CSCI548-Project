#!/usr/bin/python

import sys,re, getopt, math
sys.path.append('monolingual-word-aligner')
sys.dont_write_bytecode = True

from glob import glob
from requests import get
from aligner import *
from collections import Counter
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
	sentence_files = [f for datadir in files for f in glob(datadir+"/data/*")  if category in f]
	gs_files = [f for datadir in files for f in glob(datadir+"/gs/*") if category in f]
	sentences = [x for f in sentence_files for x in read_sentence_file(f)]
	gold_standard_scores = [x for f in gs_files for x in read_gs_file(f)]
	return sentences, gold_standard_scores
	
def extractCategories(dir, dataset_type):
	files = [f for f in glob(dir+"/*") if dataset_type in f]
	files = [f for datadir in files for f in glob(datadir+"/data/*")]
	files = [os.path.split(f)[1] for f in files]
	files = [f[:f.index('_')] if '_' in f else f[:f.index('.')] for f in files]
	return sorted(list(set(files)))

def printUsage():
	print 'Usage: semeval.py [options]'
	print '       -r <dirname> or --traindir=<dirname> : Directory containing training data.'
	print '       -t <dirname> or --testdir=<dirname> : Directory containing test data.'
	print '       -e <dirname> or --evaldir=<dirname> : Directory containing evaluation data. No need of gold standard files.'
	print '       -h : Print this help.'
	print 'NOTE: Refer to https://github.com/mit2nil/CSCI548-Project for further installation and usage details.\n'

def main(argv):
	traindir = ""
	testdir = ""
	evaldir = ""

	try:
		opts, args = getopt.getopt(argv,"hr:t:e:",["traindir=","testdir=","evaldir="])
	except:
		print "Please provide appropriate options!\n"
		printUsage()
		sys.exit(1)

	if len(opts) == 0:
		print "Please provide appropriate options!\n"
		printUsage()
		sys.exit(1)

	for opt, arg in opts:
		if opt == '-h':
			printUsage()
			sys.exit(1)
		elif opt in ('-r','--traindir'):
			traindir = arg
			print "Training dir is set to :",traindir
		elif opt in ('-t','--testdir'):
			testdir = arg
			print "Test dir is set to :",testdir
		elif opt in ('-e','--evaldir'):
			evaldir = arg
			print "Evaluation dir is set to:",evaldir
	
	# Arguments checks
	if traindir == "" or (traindir != "" and not os.path.isdir(traindir)):
		print "Training data is required.\n"
		printUsage()
		sys.exit(1)
	
	if testdir == "" and evaldir == "":
		print "Both test and evaluation directory can't be skipped.\n"
		printUsage()
		sys.exit(1)
	elif testdir != "" and not os.path.isdir(testdir):
		print "Can't find test directory.",testdir
		sys.exit(1)
	elif evaldir != "" and not os.path.isdir(evaldir):
		print "Can't find evaluation directory.",evaldir
		sys.exit(1)

	# Create list of categories based on the names of the file without suffixes.
	traincat = extractCategories(traindir,"train")
	testcat = extractCategories(testdir,"test")
	evalcat = extractCategories(evaldir,"test")

	# Run everything for test category
	print "# Running Semantic Textual Similarity for test data set"
	for category in testcat:
		print "## SemEval category : "+category+"\n"
		test_sentences, gold_standard_scores = read_data(category,"test")
		print "## Test dataset size : ",len(test_sentences),"\n"
		
		print "## Method 1"
		print "## SemEval 2015 rank 5 - DLS@CU-U"
		predicted_alignment_scores = map(get_alignment_score, test_sentences)
		print "## Pearson coefficient : ", pearsonr(gold_standard_scores, predicted_alignment_scores)[0]
		print "\n"

		print "## Method 2"
		print "## SemEval 2013 rank 1 - UMBC EBIQUITY-CORE"
		predicted_umbc_scores = map(get_umbc_score, test_sentences)
		print "## Pearson coefficient : ", pearsonr(gold_standard_scores, predicted_umbc_scores)[0]
		print "\n"

		print "## Method 3"
		print "## SemEval 2015 rank 1+3 - DLS@CU-S1 and DLS@CU-S2"

		if category not in traincat:
			print "## Pearson coefficient : 0\n"
		else:
			word_token_expr = re.compile(r'\w+')
			predicted_cosine_similarity_scores = map(get_cosine_similarity, test_sentences)

			# Get the training data
			ridge_training_sentences, ridge_gold_standard_scores = read_data(category, "train")
			print "## Train dataset size : ",len(ridge_training_sentences),"\n"

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
			print "## Pearson coefficient : ", pearsonr(gold_standard_scores, predicted_ridge_scores)[0],"\n"

if __name__ == "__main__": 
	main(sys.argv[1:])