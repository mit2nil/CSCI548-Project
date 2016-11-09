import sys
sys.path.append('monolingual-word-aligner')
from glob import glob

from aligner import *
from collections import Counter
import re
import math

def read_training_file(filename):
	with open(filename) as f:
		return [x.strip().split("\t") for x in f.readlines()]

def read_training_gs(filename):
	with open(filename) as f:
		return [float(x) for x in f.readlines()]

def get_alignment_score(sentence_pair):
	alignment = align(sentence_pair[0].decode("utf-8").encode("ascii","ignore"), sentence_pair[1].decode("utf-8").encode("ascii","ignore"))
	align_count1 = len([x[0] for x in alignment[1] if x[0] not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	align_count2 = len([x[1] for x in alignment[1] if x[1] not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	content_count1 = len([x for x in alignment[2] if x not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	content_count2 = len([x for x in alignment[3] if x not in stopwords + punctuations + ['\'s', '\'d', '\'ll']])
	prop1 = align_count1/float(content_count1)
	prop2 = align_count2/float(content_count2)
	return (2*prop1*prop2)/(prop1 + prop2)

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

def read_training_data(category):
	files = [f for f in glob("data/*") if "train" in f]
	training_sentence_files = [f for train_dir in files for f in glob(train_dir+"/data/*")  if category in f]
	training_gs_files = [f for train_dir in files for f in glob(train_dir+"/gs/*") if category in f]
	training_sentences = [x for f in training_sentence_files for x in read_training_file(f)]
	gold_standard_scores = [x for f in training_gs_files for x in read_training_gs(f)]
	return training_sentences, gold_standard_scores
	
if __name__ == "__main__": 
	training_sentences, gold_standard_scores = read_training_data("c4")
	predicted_alignment_scores = map(get_alignment_score, training_sentences[:2])

	word_token_expr = re.compile(r'\w+')
	predicted_cosine_similarity_scores = map(get_cosine_similarity, training_sentences[:2])
	print predicted_alignment_scores, predicted_cosine_similarity_scores
