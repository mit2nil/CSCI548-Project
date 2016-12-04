# CSCI548-Project

#### Installation Instructions.

* Install python 2.7.x, pip, virtualenv, git etc.
* git clone https://github.com/mit2nil/CSCI548-Project.git
* cd CSCI548-Project
* virtualenv --no-site-packages <pyenv2>
* source pyenv2/bin/activate
* pip install -r packages.txt

##### Setting up word aligner

* git clone git://github.com/dasmith/stanford-corenlp-python.git
* cd stanford-corenlp-python
* Replace below line in the stanford-corenlp-python/corenlp.py with line below it.
* ``` rel, left, right = map(lambda x: remove_id(x), split_entry) ``` with 
* ``` rel, left, right = split_entry ```
* wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip
* unzip stanford-corenlp-full-2014-08-27.zip
* Run server by ```python corenlp.py```
* open another terminal in the same directory 
* cd <path>/CSCI548-Project
* python -m nltk.downloader stopwords
* add below lines in the monolingual-word-aligner/wordSim.py inside function loadPPDB
``` import os 
	ppdbFileName = os.path.join(os.path.dirname(os.path.realpath(__file__)),ppdbFileName) ```
* Run ```python clientAlignScore.py```