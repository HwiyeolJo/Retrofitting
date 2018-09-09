from __future__ import print_function
import gzip
import math
import numpy as np
import re
from copy import deepcopy
import h5py
from tqdm import tqdm_notebook

### Hyperparameter
numIters = 10
WordDim = 300
#####

def read_word_vecs(filename):
    print("Vectors read from", filename)
    wordVectors = {}
    if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
    elif filename.endswith('.h5'):
        fileObject_ = h5py.File(filename, 'r')
        fileObject = fileObject_[list(fileObject_.keys())[0]]
        wordlist = np.array(fileObject[u'axis1'])
        pbar = tqdm_notebook(total = len(wordlist))
        unk_counter = 0
        for i, w in enumerate(wordlist):
            pbar.update(1)
            wordVectors[w[6:]] = fileObject[u'block0_values'][i]
        pbar.close()
    else:
        fileObject = open(filename, 'r')
        fileObject.readline() # For handling First Line
        for line in fileObject:
            line = line.strip().lower()
            word = line.split()[0]
            wordVectors[word] = np.zeros(len(line.split())-1, dtype=np.float64)
            vector = line.split()[1:]
            for index, vecVal in enumerate(vector):
                wordVectors[word][index] = float(vecVal)
            wordVectors[word] = wordVectors[word] / math.sqrt((wordVectors[word]**2).sum() + 1e-5)
    return wordVectors


isNumber = re.compile(r'\d+')
def norm_word(word): # Could Add Preprocessing
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()
    
def read_lexicon(filename, wordVecs):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

def retrofit(wordVecs, lexicon, numIters):
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    print(len(loopVocab), "words will be retrofitted")
    pbar = tqdm_notebook(total = numIters, desc = 'Epoch')
    for it in range(numIters):
        pbar.update(1)
        for word in loopVocab:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
            if numNeighbours == 0:
                continue
            ### Retrofitting
            newVec = numNeighbours * wordVecs[word]
            for ppWord in wordNeighbours:
                newVec = newVec + newWordVecs[ppWord]
            newWordVecs[word] = newVec/(2*numNeighbours)
            #####
    pbar.close()
    return newWordVecs

def print_word_vecs(wordVectors, outFileName):
    print('Writing down the vectors in', outFileName)
    outFile = open(outFileName, 'w')
    outFile.write(str(len(wordVectors)) + ' ' + str(WordDim) + '\n')
    pbar = tqdm_notebook(total = len(wordVectors), desc = 'Writing')
    for word, values in wordVectors.iteritems():
        pbar.update(1)
        outFile.write(word+' ')
        for val in wordVectors[word]:
            outFile.write('%.5f' %(val)+' ')
        outFile.write('\n')
    outFile.close()
    pbar.close()
    
wordVecs = read_word_vecs("") # Pretrained word vector
lexicon = read_lexicon("", wordVecs) # Choosing Semantic Lexicon
wordVecs_retro = retrofit(wordVecs, lexicon, numIters) # Running Retrofitting
print_word_vecs(wordVecs_retro, "") # Printing the output
