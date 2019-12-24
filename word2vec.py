from argparse import ArgumentParser
import logging
import time
from pathlib import Path
from gensim.models import Word2Vec

def validate_path(path):
    if not path.is_dir():
        logger.error('{} is not a valid directory'.format(path))
        exit(123)

def vocab_builder(sentences, size=100, window=5, min_count=5, workers=1, sg=0, sample=0.001):
    # train model
    trained_model = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg, sample = sample)
    return trained_model
    
if __name__ == '__main__':
    #text = list of tokenized sentences [['Text','for','word2vec'],['Hi','this','is','sarthak']]
    model = vocab_builder(sentences, min_count=1)
    #access vector for one word
    print(model['Text'])
    word_embedding_save(model, 'model.bin')
