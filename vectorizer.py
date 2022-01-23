from argparse import ArgumentParser
import codecs
import csv
import io
import json
import nltk
import os
import pickle

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from file import save_json, save_sparse, save_array, \
    FILE_DATA_DOCS, FILE_DATA_X_TRAIN, FILE_DATA_X_TEST, \
    FILE_DATA_Y_TRAIN, FILE_DATA_Y_TEST, FILE_DATA_TARGETS, \
    save_json_pickle, FILE_DATA_VOC, load_json, DIR_VECTORIZED_DATA, \
    DIR_RAW_DATA, save_json_unicode
from models.model import ModelClf
from text.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
from text.mining import TextMiningTweepy
from text.sdCountVectorizer import SDCountVectorizer
from text.text import train_test_split, TextStats, preprocess_doc


def trainAndPickleGermanTagger():
    #corpus: http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/tigercorpus-2.2.conll09.tar.gz
    corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')
    
    tagged_sents = corp.tagged_sents()
    tagger = ClassifierBasedGermanTagger(train=tagged_sents)
    
    with open('nltk_german_classifier_data.pickle', 'wb') as f:
        pickle.dump(tagger, f, protocol=2)

def getGermanPOSTagger():
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_german_classifier_data.pickle')
    
    if not os.path.isfile(file_path):
        trainAndPickleGermanTagger()
    
    with open(file_path, 'rb') as f:
        tagger = pickle.load(f)
    return tagger

def main():
    print("Vectorizing tweets - load config.")
    
    """ PARSE ARGUMENTS """
    parser = ArgumentParser(description = "Preprocessing tool vectorize raw documents")
                
    parser.add_argument('--config', help = 'The configuration JSON')
                        
    args = parser.parse_args()
    
    cfg_file = os.path.join(os.path.dirname(__file__), *['config', 'vec', args.config])
    cfg = load_json(cfg_file)
    
    filepath_raw = os.path.join(DIR_RAW_DATA, cfg["filepath_raw_data"])
    target_input_file = cfg["target_input_files"]
    filepath_output = os.path.join(DIR_VECTORIZED_DATA, cfg["filepath_output"])
    ngram_range = tuple(cfg["n_gram_range"])
    targets = cfg["target_names"]
    dict_features_weight = cfg["dict_features_weight"]
    
    raw_docs = {}
    
    stop_words = ["ironie", "ironisch", "sarkasmus", "sarkastisch"]
    
    print("Done. Load raw documents")
    
    for target, input_file in target_input_file.items():
        input_file = os.path.join(filepath_raw, input_file)
                
        def unicode_csv_reader(utf8_data, delimiter=';'):
            csv_reader = csv.reader(utf8_data, delimiter=delimiter)
            for row in csv_reader:
                yield [unicode(cell, 'utf-8') for cell in row]
        
        docs = [preprocess_doc(row[4]) for row in unicode_csv_reader(open(input_file))]
        
        if "max_samples" in cfg:
            docs = docs[1:cfg["max_samples"]]
        
        raw_docs[int(target)] = docs
            
    """ CREATE TRAINING/TEST DATA """
    print("Done. Create training/test data")
    X_train, X_test, y_train, y_test = train_test_split(raw_docs=raw_docs)

    """ PROCESS DATA """ 
    print("Done. Vectorize documents")
    pos_tagger = getGermanPOSTagger()
    vectorizer = SDCountVectorizer(pos_tagger = pos_tagger, ngram_range=tuple(ngram_range), stop_words=stop_words)
    dict_vectorizer = DictVectorizer()
        
    feature_union = FeatureUnion(
        transformer_list=[
            ('vect', Pipeline([
                ('cnt', vectorizer),
                ('tfidf', TfidfTransformer())
            ])),
            ('text_stats', Pipeline([
               ('stats', TextStats(pos_tagger)),
               ('dict_feat', dict_vectorizer)
            ]))
        ],
        transformer_weights = {
            'vect' : 1.0,
            'text_stats' : dict_features_weight
        }
    )
    
    X_train = feature_union.fit_transform(X_train)
    X_test = feature_union.transform(X_test)
    
    voc = {k : int(v) for k,v in vectorizer.vocabulary_.items()}
    
    n = len(voc)
    for idx, feature in enumerate(dict_vectorizer.get_feature_names()):
        voc[feature] = n+idx
    
    """ SAVE """
    print("Done. Save results")
    save_sparse(filepath_output, FILE_DATA_X_TRAIN, X_train)
    save_sparse(filepath_output, FILE_DATA_X_TEST, X_test)
    save_array(filepath_output, FILE_DATA_Y_TRAIN, y_train)   
    save_array(filepath_output, FILE_DATA_Y_TEST, y_test)   
    save_json_unicode(filepath_output, FILE_DATA_VOC, voc)
    save_json(filepath_output, FILE_DATA_TARGETS, targets)
    
    print("Finished")
    
if __name__ == '__main__':
    main()