from argparse import ArgumentParser
import os

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier

from text.text import train_test_split
from file import save_json, save_sparse, save_array, \
    FILE_DATA_DOCS, FILE_DATA_X_TRAIN, FILE_DATA_X_TEST, \
    FILE_DATA_Y_TRAIN, FILE_DATA_Y_TEST, FILE_DATA_TARGETS, \
    save_json_pickle, FILE_DATA_VOC, load_json
from text.mining import TextMiningTweepy


def main():
    """ argparse """
    parser = ArgumentParser(description = "Preprocessing tool for data mining with tweepy")
    
    parser.add_argument('--config', help = 'The configuration JSON, containing the output filename (output) and the search query (search_query)')
    
    args = parser.parse_args()
    
    """ mining """
    auth_file = os.path.join(os.path.dirname(__file__), *['config', 'auth', 'tweepy.json'])
    auth = load_json(auth_file)
    
    config_file = os.path.join(os.path.dirname(__file__), *['config', 'mine', args.config])
    config = load_json(config_file)
        
    miner = TextMiningTweepy(auth['consumer_key'], auth['consumer_secret'], 
                             auth['access_token'], auth['access_token_secret'])
    
    miner.initConnection()
    
    miner.mine_unlimited(config["search_query"], config["output"])
    
if __name__ == '__main__':
    main()