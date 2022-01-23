from collections import Counter
import random
import nltk
import os
import pickle
import re

from sklearn.base import BaseEstimator, TransformerMixin

#from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

REGEX_USERNAMES = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"

REGEX_HASHTAGS = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)"

REGEX_URLS = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S"

#list taken from https://www.sketchengine.eu/german-stts-part-of-speech-tagset/
POS_TAGS_CONSIDERED = [
    ('ADJ_ADV', ['ADJA', 'ADJD', 'ADV']),
    ('ITJ', ['ITJ']),
    ('NN', ['NA', 'NN', 'NE']),
    ('VV', ['VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP'])
]

LAUGHING_ACRONYMS = [
    'lol',
    'lawl',
    'lul',
    'rofl',
    'roflmao',
    'lmao',
    'lmfao',
    '*grins*',
    '*gg*',
    '*g*',
    'haha',
    'hehe',
    'hihi'
]

PUNCTUATION = [
    '...',
    '??',
    '!?',
    '?!',
    '!!',
    'einself',
    '1elf'
]

SMILEY_POS = [':)', ':-)', ':D', ':-D', '^^', '=D', ';)', ';-)']
SMILEY_NEG = [':(', ':-(', ':|', ':-|', '>:(', '>:-(', 'X(', 'X-(']


def preprocess_doc(doc):
    return re.sub(pattern=REGEX_URLS, repl="", string=doc, flags=re.IGNORECASE)

def train_test_split(raw_docs, test_size=.25, shuffle=True):
    
    train = []
    test = []
    
    for target, docs in raw_docs.items():
        split = int(round(len(docs) * (1-test_size)))
        train += [(doc, target) for doc in docs[:split]]
        test += [(doc, target) for doc in docs[split:]]

    if shuffle:
        random.shuffle(train)
        random.shuffle(test)

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return X_train, X_test, y_train, y_test

class TextStats(BaseEstimator, TransformerMixin):
    
    def __init__(self, pos_tagger):
        
        self.pos_tagger = pos_tagger

    def fit(self, X, y=None):
        return self

    def transform(self, docs):
        
        pos_tagger = self.pos_tagger
        
        stats = []
        
        for doc in docs:
            stats_doc = {}
            doc = doc.replace('@', '')
            doc = doc.replace('#', '')
            
            capitalized = 0
            tags = []
            
            pattern = re.compile('[a-zA-Z]')
            
            """ POS TAGGING """
            for token, tag in pos_tagger.tag(doc.split(' ')):
                tags.append(tag)
                if pattern.match(token) and token.upper() == token:
                    capitalized += 1
                
            #tags = [tag for token, tag in pos_tagger.tag(doc.split(' '))]
            pos_cnt = Counter(tags)
            
            for tag, tag_list in POS_TAGS_CONSIDERED:
                stats_doc[tag] = sum([pos_cnt[t] for t in tag_list])
                            
            stats_doc['CAPITALIZED'] = capitalized
            
            stats_doc['SMILEY_POS'] = 1 if any(p in doc for p in SMILEY_POS) else 0
            stats_doc['SMILEY_NEG'] = 1 if any(p in doc for p in SMILEY_NEG) else 0
            
            stats_doc['LAUGHING_ACRONYMS'] = 1 if any(la in doc for la in LAUGHING_ACRONYMS) else 0
            stats_doc['PUNCTUATION'] = 1 if any(p in doc for p in PUNCTUATION) else 0
            
            
            stats.append(stats_doc)
        
        self.stats = stats
        
        return stats
        
class Document(object):
        
    def __init__(self, tweet, target):
        self.tweet = tweet
        self.text = self.process(tweet)
        self.target = target
        
    def getText(self):
        return self.text
    
    def getTarget(self):
        return self.target
    
    def process(self, tweet):
        text = None
        
        if hasattr(tweet, 'text'):
            text = tweet.text
        elif hasattr(tweet, 'full_text'):
            text = tweet.full_text
        
        return text

    
    