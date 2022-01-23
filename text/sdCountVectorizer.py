from operator import itemgetter
import re
import six

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from text import REGEX_USERNAMES, REGEX_HASHTAGS


def get_feature_names(voc):
    return np.array([t for t, i in sorted(six.iteritems(voc), key=itemgetter(1))])

class SDCountVectorizer(CountVectorizer):
        
    def __init__(self, pos_tagger, **kwds):
        super(SDCountVectorizer, self).__init__(**kwds)
        
        self.pos_tagger = pos_tagger
    
    def tokenize(self, doc):
        """
        Performs additional operations:
            - remove usernames
            - remove hashtags
        """        
        doc = re.sub(pattern=REGEX_USERNAMES, repl="", string=doc, flags=re.MULTILINE)
        
        #doc = re.sub(pattern=REGEX_HASHTAGS, repl="", string=doc, flags=re.MULTILINE)
        
        token_pattern = re.compile(self.token_pattern)
        return token_pattern.findall(doc)
    
    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        return self.tokenize
                
    def _word_ngrams(self, tokens, stop_words=None):                
        #print(tokens)        
        #TODO hashtags expressing emotions
                
        return super(SDCountVectorizer, self)._word_ngrams(tokens, stop_words)
    