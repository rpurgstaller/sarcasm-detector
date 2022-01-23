import random
import sys

import jsonpickle
import tweepy

from text import Document

class TextMiningBase(object):
    
    def __init__(self):
        self.docs = {}
        
    def getDocs(self):
        return self.docs
    
    def initConnection(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def mine(self, search_query, count, target):
        raise NotImplementedError("Subclass must implement abstract method")
    
    
class TextMiningTweepy(TextMiningBase):
    
    TWEETS_PER_QUERY = 100
    
    LANGUAGE = 'de'
    
    MAX_TWEETS = 1e5
    
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        TextMiningBase.__init__(self)
        
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

    def initConnection(self):
        auth = tweepy.OAuthHandler(consumer_key=self.consumer_key, consumer_secret=self.consumer_secret)
        auth.set_access_token(key=self.access_token, secret=self.access_token_secret)
        
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True)
        
        if (not api):
            print("ERROR: Unable to authenticate")
            sys.exit(-1)
        
        self.api = api            
    
    def execute(self, targets, search_queries, doc_count):
        docs = self.docs
        for t_name, t_value in targets.iteritems():
            docs[t_value] = []
            for tweet in tweepy.Cursor(self.api.search, q=search_queries[t_name], 
                                       lang=TextMiningTweepy.LANGUAGE, tweet_mode='extended').items(doc_count):
                    #print("[" + str(tweet.retweeted) + "] - " + self.api.get_user(tweet.author.id_str).screen_name + " - " + tweet.id_str + " - " + tweet.text)
                docs[t_value].append(Document(tweet=tweet, target=t_value))        
        
        return docs
    
    def mine_unlimited(self, search_query, file_name):
        
        api = self.api
        
        tweet_count = 0
        max_id = 1L
        since_id = None        
        
        with open(file_name, 'w') as f:
            while tweet_count < TextMiningTweepy.MAX_TWEETS:
                try:
                    if max_id <= 0:
                        if not since_id:
                            new_tweets = api.search(q = search_query, count = TextMiningTweepy.TWEETS_PER_QUERY)
                        else:
                            new_tweets = api.search(q = search_query, count = TextMiningTweepy.TWEETS_PER_QUERY,
                                                    since_id = since_id)
                    else:
                        if not since_id:
                            new_tweets = api.search(q = search_query, count = TextMiningTweepy.TWEETS_PER_QUERY,
                                                    max_id = str(max_id - 1))
                        else:
                            new_tweets = api.search(q = search_query, count = TextMiningTweepy.TWEETS_PER_QUERY,
                                                    max_id = str(max_id - 1), since_id = since_id)
                    if not new_tweets:
                        print("No more tweets found")
                        break
                    for tweet in new_tweets:
                        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                    tweet_count += len(new_tweets)
                    print("Downloaded {0} tweets".format(tweet_count))
                    max_id = new_tweets[-1].id
                except tweepy.TweepError as e:
                    print("error while mining: %s" % str(e))
        
        print ("Downloaded {0} tweets, Saved to {1}".format(tweet_count, file_name))
        
        
        
        
        
        
        
        
        
        
        
        
    