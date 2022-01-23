from argparse import ArgumentParser, Action

def getArguments():
    parser = ArgumentParser(description = "Mining sarcarstic and ironic tweets to create a sarcarsm"
                                        + " and irony detection machine learning model. What a great idea!")
    
    """ Twitter key and access tokens """
    parser.add_argument('--consumer_key', help = 'Consumer Key (API Key)')
    parser.add_argument('--consumer_secret', help = 'Consumer Secret (API Secret)')
    parser.add_argument('--access_token', help = 'Access Token')
    parser.add_argument('--access_token_secret', help = 'Access Token Secret')
        
    parser.add_argument('--max_docs', help='maximum number of Documents mined', type=int)
        
    parser.add_argument('--ngram_range', default=[1,1], help='The lower and upper boundary of the range of n-values for different n-grams to be extracted')
            
    parser.add_argument('--output', help = 'Name of the directory where the output is saved')
        
    return parser.parse_args()