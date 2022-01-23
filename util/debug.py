'''
Created on May 20, 2018

@author: roman
'''

debug_enabled = False

class Debug(object):

    def __init__(self, params):
        '''
        Constructor
        '''
    
    def __call__(self):
        global debug_enabled
        debug_enabled = True
        print("Debug enabled")