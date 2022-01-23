

class Report(object):
    
    def __init__(self):
        self.txt = ""    
    
    def addSeperator(self):
        self.addLn("=" * 80 )
        
    def addLn(self, text):
        self.add(text + "\n")
    
    def add(self, text):
        self.txt += text
        
    def getText(self):
        return self.txt
        
    