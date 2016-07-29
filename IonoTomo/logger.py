
# coding: utf-8

# In[ ]:

from sys import stdout

class logger(object):
    def __init__(self,logFile=None):
        if logFile is not None:
            try:
                self.file = open(logFile,"a+")
                self.logFile = logFile
            except:
                print("Failed to open log file {0}".format(logFile))
        else:
            self.file = stdout
    def __exit__(self):
        try:
            self.file.close()
        except:
            pass
    def log(self,message):
        self.file.write("{0}\n".format(message))
        self.file.flush()

