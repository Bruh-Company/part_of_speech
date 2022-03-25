import numpy as np
import sys

class PosTagging:
    def __init__(self, feeder):
        '''
        feeder as training data
        '''
        
        #constant
        self.__lambda = 0.7
        
        self.feeder = feeder 
        self.emit = {}
        self.transition = {}
        self.context = {}
        self.uniqueWords = []
        self.countWords = 0
        self.uniqueTags = []

    def get_lambda(self):
        return self.__lambda
       
    def set_lambda(self, value):
         self.__lambda = value
        
    def insertTransition(self, previousTag, tag):
        if previousTag not in self.transition:
            self.transition[previousTag] = {}
        self.transition[previousTag][tag] = self.transition[previousTag].get(tag, 0) + 1
        
    def insertEmit(self, tag, word):
        if tag not in self.emit:
            self.emit[tag] = {}
        self.emit[tag][word] = self.emit[tag].get(word, 0) + 1
        
    def insertContext(self, tag):
        self.context[tag] = self.context.get(tag, 0) + 1

        
    def evaluatePOS(self, sentence):
        '''
        evaluate statistical data from a sentence
        '''
        previous = "<s>"
        self.insertContext(previous)

        for wordtag in sentence:
            word, tag = wordtag
            self.uniqueWords.append(word)
            #Transition
            self.insertTransition(previous,tag)

            # Context
            self.insertContext(tag)

            #Emit
            self.insertEmit(tag,word)

            previous = tag
            
        self.insertTransition(previous, '</s>')

        
    def evaluateFeed(self):
        '''
        evaluate statistical data (words and tags)
        from every sentence in feeder
        '''
        lst = -1
        for idx,sentence in enumerate(self.feeder):
            self.evaluatePOS(sentence)
            
            prc = (idx + 1) * 50 // len(self.feeder)
            if prc > lst :
                sys.stdout.write('[%s/%s]\t|%s%s|\r' % (idx+1, len(self.feeder), prc * "█", (50 - prc) * "."))
                sys.stdout.flush()
                lst = prc
            
        sys.stdout.write('\nfinished...')
        
        self.uniqueWords = list(set(self.uniqueWords))
        self.countWords = len(self.uniqueWords)
        
        self.uniqueTags = [tags for tags,_ in self.context.items()]
        
        return self.emit, self.transition, self.context

    
    def predict(self, sentence):
        '''
        computing best chain of tag from a sentence based on computed probabilistic
        using dynamic programming method
        '''
        
        '''
        forward step, compute best desicion for each tags
        '''
        # probabilistic functon
        emissionProbability =\
            lambda word, tag :\
                self.emit[tag].get(word,0) / self.context[tag]
        transitionProbability =\
            lambda previousTag, tag :\
                self.transition[tag].get(previousTag, 0) / self.context[tag]
        # smoothed
        Pe = lambda word, tag : self.__lambda * emissionProbability(word, tag) + (1 - self.__lambda) * (1 / self.countWords)
        Pt = lambda prevTag, tag : transitionProbability(prevTag, tag)
        
        # preparation
        [words, labels] = np.array(sentence).T
        N = len(words)
        
        score = {}
        best_trans = [{} for _ in range(N + 2)]
        
        for tag in self.uniqueTags:
            score[tag] = float('inf')
        score['<s>'] = 0
        
        # relation checker
        exists = lambda prv, nxt :\
            prv in self.transition and nxt in self.transition[prvTag]
        
        # check for every sentence edge
        for i in range(N):
            # initialize every best score from a tag with infinite
            nscore = {}
            for tag in self.uniqueTags:
                nscore[tag] = float('inf')
            
            # iterate over every relation on the sentece
            for prvTag in self.uniqueTags:
                for nxtTag in self.uniqueTags:
                    '''
                    brute force, checking every possible relation
                    (there's exists an optimization technique but we'll try to implement it later)
                    '''
                    if exists(prvTag, nxtTag):
                        '''
                        score[nxtTag] = score[prvTag] - log(Pt(prvTag | nxtTag)) - log(Pe(nxtTag | word))
                        '''
                        prob = np.log(Pt(nxtTag, prvTag)) + np.log(Pe(words[i], nxtTag))
                        value = score.get(prvTag) - prob

                        if nscore[nxtTag] > value:
                            nscore[nxtTag] = value
                            best_trans[i + 1][nxtTag] = prvTag
            score = nscore
            
        listScore = {}
        
        for prvTag, val in score.items():
            if val >= float('inf') :
                continue
                
            transProb = self.transition[prvTag].get('</s>', 0) / self.feeder.shape[0]
            
            if transProb == 0 :
                listScore[best_trans[-2][prvTag]] = float('inf')
            else :
                listScore[best_trans[-2][prvTag]] = score[prvTag] + (-np.log(transProb))

        best_trans[-1]['</s>'] = min(listScore, key=listScore.get)
        
        '''
        backward step
        '''
        tags = ["</s>"]
        pos = N = len(sentence)
        
        while pos >= 0:
            tags.append(best_trans[pos + 1][tags[N - pos]])
            pos -= 1
        tags = tags[::-1]
        return tags[1:-1]
        

