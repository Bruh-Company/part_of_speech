import numpy as np

class PosTagging:
    def __init__(self, feeder):
        # Feeder itu Array of Kalimat
        
        #constant
        self.__lambda = 0.7
        
        self.feeder = feeder 
        self.emit = {}
        self.transition = {}
        self.context = {}
        self.uniqueWords = []
        self.countWords = 0

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
        for sentence in self.feeder:
            self.evaluatePOS(sentence)
        self.uniqueWords = list(set(self.uniqueWords))
        self.countWords = len(self.uniqueWords)
        # print(self.uniqueWords)
        # print(self.countWords)
        return self.emit, self.transition, self.context

    def smoothEmissionProbability(self, word, tag):
        # print('SMOOTH:',(1-__lambda) * (1/self.countWords))
        return (self.__lambda*self.emissionProbability(word, tag)) + ((1-self.__lambda) * (1/self.countWords))

    def emissionProbability(self, word, tag):
        if word not in self.emit[tag]:
            # print("EMIT IS 0")
            return 0
        # print('EMIT:',self.emit[tag].get(word),self.emit[tag].get(word) / self.context[tag])
        return self.emit[tag].get(word,0) / self.context[tag]

    def transitionProbability(self, previousTag, tag):
        # print('TRANSITIONS:', self.transition[previousTag].get(tag,0),self.transition[previousTag].get(tag,0) / self.context[tag])
        return self.transition[tag].get(previousTag,0) / self.context[tag]

    def forwardStep(self, sentence):
        # sentence = np.insert(sentence, 0, ['<s>','<s>'], axis=0)
        # sentence = np.insert(sentence, sentence.shape[0], ['</s>','</s>'], axis=0)
        
        words = [x[0] for x in sentence]
        labels = [x[1] for x in sentence]
        
        best_score = [{} for i in range(len(words)+2)]
        best_edge = [{} for i in range(len(words)+2)]
        
        best_score[0]['<s>'] = 0
        best_edge[0]['<s>'] = None
        
        for i in range(len(words)):
            for prevTag, val1 in self.context.items():
                for nextTag, val2 in self.context.items():
                    
                    if prevTag in best_score[i] and prevTag in self.transition and nextTag in self.transition[prevTag]:
                        
                        prob = (-np.log(self.transitionProbability(nextTag, prevTag))) + (-np.log(self.smoothEmissionProbability(words[i], nextTag)))
                        score = best_score[i].get(prevTag) + prob

                        if nextTag not in best_score[i+1] or best_score[i+1][nextTag] > score:
                            best_score[i+1][nextTag] = score
                            best_edge[i+1][nextTag] = prevTag
        
        listScore = {}
        for prevTag in best_score[-2]:
            transProb = self.transition[prevTag].get('</s>',0) / self.feeder.shape[0]
            value = best_score[-2][prevTag] + (-np.log(transProb))
            listScore[best_edge[-2][prevTag]] = value

        best_key = min(listScore, key=listScore.get)
        best_score[-1]['</s>'] = listScore[best_key]
        best_edge[-1]['</s>'] = best_key
        return best_score, best_edge

    def backwardStep(self, best_edge, sentence):
        tags = ["</s>"]
        tag = next_edge = best_edge[-1]['</s>']
        pos = len(sentence) + 1
        while pos != 0 and tag != '<s>':
            tags.append(tag)
            tag = best_edge[int(pos) - 1][tag]
            pos -= 1
        words = (x[0] for x in sentence)

        return tags[::-1]
        

