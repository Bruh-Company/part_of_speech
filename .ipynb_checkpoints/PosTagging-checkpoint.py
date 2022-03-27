import numpy as np
import sys
from tqdm.notebook import tqdm
import psutil
import ray

class PosTagging:
    
    def __init__(self, feeder):
        '''
        feeder as training data
        '''
        
        #constant
        self.__lambda = 0.7
        
        self.__feeder = feeder 
        self.emit = {}
        self.transition = {}
        self.context = {}
        self.unique_words = []
        self.count_words = 0
        self.unique_tags = []

#     def get_lambda(self):
#         return self.__lambda
       
    def set_lambda(self, value):
         self.__lambda = value
        
        
    def __ins_trans(self, previousTag, tag):
        if previousTag not in self.transition:
            self.transition[previousTag] = {}
        self.transition[previousTag][tag] = self.transition[previousTag].get(tag, 0) + 1
        
        
    def __ins_emit(self, tag, word):
        if tag not in self.emit:
            self.emit[tag] = {}
        self.emit[tag][word] = self.emit[tag].get(word, 0) + 1
        
        
    def __ins_ctx(self, tag):
        self.context[tag] = self.context.get(tag, 0) + 1

        
    def __evaluate_pos(self, sentence):
        '''
        evaluate statistical data from a sentence
        '''
        previous = "<s>"
        self.__ins_ctx(previous)

        for wordtag in sentence:
            word, tag = wordtag
            self.unique_words.append(word)
            
            self.__ins_trans(previous,tag)
            self.__ins_ctx(tag)
            self.__ins_emit(tag,word)

            previous = tag
            
        self.__ins_trans(previous, '</s>')

        
    def train(self):
        '''
        evaluate statistical data (words and tags)
        from every sentence in feeder
        '''
        lst = -1
        for idx,sentence in enumerate(self.__feeder):
            self.__evaluate_pos(sentence)
            
            prc = (idx + 1) * 50 // len(self.__feeder)
            if prc > lst :
                sys.stdout.write('[%s/%s]\t|%s%s|\r' % (idx+1, len(self.__feeder), prc * "█", (50 - prc) * "."))
                sys.stdout.flush()
                lst = prc
            
        sys.stdout.write('\nfinished...')
        
        self.unique_words = list(set(self.unique_words))
        self.count_words = len(self.unique_words)
        
        self.unique_tags = [tags for tags,_ in self.context.items()]
        
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
        Pe = lambda word, tag : self.__lambda * emissionProbability(word, tag) + (1 - self.__lambda) * (1 / self.count_words)
        Pt = lambda prevTag, tag : transitionProbability(prevTag, tag)
        
        # preparation
        [words, labels] = np.array(sentence).T
        N = len(words)
        
        score = {}
        best_trans = [{} for _ in range(N + 2)]
        
        for tag in self.unique_tags:
            score[tag] = float('inf')
        score['<s>'] = 0
        
        # relation checker
        exists = lambda prv, nxt :\
            prv in self.transition and nxt in self.transition[prvTag]
        
        # check for every sentence edge
        for i in range(N):
            # initialize every best score from a tag with infinite
            nscore = {}
            for tag in self.unique_tags:
                nscore[tag] = float('inf')
            
            # iterate over every relation on the sentece
            for prvTag in self.unique_tags:
                for nxtTag in self.unique_tags:
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
                
            transProb = self.transition[prvTag].get('</s>', 0) / self.__feeder.shape[0]
            
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
    
    def evaluate(self, datum, label):
        tags = self.predict(datum)
        return (tags == label).sum()
        
    def get_tag_accuracy(self, data, labels, hide=True):
        cnt = np.array([data[i].shape[0] for i in range(len(data))]).sum()
        right = 0
        #Iterasi untuk per kalimat
        # (tqdm(iterator) if verbose else iterator):
        print("BRUH")
        for i in tqdm(range(len(data)), disable=hide):
            right += self.evaluate(data[i], labels[i])
        return right / cnt
    
    def validate(self, validation_data, step_up):
        ray.init(num_cpus=psutil.cpu_count(logical=False), ignore_reinit_error=True)
        '''
        caranya chen
        '''
        self.__lambda = 0.0
        lambda_values = []
        # Gets all the labes for each sentences (by transposing)
        labels = [data.T[1] for data in validation_data]
            
        for i in tqdm(range(int(np.reciprocal(step_up)))):
            lambda_values.append((self.__lambda, self.get_tag_accuracy(validation_data, labels)))
            self.__lambda += step_up
        
        ray.shutdown()
        return lambda_values
    def test(self, testing_data, __lambda):
        labels = [data.T[1] for data in testing_data]
        self.__lambda = __lambda
        return self.get_tag_accuracy(testing_data, labels, hide=False)

    def validate_with_ternary_search(self, validation_data):
        '''
        ternary search
        '''
        pass