import math
import copy

# path = 'C:\\Users\\miked\\OneDrive\\Desktop\\brown-corpus.txt' - This is here for my reference, please ignore

def load_corpus(path):
    
    corpus = []

    #start iterating through the lines of the training text
    for line in open(path, 'r'):
        
        i = 0

        #split into the individual pairs
        sentence = line.split()
        
        while i < len(sentence):
            #split the pairings into tuples of word and type
            sentence[i] = tuple(sentence[i].split('='))
            i+=1

        #add results of the sentence to the corpus
        corpus.append(sentence)

    return corpus
    

class Tagger(object):

    def __init__(self, sentences):

        #my smoothing constant
        self.smoothing = 1e-10

        self.tag_counts = {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0}
        
        self.init_counts = {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0}
        
        self.trans_probs = {'NOUN': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'VERB': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'ADJ': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'ADV': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'PRON': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'DET': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'ADP': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'NUM': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'CONJ': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'PRT': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            '.': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0},
                            'X': {'NOUN':0, 'VERB':0, 'ADJ':0, 'ADV':0, 'PRON':0, 'DET':0, 'ADP':0, 'NUM':0, 'CONJ':0, 'PRT':0, '.':0, 'X':0, 'TOTAL':0}}
        
        self.em_probs = {'NOUN': {'<UNK>':0},
                         'VERB': {'<UNK>':0},
                         'ADJ': {'<UNK>':0},
                         'ADV': {'<UNK>':0},
                         'PRON': {'<UNK>':0},
                         'DET': {'<UNK>':0},
                         'ADP': {'<UNK>':0},
                         'NUM': {'<UNK>':0},
                         'CONJ': {'<UNK>':0},
                         'PRT': {'<UNK>':0},
                         '.': {'<UNK>':0},
                         'X': {'<UNK>':0}}

        #start looking through our set of sentences
        for sentence in sentences:
            
            #for the first entry in each sentence, record the tag
            self.init_counts[sentence[0][1]] += 1
            self.init_counts['TOTAL'] += 1

            i = 0

            #go through each token
            while i < len(sentence):

                #total up the tags
                self.tag_counts[sentence[i][1]] += 1
                self.tag_counts['TOTAL'] += 1

                #if it is not the first entry, get the counts for ti -> tj
                if i != 0:
                    self.trans_probs[sentence[i][1]][sentence[i-1][1]] += 1
                    self.trans_probs[sentence[i][1]]['TOTAL'] += 1

                #count up how many times a given token appears as each part of speech
                if sentence[i][0] not in self.em_probs[sentence[i][1]]:
                    self.em_probs[sentence[i][1]][sentence[i][0]] = 1
                else:
                    self.em_probs[sentence[i][1]][sentence[i][0]] += 1

                i += 1

        #convert all counts into probabilities in log space here
        for key in self.init_counts:
            if key != 'TOTAL':
                self.init_counts[key] = math.log( (self.init_counts[key] + self.smoothing)/(self.init_counts['TOTAL'] + self.smoothing * (len(self.init_counts) - 1) ) )

        for key in self.trans_probs:
            for item in self.trans_probs[key]:
                if item != 'TOTAL':
                    self.trans_probs[key][item] = math.log( (self.trans_probs[key][item] + self.smoothing)/(self.trans_probs[key]['TOTAL'] + self.smoothing * (len(self.trans_probs[key]) - 1)) )

        for key in self.em_probs:
            for item in self.em_probs[key]:
                self.em_probs[key][item] = math.log( (self.em_probs[key][item] + self.smoothing)/(self.tag_counts[key] + self.smoothing * len(self.em_probs[key]) ) )

    def most_probable_tags(self, tokens):

        outputs = []

        #iterate through the tokens list
        for token in tokens:

            working_set = []
            
            for tag in self.em_probs:

                #create a set of all appearances of the given token
                if token in self.em_probs[tag]:
                    working_set.append( (self.em_probs[tag][token], tag) )
                #if it does not appear, use the unknown token
                else:
                    working_set.append( (self.em_probs[tag]['<UNK>'], tag) )

            #get the most probable and send to the output list
            maxed = max(working_set)
            outputs.append(maxed[1])

        return outputs

    def viterbi_tags(self, tokens):

        working_set = []
        output = []

        i = 0

        #iterate through tokens
        while i < len(tokens):

            #this will hold our probabilities for each token
            set_dict = {}

            #for the first token, we claculate based on init probabilities
            if i == 0:

                #maintain these for backtracking
                previous_state = {}
                current_set = []

                for tag in self.em_probs:

                    if tokens[i] in self.em_probs[tag]:

                        current_set.append( (self.init_counts[tag] + self.em_probs[tag][tokens[i]] , tag, 'END') )
                        previous_state[tag] = current_set[-1]
                        set_dict[tag] = ((current_set[-1][0], current_set[-1][2]))

                    else:

                        current_set.append( (self.init_counts[tag] + self.em_probs[tag]['<UNK>'] , tag, 'END') )
                        previous_state[tag] = current_set[-1]
                        set_dict[tag] = ((current_set[-1][0], current_set[-1][2]))

            else:
                
                #maintain these for backtracking (temp gives us a place to hold the new previous_state while we are still using the current one)
                temp = {}
                current_set = []

                for tag in self.em_probs:

                    prelim = []

                    #we will find the most optimal trasition and record both the probability and the previous tag
                    for item in previous_state:
                        prelim.append( (self.trans_probs[tag][item] + previous_state[item][0], item) )

                    if tokens[i] in self.em_probs[tag]:
                        maximum = max(prelim)
                        current_set.append( (maximum[0] + self.em_probs[tag][tokens[i]], tag, maximum[1]) )
                        temp[tag] = current_set[-1]
                        set_dict[tag] = ((current_set[-1][0], current_set[-1][2]))

                    else:
                        maximum = max(prelim)
                        current_set.append( (maximum[0] + self.em_probs[tag]['<UNK>'], tag, maximum[1]) )
                        temp[tag] = current_set[-1]
                        set_dict[tag] = ((current_set[-1][0], current_set[-1][2]))

                previous_state = copy.deepcopy(temp)
                
            working_set.append(copy.deepcopy(set_dict))
            i+=1


        i-=1
        nxt = ''

        #find the end state with the greatest probability and backtrack
        while nxt != 'END':
            
            if i == len(working_set) - 1:
                
                output.append(max(working_set[-1], key = working_set[-1].get))
                nxt = working_set[i][output[0]][1]

            else:
                output.insert(0, nxt)
                nxt = working_set[i][nxt][1]

            i-=1
        
        return output
