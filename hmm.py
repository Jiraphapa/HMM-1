# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from classifier import CodeBook
from corpus import Document

class HMM(Classifier):
    """A Hidden Markov Model classifier."""
    def __init__(self):
        '''Init the HMM ,create model'''
        self.HmmModel=HMM_model()
    def get_model(self): return self.HmmModel
    def set_model(self, model): self.HmmModel=model
    model = property(get_model, set_model)
    def train(self,sentence,initial_probabilities=[],transition_probabilities=[],
              emission_probabilities=[],states=(),vocabulary=[]):#any better way to do that?
        '''Train the model in the HMM, change value in model in place. States are stored are represented as ID'''
        
        if len(sentence)==0:#just for testing the IceCream problem
            self.HmmModel.initial_probabilities=initial_probabilities
            self.HmmModel.transition_probabilities=transition_probabilities
            self.HmmModel.emission_probabilities=emission_probabilities
            self.HmmModel.states=states
            for name in states:
                self.HmmModel.statesbook.add(name)
            self.HmmModel.vocabulary=vocabulary
            for word in vocabulary:
                self.HmmModel.vocabularybook.add(word)
            for i in range(len(states)):
                self.HmmModel.final_transition_probabilities.append(1)
        else:               #extract data from sentence here
            for word in sentence:
                pass
            
    def likelihood(self,Document):
        '''Given the observed array, calculate the likelihood of the each state, 
        and store the likelihood in the likelihood_matrix,return the maximal likelihood'''  
        score=0
        observed_array=Document.features()
        Time=len(observed_array)# the number of time step
        State_num=len(self.HmmModel.states)#how many states in the Hmm model
        forward=[0]*State_num
        forward_old=[0]*State_num
        #initial the vector
        ob1=self.HmmModel.vocabularybook.get(observed_array[0])#The id of first state 
        for i in range(State_num):
            forward_old[i]=self.HmmModel.initial_probabilities[i]*self.HmmModel.emission_probabilities[i][ob1]#a0,s*bs(ob1)
        for time in range(1,Time):
            ob=self.HmmModel.vocabularybook.get(observed_array[time])
            for i in range(State_num):
                for n in range(State_num):
                    forward[i]+=forward_old[n]*self.HmmModel.transition_probabilities[n][i]*self.HmmModel.emission_probabilities[i][ob]
            forward_old=forward
            forward=[0 for count in range(State_num)]
        #get final transition
        for i in range(State_num):
            score+=forward_old[i]*self.HmmModel.final_transition_probabilities[i]     
        return score
    def classify(self,Document,Test=False,T=0):
        '''Using viterbi algorithm to tag the state of the observed array,return a state array'''
        final_score=0
        back_trace_final=0
        path=[]
        observed_array=Document.features()
        Time=len(observed_array)
        State_num=len(self.HmmModel.states)
        back_matrix=[ [0 for n in range(State_num)] for i in range(Time)]#to record to back trace path
        vertibi=[0]*State_num
        vertibi_old=[0]*State_num
        ob=self.HmmModel.vocabularybook.get(observed_array[0])
        for i in range(State_num):
            vertibi_old[i]=self.HmmModel.initial_probabilities[i]*self.HmmModel.emission_probabilities[i][ob]
            back_matrix[0][i]=-1
        for time in range(1,Time):
            ob=self.HmmModel.vocabularybook.get(observed_array[time])
            for i in range(State_num):
                for n in range(State_num):
                    if vertibi_old[n]*self.HmmModel.transition_probabilities[n][i]*self.HmmModel.emission_probabilities[i][ob]>vertibi[i]:
                        vertibi[i]=vertibi_old[n]*self.HmmModel.transition_probabilities[n][i]*self.HmmModel.emission_probabilities[i][ob]
                        back_matrix[time][i]=n
            
            vertibi_old=vertibi
            vertibi=[0 for count in range(State_num)]
            if Test==True and time==T:#output the test result
                return vertibi_old
        #get the final transition
        for i in range(State_num):
            if vertibi_old[i]*self.HmmModel.final_transition_probabilities[i]>final_score:
                final_score=vertibi_old[i]*self.HmmModel.final_transition_probabilities[i]
                back_trace_final=i
        #Making the path
        uper_layer=back_trace_final
        state=self.HmmModel.statesbook.name(uper_layer)
        path.append(state);
        for time in range(Time-1,0,-1):
            uper_layer=back_matrix[time][uper_layer]
            state=self.HmmModel.statesbook.name(uper_layer)
            path.append(state);
        return path
  


class HMM_model():
    def __init__(self):
        self.statesbook=CodeBook([])
        self.vocabularybook=CodeBook([])
        self.initial_probabilities=[]
        self.transition_probabilities=[]
        self.emission_probabilities=[]
        self.states=()
        self.final_transition_probabilities=[]
        self.vocabulary=[]