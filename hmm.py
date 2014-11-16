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
        #get final transition
        for i in range(State_num):
            score+=forward_old[i]*self.HmmModel.final_transition_probabilities[i]     
        return score
    def classify(self,test_sentence):
        '''Using Viterbe algorithm to tag the state of the observed array,return a state array'''
        state=[]
        return state
  


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