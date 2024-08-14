import numpy as np
import pandas as pd

#from sklearn.hmm import GMMHMM 

NoOfStates = 4   
NoOfObservations = 0   # T, couninous observations
TrainingData = []
TestData = []
M = 3

A = None #transition_probability_matrix
Pi = None #initial probability distribution
alpha = None
beta = None
mu = []   #states * M
sigma = []

def initialize():
    global A,Pi,NoOfStates, NoOfObservations, mu, sigma
    df = pd.read_csv ('./data/Corporation4.csv', header = None)
    df.columns=['Close','Open','High','Low']

    df = df.iloc[::-1]
    df.reset_index(inplace = True, drop = True)

    X_test = df.iloc[2320:]
    X_train = df.iloc[:2320]

    close_train = X_train['Close']
    close_test = X_test['Close']

    NoOfStates = 4
    Pi = np.ones(NoOfStates) / NoOfStates
    A = np.ones([NoOfStates,NoOfStates]) /  (NoOfStates*NoOfStates)
    mu = np.ones(NoOfStates)
    sigma = np.ones(NoOfStates)*10
    NoOfObservations = 2320
    return close_train, close_test

def B(state,observation, debugging=0):   #GMM by means of לףא density function    
    global M
    sum = 0
    for i in range(1,M):
        sum = sum + ( 1/np.sqrt(2*3.14*sigma[state]*sigma[state]) )* np.exp(- (observation-mu[state])*(observation-mu[state]) / (2*sigma[state]*sigma[state]) )
    if(debugging==1):
        print("B")
        print(sum, M)
    return sum/M

def forwardAlgorithm(ObservationSeq, test =0):
    global A,B,NoOfStates,Pi, alpha
    alpha = np.zeros((NoOfStates,len(ObservationSeq)))
#    if(test==1):
#        print(alpha)
#        print(NoOfStates)
    for s in range(NoOfStates):        
        alpha[s,0] = Pi[s] * B(s,ObservationSeq[0])
    for t in range(1,len(ObservationSeq)):
        for s in range(NoOfStates):
            alpha[s,t] = np.sum(alpha[:,t-1]* A[s,:]* B(s,ObservationSeq[t]))
    return np.sum(alpha[:,-1])

def backwardAlgorithm(ObservationSeq):
    global A,B,NoOfStates,Pi, beta
    beta = np.zeros((NoOfStates,len(ObservationSeq)))
    for s in range(NoOfStates):
        beta[s,-1] = 1
    for t in range(len(ObservationSeq)-2,-1,-1):
        for s in range(NoOfStates):
            beta[s,t] = np.sum(beta[:,t+1]* A[s,:]* B(s,ObservationSeq[t+1]))
    return np.sum(beta[:,-1])

def viterbiAlgorithm(ObservationSeq):
    global A,B,NoOfStates,Pi
    viterbi = np.zeros((NoOfStates,len(ObservationSeq)))
    p = np.zeros((NoOfStates,len(ObservationSeq)))
    for s in range(NoOfStates):
        viterbi[s,0] = Pi[s] * B(s,ObservationSeq[0])
    print(viterbi)
    for t in range(1,len(ObservationSeq)):
        for s in range(NoOfStates):
            max_s_prim = 0
            max_prob = 0
            for s_prim in range(NoOfStates):
                prob = viterbi[s_prim,t-1]*A[s_prim,s]*  B(s,ObservationSeq[t])
                if prob > max_prob:
                    print(prob)
                    max_prob = prob
                    max_s_prim = s_prim
            viterbi[s,t] = max_prob
            p[s,t] = max_s_prim
    bestpathprob = np.max(viterbi[:,-1])
    bestpathpointer = np.argmax(viterbi[:,-1])
    path = [bestpathpointer]
    pointer = bestpathpointer
    for t in range(len(ObservationSeq)-1,0,-1):
        pointer = p[int(pointer),t]
        path.insert(0,pointer)
    return path,bestpathprob
        

def forwardBackwardAlgorithm(iteration_num):
    global NoOfStates,NoOfObservations,TrainingData,TestData, A,B,observation_probability,alpha,beta,gamma,kisi
    
    T = len(TrainingData)
    TrainingData = np.array(TrainingData)
    A = 1 / NoOfStates * np.ones((NoOfStates,NoOfStates)) 
#    B = 1 / NoOfObservations * np.ones((NoOfStates,NoOfObservations))
    
    gamma = np.zeros((NoOfStates,T))
    kisi = np.zeros((NoOfStates,NoOfStates,T))
 
    for iterate_num in range(iteration_num):
        #start = time.time()
        forwardAlgorithm(TrainingData)
        backwardAlgorithm(TrainingData)
        observation_probability = np.sum(alpha[:,T-1])

        for t in range(len(TrainingData)-1): 
            for j in range(NoOfStates):
                gamma[j,t]= alpha[j,t]*beta[j,t]/observation_probability
        
        
        for t in range(len(TrainingData)-1): 
            for i in range(NoOfStates):
                for j in range(NoOfStates):
                    kisi[i,j,t] = alpha[i,t]*A[i,j]* B(j,TrainingData[t+1])*beta[j,t+1].T/observation_probability
        
        for i in range(NoOfStates):
            for j in range(NoOfStates):
                numerator = np.sum(kisi[i,j,:T-1])
                denomerator = np.sum(kisi[i,:,:T-1])
                A[i,j] = numerator / denomerator
        
        
        for j in range(NoOfStates):
            numerator1 = 0
            for t in range(len(TrainingData)-1): 
                numerator1 = numerator + gamma[j,t]* TrainingData[t]
            denomerator = np.sum(gamma[j,:])
            mu[j] = numerator1 / denomerator
                
        for j in range(NoOfStates):
            numerator2 = 0
            for t in range(len(TrainingData)-1): 
                numerator2 = numerator2 + gamma[j,t]* (TrainingData[t]- mu[j])*(TrainingData[t]- mu[j])
            denomerator = np.sum(gamma[j,:])
            sigma[j] = numerator2 / denomerator

        for state in range(NoOfStates):
            A[state,:] /= np.sum(A[state,:])
            ###B[state,:] /= np.sum(B[state,:])
        #end = time.time()
        #print(end-start)
        print("iterate_num: ",iterate_num)
    print("\n")
    
def test():
    global TestData, Raw_words,NoOfStates,NoOfObservations,TrainingData,A,B,observation_probability,alpha,beta,gamma,kisi
    TestData = np.array(TestData)    
    num_test = len(TestData)
#    print(TestData)
#    print("num_test:")
#    print(num_test)
#    for i in range(num_test-1):
#        print(TestData[i])
    prob_f = forwardAlgorithm(TestData,1)
    prob_b = backwardAlgorithm(TestData)
    path,bestprob = viterbiAlgorithm(TestData)
    print(path)


        
def main(data, RawFile, Output):
    global TrainingData, TestData, Pi
    TrainingData, TestData = initialize()
#    print(Pi)
    forwardBackwardAlgorithm(5)
    test()


main('close', 'Data/rawClose', 'Output/OUT')
