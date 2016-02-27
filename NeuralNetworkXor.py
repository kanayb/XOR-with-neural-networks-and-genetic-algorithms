import numpy as np
import random
from deap import algorithms, base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Training set.
truthTableXor = [[1,1,0],[1,0,1],[0,1,1],[0,0,0]]

# Activation function which is sigmoid
def activationFunction(a):
    return (1/(1+np.exp(-a)))
# First hidden node
def hiddenNode1(x1,x2,w1,w2,w3):
    bi = 1
    resultValue = x1 * w1 + x2* w2 + bi*w3
    return activationFunction(resultValue)
# Second hidden node
def hiddenNode2(x1,x2,w1,w2,w3):
    bi = 1
    resultValue = x1 * w1 + x2* w2 + bi*w3
    return activationFunction(resultValue)

def xorMethod(x1,x2,w1,w2,w3):
    bi = 1
    resultValue = x1 * w1 + x2* w2 + bi*w3
    return activationFunction(resultValue)
# Evaluation function.
def xorEvaluator(individual):
    differenceTotalXor = 0
    for i in range(4):
        actualValueXor = truthTableXor[i][2]
        valueFromHid1 = hiddenNode1(truthTableXor[i][0],truthTableXor[i][1],individual[0],individual[1],individual[2])
        valueFromHid2 = hiddenNode2(truthTableXor[i][0],truthTableXor[i][1],individual[3],individual[4],individual[5])
        differenceXor = (actualValueXor - xorMethod(valueFromHid1,valueFromHid2,individual[6],individual[7],individual[8]))**2
        differenceTotalXor =  differenceTotalXor + differenceXor
    return (differenceTotalXor,)
# Every individual starts with 9 random floating numbers between -10 and 10. 3 for first hidden node, 3 for second hidden node and 3 for xor node which is the output node.
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=9)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual)
# Uses blend crossover and gaussian mutation
# Selection is torunament selection with 3 individuals
toolbox.register("evaluate", xorEvaluator)
toolbox.register("mate", tools.cxBlend,alpha = 0.5)
toolbox.register("mutate", tools.mutGaussian, mu = 0.0,sigma= 0.5, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
# 500 individual population and 150 generations
pop = toolbox.population(n=500)
result = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                             ngen=150, verbose=False)

bestIndividual = tools.selBest(pop, k=1)[0]
# Printing out the results
print 'The min value is', xorEvaluator(bestIndividual)[0]
print 'The weights for first Hidden node are',bestIndividual[0],'and',bestIndividual[1],'The bias weight is ',bestIndividual[2]
print 'The weights for second Hidden node are',bestIndividual[3],'and',bestIndividual[4],'The bias weight is ',bestIndividual[5]
print 'The weights for the output node are',bestIndividual[6],'and',bestIndividual[7],'The bias weight is ',bestIndividual[8]
print
for i in range(4):
    print 'First input value',truthTableXor[i][0],'Second input value',truthTableXor[i][1]
    valueFromHid1 = hiddenNode1(truthTableXor[i][0],truthTableXor[i][1],bestIndividual[0],bestIndividual[1],bestIndividual[2])
    valueFromHid2 = hiddenNode2(truthTableXor[i][0],truthTableXor[i][1],bestIndividual[3],bestIndividual[4],bestIndividual[5])
    print 'output of optimized XOR ', xorMethod(valueFromHid1,valueFromHid2,bestIndividual[6],bestIndividual[7],bestIndividual[8])
    
