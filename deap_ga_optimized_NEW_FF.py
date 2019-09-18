import random
import time
from deap import base
from deap import creator
from deap import tools
import numpy as np
from PIL import Image
import sys

start_time = time.time() # Calculates runtime
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_bool, 30000) 

# IND Size 8000 BETTER
# pop 1000 – 5000 anything less converges too early


toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Archetype
temp=Image.open('Guernica_large.png')
temp=temp.convert('1')      # Convert to black&white
A = np.array(temp)             # Creates an array, white pixels==True and black pixels==False
new_A=np.empty((A.shape[0],A.shape[1]),None)    #New array with same size as A
for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j]==True:
            new_A[i][j]=0
        else:
            new_A[i][j]=1
            
np.set_printoptions(threshold=sys.maxsize)

new_A = np.reshape(new_A, 30000)
new_A = np.array(new_A, dtype='int')


#A = np.random.choice([0, 1], size=(8000))

# Higher pop seems to maximize in fewer gens (each gen takes longer to compute)
# Very small population to ind siz ratio (i.e. 20:15000) results in very early convergence with poor fitness score (or slow progress)
# ind=15000, gens=50000: pop=50 20h, pop=20 8h
pop = toolbox.population(n=6000)
pop = (pop)



#print(A)
print("\n")
#print(toolbox.population(30))
print("\n")


def evalComp(individual):
    compare = np.absolute(new_A - individual)
    compare = np.sum(compare, axis=0)/30000
    return compare,

toolbox.register("evaluate", evalComp)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    pop = toolbox.population(n=6000)
   
    
    print("Start of evolution")
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    print(" Evaluated %i individuals" % len(pop))
    
    # CXPB After 50 gens At 0.3 –– 0.285-0.325, At 0.5 –– 0.31-0.34, At 0.7 –– 0.32-0.33, At 0.9 –– 0.315-0.335
    CXPB, MUTPB = 0.5, 0.3
    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
    while max(fits) < 1.0 and g < 500:
        g = g + 1
        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2, 0.3) # At 0.5 216-269 gens –– At 0.4 217-238 gens –– At 0.3 199-249 gens –– At 0.2 230-278
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print(" Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" Std %s" % std)
    
    print("-- End of (successful) evolution --")
    print("\n")
    print("\n")
    print("Archetype")
#    print(A)
    print("\n")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    
if __name__ == "__main__":
    main()
        

print("--- %s seconds ---" % (time.time() - start_time))       