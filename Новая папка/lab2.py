import random
import numpy as np
from deap import base, creator, tools, algorithms

def f(individual):
    x, y, z = individual
    return 1 / (1 + (x-2)**2 + (y+1)**2 + (z-1)**2),

def create_individual():
    return [random.uniform(-10, 10) for _ in range(3)]

def crossover(parent1, parent2):
    alpha = random.random()
    offspring1 = [alpha*p1 + (1-alpha)*p2 for p1, p2 in zip(parent1, parent2)]
    offspring2 = [(1-alpha)*p1 + alpha*p2 for p1, p2 in zip(parent1, parent2)]
    return creator.Individual(offspring1), creator.Individual(offspring2)

def mutate(individual):
    index = random.randrange(len(individual))
    individual[index] += random.uniform(-1, 1)
    return individual,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", f)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                        stats=stats, halloffame=hof, verbose=True)
    return pop, logbook, hof

if __name__ == "__main__":
    main()
