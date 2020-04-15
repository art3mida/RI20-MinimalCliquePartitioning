'''
    Encoding:
        A chromosome is a list of N elements, where N is the number of nodes
        in the graph. List indices represent nodes, and the value on each
        index represents the color of that node.

        In addition to that, there is an adjacency list that essentially
        represents the graph. The list on each index of the adjacency list
        represents the nodes adjacent to the one with that index.

        Fitness of a chromosome is equal to the number of nodes that are
        invalidly colored. A node is invalidly colored if it has the same color
        as any of its adjacent nodes.
        TODO: Fitness could maybe be improved by taking into consideration
                 the number of conflicting nodes or even just the number of
                 adjacent nodes. For example, a node that is invalidly colored,
                 but has only 2 adjacent nodes is "less wrong" than the one
                 that has 30 adjacent nodes, since changing its color removes
                 more conflicts. It's important to note that it is essential
                 to change its color to a valid one, otherwise you are also
                 creating a lot more conflicts.
        COMMENT: Added a very simple improvement for this: instead of just
                 checking whether there is a conflict, for each of the conflicting
                 nodes within the adjacency list, fitness is increased by one.
                 This will (usually) make the chromosomes that have conflicts on
                 high degree nodes "more wrong".

        Since higher fitness represent a worse chromosome, the goal is to
        minimize the fitness function, hence the target is 0.

        Since we have a clear target function, the termination criteria is
        either reaching the target or exceeding the maximum number of
        iterations.

        Crossover operator is a simple one-point crossover.
        TODO: This can be improved a lot. Possible idea: dominant and
              recessive genes.

        TODO: Explain mutation operator and tabu search.
'''

import random
import heapq
import time
from operator import attrgetter

class Chromosome:
    def __init__(self, code, fitness):
        self.code = code
        self.fitness = fitness

    def __str__(self):
        return str(self.code)

    def __repr__(self):
        return str(self.code)

    def __gt__(self, other_chromosome):
        return self.fitness > other_chromosome.fitness

    def __lt__(self, other_chromosome):
        return self.fitness < other_chromosome.fitness

class GeneticAlgorithm:
    def __init__(self, chromatic_num):
        # Tunable genetic algorithm parameters.
        self.max_iters = 100
        self.population_size = 50
        self.reproduction_size = 20
        self.elitism_rate = 5
        self.mutation_rate = 0.4
        self.tournament_size = 10

        # Other GA attributes.
        self.chromatic_num = chromatic_num
        self.possible_gene_values = [i for i in range(self.chromatic_num)]
        self.target = 0
        self.current_iter = 0
        self.best_chromosome = None

        # Additional LS and graph parameters.
        self.num_vertices = 0
        self.adjacency_list = {}
        self.tabu_quality_boundary = 5
        self.max_tabu_iters = 4

    def create_initial_population(self):
        population = []
        for i in range(self.population_size):
            code = []
            for j in range(self.num_vertices):
                code.append(random.choice(self.possible_gene_values))
            population.append(code)
        population = [Chromosome(code, self.calculate_fitness(code)) for code in population]
        return population

    def calculate_fitness(self, code):
        fitness = 0
        for i in range(self.num_vertices):
            for j in self.adjacency_list[i]:
                if code[i] == code[j]:
                    fitness += 1
        return fitness

    def select_best_chromosome(self, current_pool):
        best = min(current_pool, key=attrgetter('fitness'))
        return best

    def tournament_selection(self):
        selected = []
        for i in range(self.reproduction_size - self.elitism_rate):
            current_pool = []
            for i in range(self.tournament_size):
                current_chromosome = random.choice(self.population)
                current_pool.append(current_chromosome)
            best_chromosome = self.select_best_chromosome(current_pool)
            selected.append(best_chromosome)
        selected.extend(heapq.nsmallest(self.elitism_rate, self.population))
        return selected

    def crossover_chunk(self, parent1, parent2):
        cross_point = random.randint(1, self.num_vertices - 1)
        child1_code = parent1.code[:cross_point] + parent2.code[cross_point:]
        child2_code = parent2.code[:cross_point] + parent1.code[cross_point:]

        child1 = Chromosome(child1_code, self.calculate_fitness(child1_code))
        child2 = Chromosome(child2_code, self.calculate_fitness(child2_code))

        return child1, child2

    def uniform_crossover(self, parent1, parent2):
        prob = 0.5
        better_parent = None
        if parent1.fitness < parent2.fitness:
            better_parent = parent1
            worse_parent = parent2
        else:
            better_parent = parent2
            worse_parent = parent1

        child1_code = []
        child2_code = []
        for i in range(self.num_vertices):
            r = random.random()
            if r < prob:
                child1_code.append(better_parent.code[i])
                child2_code.append(better_parent.code[i])
            else:
                child1_code.append(better_parent.code[i])
                child2_code.append(worse_parent.code[i])

        child1 = Chromosome(child1_code, self.calculate_fitness(child1_code))
        child2 = Chromosome(child2_code, self.calculate_fitness(child2_code))

        return child1, child2


    def get_available_colors(self, chromosome, i):
        return list(set(self.possible_gene_values) - set([chromosome[vertex] for vertex in self.adjacency_list[i]]))

    def mutate(self, chromosome):
        p = random.random()
        if p < self.mutation_rate:
            i = random.randrange(0, len(chromosome) - 1)
            # We are converting both of these lists to sets just for the purpose
            # of using set difference operator.
            available_colors = self.get_available_colors(chromosome, i)
            if len(available_colors) > 0:
                chromosome[i] = random.choice(available_colors)
            else:
                chromosome[i] = random.choice(self.possible_gene_values)
        return chromosome

    def generate_neighbourhood(self, chromosome, tabu_list):
        nbhd = []
        moves = []
        for i in range(len(chromosome.code)):
            available_colors = self.get_available_colors(chromosome.code, i)
            for color in available_colors:
                moves.append((i, color))

        moves = list(set(moves) - set(tabu_list))

        for move in moves:
            new_config = chromosome.code
            new_config[move[0]] = move[1]
            nbhd.append(Chromosome(new_config, self.calculate_fitness(new_config)))

        return nbhd, moves

    def tabu_search(self, chromosome):
        current_solution = chromosome
        best_solution = chromosome
        tabu_list = []
        for i in range(self.max_tabu_iters):
            if current_solution.fitness < best_solution.fitness:
                best_solution = current_solution
            nbhd, moves = self.generate_neighbourhood(chromosome, tabu_list)
            current_solution = self.select_best_chromosome(nbhd)
            index_of_current_solution = nbhd.index(current_solution)
            tabu_list.append(moves[index_of_current_solution])
        return best_solution

    def create_generation(self, reproduction_group):
        current_generation = []
        while len(current_generation) < self.population_size:
            parents = random.sample(reproduction_group, 2)
            child1, child2 = self.uniform_crossover(parents[0], parents[1])

            child1.code = self.mutate(child1.code)
            child2.code = self.mutate(child2.code)

            current_generation.append(child1)
            current_generation.append(child2)

        # Tabu search
        for i in range(self.population_size):
            if current_generation[i].fitness < self.tabu_quality_boundary:
                current_generation[i] = self.tabu_search(current_generation[i])

        return current_generation

    def should_terminate(self):
        if self.current_iter >= self.max_iters or self.best_chromosome.fitness == self.target:
            return True
        else:
            return False

    def optimize(self):
        self.population = self.create_initial_population()
        self.best_chromosome = self.select_best_chromosome(self.population)

        while not self.should_terminate():
            reproduction_group = self.tournament_selection()
            self.population = self.create_generation(reproduction_group)
            self.best_chromosome = self.select_best_chromosome(self.population)

        return self.best_chromosome, self.current_iter

def main():
    graphs = ['myciel3.col','myciel4.col', 'myciel5.col', 'myciel6.col', 'games120.col', 'huck.col', 'jean.col']
    

if __name__ == '__main__':
    main()
