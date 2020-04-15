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

    # Ovo je falilo za heap. Treba definisati operatore '<' i '>' za klasu.
    def __gt__(self, other_chromosome):
        return self.fitness > other_chromosome.fitness

    def __lt__(self, other_chromosome):
        return self.fitness < other_chromosome.fitness

class GeneticAlgorithm:
    def __init__(self, chromatic_num):
        # Tunable genetic algorithm parameters.
        self.max_iters = 1000
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

    def create_initial_population(self):
        population = []
        for i in range(self.population_size):
            code = []
            for j in range(self.num_vertices):
                code.append(random.choice(self.possible_gene_values))
                # Ne moze jer ih nisu svi instancirani jos, moraju prvo svi
                # kodovi da postoje.
                # chromosome = Chromosome(code, self.calculate_fitness(code))
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
        # The best chromosome is the one with the min value of fitness function!
        best = min(current_pool, key=attrgetter('fitness'))
        return best

    def tournament_selection(self):
        selected = []
        # If we don't want the elite elements to necessarily qualify for
        # reproduction, we should delete the '- self.elitism_rate' part.
        for i in range(self.reproduction_size - self.elitism_rate):
            current_pool = []
            for i in range(self.tournament_size):
                current_chromosome = random.choice(self.population)
                current_pool.append(current_chromosome)
            best_chromosome = self.select_best_chromosome(current_pool)
            selected.append(best_chromosome)
        # Adding the elite ones to the selected list.
        selected.extend(heapq.nsmallest(self.elitism_rate, self.population))
        return selected

    def crossover_chunk(self, parent1, parent2):
        # Problem je bio u odabiru najboljeg hromozoma - nije bilo
        # potrebno da se i tu koristi konstruktor za hromozom, jer se onda
        # dobije neka cudna konstrukcija i parent1.code nije tipa lista, nego
        # tipa Chromosome.
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


    def create_generation(self, reproduction_group):
        current_generation = []
        while len(current_generation) < self.population_size:
            parents = random.sample(reproduction_group, 2)
            child1, child2 = self.crossover_chunk(parents[0], parents[1])

            child1.code = self.mutate(child1.code)
            child2.code = self.mutate(child2.code)

            current_generation.append(child1)
            current_generation.append(child2)

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
