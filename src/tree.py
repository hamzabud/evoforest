import hashlib
import math
from typing import Counter
import numpy as np
import random
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
from binarytree import build, Node

@dataclass
class EvoParams:
    outcome_type: str = 'boolean'
    population_size: int = 100
    max_depth: int = 15
    crossover_rate: float = 0.5
    mutation_rate: float = 0.1
    n_elites: int = 10
    n_classifiers: int = 10
    early_stopping_patiance: int = 10
    training_sample_size: int = None
    initial_temperature: float = 1
    cooling_rate: float = 0.999
    min_temp: float = 0.1
    n_species: int = 10
    
    def __str__(self):
        return f"Population size: {self.population_size}\n" + \
        f"Max depth: {self.max_depth}\n" + \
        f"Crossover rate: {self.crossover_rate}\n" + \
        f"Mutation rate: {self.mutation_rate}\n" + \
        f"Number of elites: {self.n_elites}\n" + \
        f"Number of classifiers: {self.n_classifiers}\n" + \
        f"Early stopping patiance: {self.early_stopping_patiance}\n" + \
        f"Training sample size: {self.training_sample_size}\n" + \
        f"Initial temperature: {self.initial_temperature}\n" + \
        f"Cooling rate: {self.cooling_rate}\n" + \
        f"Minimum temperature: {self.min_temp}\n" + \
        f"Number of species: {self.n_species}"


class EvolutionaryForest:
    def __init__(self, data, params: EvoParams):
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.outcomes = np.unique(self.y_train)
        self.internal_outcomes = set()
        self.stats = data['stats']
        self.evo_params = params
        self.ensemble = []
        self.temperature = params.initial_temperature
        self.feature_mappings = {feature: idx for idx, feature in enumerate(self.stats.keys())}
        self.populations = self._generate_initial_population()

    def _generate_initial_population(self):
        populations = []
        print("Generating initial population")
        for _ in range(self.evo_params.n_species):
            population = []
            while len(population) < self.evo_params.population_size // self.evo_params.n_species:
                tree = self._generate_random_tree()
                population.append(tree)
            populations.append(population)
        return populations

    def _evaluate_species(self):
        print("Evaluating species")
        for i in range(len(self.populations)):
            self.populations[i] = self._evaluate_population(self.populations[i])
        print("Evaluating species complete")
        
        
    def _select_species(self):
        print("Selecting for all species")
        for i in range(len(self.populations)):
            self.populations[i] = self._select_population(self.populations[i])
            
        print("Selecting for all species complete")
        
    
    def _crossover_species(self):
        print("Crossover for all species")
        for i in range(len(self.populations)):
            self.populations[i] = self._crossover_population(self.populations[i])
        print("Crossover for all species complete")
            

    def _evaluate_population(self, population):
        n_samples = self.evo_params.training_sample_size or self.X_train.shape[0] // 2
        X_sample, y_sample = resample(self.X_train, self.y_train, replace=True, n_samples=n_samples)
        hashes = set()
        new_population = []
        for tree in population:
            hash = self._hash_tree(tree)
            complexity_penelty =  len(tree.values) * 0.001
            duplicate_penelty = 0
            duplicate_features = set()
            for node in tree.inorder:
                if node.value.startswith('-1'):
                    continue
                feature = node.value.split('?')[1]
                if feature in duplicate_features:
                    duplicate_penelty += 0.4
                else:
                    duplicate_features.add(feature)
            if hash in hashes:
                duplicate_penelty += 0.5
            else:
                hashes.add(hash)
                
            predictions = np.array([self._predict_sample(sample, tree) for sample in X_sample])
            accuracy = accuracy_score(y_sample, predictions)
            recall = recall_score(y_sample, predictions, average='weighted', zero_division=0)
            precision = precision_score(y_sample, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_sample, predictions, average='weighted', zero_division=0)
            tree.fitness = max(0, 0.7 * accuracy + 0.1 * recall + 
                               0.1 * precision + 0.1 * f1 - complexity_penelty - duplicate_penelty)
            new_population.append(tree)
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        return new_population

    def _generate_random_tree(self):
        num_nodes = random.randint(1, len(self.feature_mappings))
        genome = []
        features = list(self.stats.keys())
        random.shuffle(features)
        selected_features = features[:num_nodes]
        for feature in selected_features:
            stats = self.stats[feature]
            threshold = round(np.random.choice([stats['mean'], random.uniform(stats['min'], stats['max'])]), 2)
            operator = np.random.choice(['<', '>', '==', '!='])
            identifier = self.feature_mappings[feature]
            genome.append(f"{identifier}?{feature}?{operator}?{threshold}")
        genome.extend([f'-1?{outcome}' for outcome in np.random.choice(self.outcomes, size=num_nodes+1).tolist()])
        tree = build(genome)
        tree.fitness = 0
        return tree


    def _predict_sample(self, sample, tree):
        node = tree
        while True:
            
            
            if node.value.startswith('-1'):
                try:
                    if self.evo_params.outcome_type == 'boolean':
                        return node.value.split('?')[1].lower() == 'true'
                    elif self.evo_params.outcome_type == 'int':
                        return int(node.value.split('?')[1])
                    elif self.evo_params.outcome_type == 'float':
                        return float(node.value.split('?')[1])
                    else:
                        return node.value.split('?')[1]
                except:
                    print(f'Node: {node.value}')
                    raise
            
            try:
                identifier, feature, operator, threshold = node.value.split("?")
            except:
                print(f'Node: {node.value} in outcomes? {node.value in self.outcomes}')
                print(f'Outcomes: {self.outcomes}')
                print(f'Type: {type(node.value)}')
                raise
            
            val = sample[int(identifier)]
            threshold = float(threshold)
            if (operator == '<' and val < threshold) or \
            (operator == '>' and val > threshold) or \
            (operator == '==' and val == threshold) or \
            (operator == '!=' and val != threshold):
                node = node.left
            else:
                node = node.right
                

    def evolve(self, generations=1):
        
        best_fitness = -1
        patience_counter = 0
        generation = 0
        print(f"-- Evolutionary forest --")
        while True:
            generation += 1
            self._evaluate_species()
            current_best_fitness = self._print_top_individuals()[1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > self.evo_params.early_stopping_patiance or generation > generations:
                self.build_ensemble()
                #print(f'-- Stagnated, early stopping triggered --')
                #print(f"-- Final population has {len(self.population)} trees - best fitness: {self.population[0].fitness} --")
                break
            trees = [tree for population in self.populations for tree in population]
            print(f"Generation {generation}/{generations} has {len(trees)} trees")
            print(f"  all-time best fitness: {best_fitness:.2f}")
            print(f"  best fitness this generation: {current_best_fitness:.2f}")
            #print(f"  average fitness: {np.mean([tree.fitness for tree in self.population]):.2f}")
            print(f"  current temperature: {self.temperature:.2f}")
            self._select_species()
            self._crossover_species()
            self.temperature = max(self.evo_params.min_temp, math.exp(-generation/(generations // 2)))
            
    def _select_population(self, population):
        elites = population[:self.evo_params.n_elites//self.evo_params.n_species]
        non_elites = population[self.evo_params.n_elites//self.evo_params.n_species:]
        
        fitnesses = np.array([tree.fitness for tree in non_elites])
        probabilities = np.exp(fitnesses / self.temperature)
        probabilities /= probabilities.sum()
        
        size = len(population) - self.evo_params.n_elites // 2
        selected_indicies = np.random.choice(len(non_elites), size=size, p=probabilities)
        selected = elites + [non_elites[i] for i in selected_indicies]
        
        return sorted(selected, key=lambda x: x.fitness, reverse=True)
            
    def build_ensemble(self, size=None):
        
        size = size if size is not None else self.evo_params.n_classifiers
        ensembles = []
        for i in range(len(self.populations)):
            classifiers = random.sample(self.populations[i], size)
            predictions = []
            for tree in classifiers:
                predictions.append([self._predict_sample(sample, tree) for sample in self.X_val])
            predictions = np.array(predictions).T
            ensemble_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions]
            score = accuracy_score(self.y_val, ensemble_predictions)
            ensembles.append((classifiers, score))
        ensembles.sort(key=lambda x: x[1], reverse=True)
        self.ensemble = ensembles[0][0]
        return self.ensemble

            
    def _crossover_population(self, population):
        new_population = population[:self.evo_params.n_elites]
        population_size = self.evo_params.population_size
        while len(new_population) < population_size // self.evo_params.n_species:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutate_tree(child1)
            child2 = self._mutate_tree(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        
        child1, child2 = parent1.clone(), parent2.clone()
        
        if random.random() > self.evo_params.crossover_rate:
            return child1, child2
            
            
        child1_values = np.array(parent1.values)
        child2_values = np.array(parent2.values)
    
        is_non_leaf1 = np.vectorize(lambda v: v is not None and not v.startswith('-1'))(child1_values)
        is_non_leaf2 = np.vectorize(lambda v: v is not None and not v.startswith('-1'))(child2_values)
        
        non_leaf_indices1 = np.where(is_non_leaf1 & (np.arange(len(child1_values)) != 0))[0]
        non_leaf_indices2 = np.where(is_non_leaf2 & (np.arange(len(child2_values)) != 0))[0]
        
        if len(non_leaf_indices1) <= 1 or len(non_leaf_indices2) <= 1:
            return self._generate_random_tree(), self._generate_random_tree()

        cutoff1 = np.random.choice(non_leaf_indices1)
        cutoff2 = np.random.choice(non_leaf_indices2)
        
        cutoff1, cutoff2 = int(cutoff1), int(cutoff2)
        child1[cutoff1], child2[cutoff2] = child2[cutoff2], child1[cutoff1]

        
        return child1, child2

    def _mutate_tree(self, tree):
        mutation_rate = self.evo_params.mutation_rate
        
        non_leaf_indices = np.where(np.vectorize(lambda v: v is not None and not v.startswith('-1'))(tree.values))[0]
        
        mutate_flags = np.random.random(len(non_leaf_indices)) < mutation_rate
        
        for idx, mutate in zip(non_leaf_indices, mutate_flags):
            if mutate:
                if random.random() < 0.5:
                    self._mutate_node(tree, idx)
                else:
                    rand_index = random.choice(non_leaf_indices)
                    tree.values[idx], tree.values[rand_index] = tree.values[rand_index], tree.values[idx]
        return tree

    def _mutate_node(self, tree, index):
        node_value = tree.values[index]
        if node_value.startswith('-1'):
            tree.values[index] = f'-1?{random.choice(self.outcomes)}'
        else:
            identifier, feature, operator, threshold = node_value.split("?")
            threshold_variation = random.uniform(-self.stats[feature]['std_dev'], self.stats[feature]['std_dev'])
            new_threshold = round(float(threshold) + threshold_variation, 2)
            tree.values[index] = f"{identifier}?{feature}?{operator}?{new_threshold}"


    def _print_top_individuals(self):
        absolute_best = (None, None)
        for i in range(len(self.populations)):
            #print(f"Species {i} has {len(self.populations[i])} individuals with best fitness: {self.populations[i][0].fitness:.2f}")
            if absolute_best[1] is None or self.populations[i][0].fitness > absolute_best[1]:
                absolute_best = (i, self.populations[i][0].fitness)
        return absolute_best
    
    
    def _hash_tree(self, tree):
        tree_str = ' '.join([str(node.value) for node in tree.inorder])
        return hashlib.md5(tree_str.encode('utf-8')).hexdigest()

    def predict(self, X):
        predictions = np.array([self._predict_sample(x, tree) for tree in self.ensemble for x in X]).reshape(len(self.ensemble), len(X)).T
        return [Counter(preds).most_common(1)[0][0] for preds in predictions]