import os
import time
import random
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pymongo
import argparse,pickle

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ---------------------------------------------------------------------------- #
# Utility: Data loading and preprocessing
# ---------------------------------------------------------------------------- #

def load_data_and_model():

    with open("../best_split.pkl", "rb") as f:
        (_, _), (X_val, y_val) = pickle.load(f)
    
    model = tf.keras.models.load_model("../best_model.keras")
    
    return X_val, y_val, model


# ---------------------------------------------------------------------------- #
# Genetic Algorithm for feature selection
# ---------------------------------------------------------------------------- #
class GeneticAlgorithm:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: tf.keras.Model,
        population_size: int = 100,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        elitism: int = 1,
        penalty_coef: float = 0.2,
        patience: int = 20,
        rel_thr: float = 0.001,
        injection_rate: float = 0.02,
        adaptive_elitism: bool = False,
        injection: bool = False,
        stagnation_threshold: int = 20,
        max_generations: int = 1000,
        selection_method: str = 'tournament',
        crossover_method: str = 'uniform'
    ):
        # Data & model
        self.X = X
        self.y = y
        self.model = model
        self.num_features = X.shape[1]

        # GA hyperparameters
        self.adaptive_elitism=adaptive_elitism
        self.injection=injection
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.penalty_coef = penalty_coef
        self.patience = patience
        self.rel_thr = rel_thr
        self.injection_rate = injection_rate
        self.stagnation_threshold = stagnation_threshold
        self.max_generations = max_generations

        # Internal state
        self.population = []
        self.fitnesses = []
        self.best_global = -np.inf
        self.prev_best = -np.inf
        self.no_improve_count = 0

        # Operator mappings
        self.selection_fns = {
            'roulette': self.roulette,
            'rank': self.rank,
            'tournament': self.tournament
        }
        self.crossover_fns = {
            'one_point': self.one_point_crossover,
            'two_point': self.two_point_crossover,
            'uniform': self.uniform_crossover
        }
        self.select_fn = self.selection_fns[selection_method]
        self.crossover_fn = self.crossover_fns[crossover_method]

    def generate_chromosome(self) -> np.ndarray:
        return np.random.randint(0, 2, size=self.num_features, dtype=np.int8)

    def initialize_population(self):
        self.population = [self.generate_chromosome() for _ in range(self.population_size)]

    def make_mask(self, mask: np.ndarray) -> np.ndarray:
        return self.X * mask.reshape(1, -1).astype(np.float32) #constant 0 to the non selected columns

    def fitness(self, mask: np.ndarray) -> float:
        Xm = self.make_mask(mask)
        _, accuracy,_ = self.model.evaluate(Xm, self.y, verbose=0)
        penalty = mask.sum() / mask.size
        return accuracy - self.penalty_coef * penalty

    def evaluate_population(self):
        self.fitnesses = [self.fitness(ind) for ind in self.population] 

    # --------------------- Selection operators --------------------- #
    def roulette(self): #simple roulette selection
        total = sum(self.fitnesses)
        probs = [f / total for f in self.fitnesses] 
        idx = np.random.choice(len(self.population), size=2, p=probs, replace=True) #random coice normalizes the propabilties and uses a random to find in which interval of the cumulative sum the random number falls into
        return self.population[idx[0]].copy(), self.population[idx[1]].copy()

    def rank(self): #rank slection with wheel based on chromose rank
        idx_sorted = np.argsort(self.fitnesses)
        ranks = np.empty_like(idx_sorted)
        ranks[idx_sorted] = np.arange(1, len(self.fitnesses) + 1)
        probs = ranks / ranks.sum()
        idx = np.random.choice(len(self.population), size=2, p=probs, replace=True)
        return self.population[idx[0]].copy(), self.population[idx[1]].copy()

    def tournament(self, k: int = 3): #tournament selection
        contenders = list(np.random.choice(len(self.population), size=k, replace=False)) 
        best = max(contenders, key=lambda i: self.fitnesses[i])
        contenders.remove(best)
        second = max(contenders, key=lambda i: self.fitnesses[i])
        return self.population[best].copy(), self.population[second].copy()

    # --------------------- Crossover operators --------------------- #
    def one_point_crossover(self, p1: np.ndarray, p2: np.ndarray):
        pt = np.random.randint(1, self.num_features)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1.astype(np.int8), c2.astype(np.int8) #split on a random point 

    def two_point_crossover(self, p1: np.ndarray, p2: np.ndarray):
        pts = sorted(np.random.choice(range(1, self.num_features), size=2, replace=False)) #on 2 points
        c1, c2 = p1.copy(), p2.copy()
        c1[pts[0]:pts[1]], c2[pts[0]:pts[1]] = p2[pts[0]:pts[1]], p1[pts[0]:pts[1]]
        return c1.astype(np.int8), c2.astype(np.int8)

    def uniform_crossover(self, p1: np.ndarray, p2: np.ndarray):
        mask = np.random.rand(self.num_features) < 0.5
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return c1.astype(np.int8), c2.astype(np.int8) #flip based on a mask randomly generated 

    # --------------------- Mutation --------------------- #
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(self.num_features):
            if random.random() < self.mutation_rate: #flip random bits
                individual[i] = 1 - individual[i]
        return individual

    # --------------------- Generation update --------------------- #
    def next_generation(self, current_generation:int):
        #both injection and adaptive elitism are optional and can be opened in the parser
        if self.adaptive_elitism and self.no_improve_count >= self.stagnation_threshold: #if elites hinder the process delete them after some gens
            current_elitism = 0
        else:
            current_elitism = self.elitism

        idx_sorted = np.argsort(self.fitnesses)[::-1]
        elites = [self.population[i].copy() for i in idx_sorted[:current_elitism]]

        new_pop = elites.copy()
        while len(new_pop) < self.population_size: #until you make the new pouplation as big as the old one 
            p1, p2 = self.select_fn() #crossover
            
            if random.random() < self.crossover_rate: #mutate if not last gen
                c1, c2 = self.crossover_fn(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if current_generation < self.max_generations : #append the result
                new_pop.append(self.mutate(c1))
            else:
                new_pop.append(c1)

            if len(new_pop) < self.population_size:
                if current_generation < self.max_generations :
                    new_pop.append(self.mutate(c2))
                else:
                    new_pop.append(c2)

        if self.injection: #inject random chromoses in order to avoid local maximums
            num_random = int(self.injection_rate * self.population_size)
            for i in range(num_random):
                new_pop[-1 - i] = self.generate_chromosome()

        self.population = new_pop[:self.population_size]

    # --------------------- Run method --------------------- #
    def run(self) -> dict:
        self.initialize_population()
        history = []

        for gen in range(1, self.max_generations + 1):
            self.evaluate_population()
            best = max(self.fitnesses)
            history.append(best)

            if best > self.best_global:
                self.best_global = best
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            print(f"Gen {gen:4d} — best fitness: {best:.4f} (global: {self.best_global:.4f})")

            #the asked stop statements
            if gen > 1:
                rel_imp = (best - self.prev_best) / abs(self.prev_best) if self.prev_best != 0 else float('inf')
                # only stop on relative improvement if fitness changed
                if best != self.prev_best and rel_imp < self.rel_thr:
                    print(f"Stopping: relative improvement {rel_imp*100:.2f}% < {self.rel_thr*100:.2f}%")
                    break

            self.prev_best = best

            if self.no_improve_count >= self.patience:
                print(f"Stopping: no improvement in {self.patience} generations")
                break

            if gen >= self.max_generations:
                print(f"Stopping: reached maximum generations {self.max_generations}")
                break

            self.next_generation(gen)

        best_idx = int(np.argmax(self.fitnesses))
        best_mask = self.population[best_idx]
        return {
            'best_mask': best_mask,
            'best_fitness': float(self.fitnesses[best_idx]),
            'generations': gen,
            'history': history
        }


# ---------------------------------------------------------------------------- #
# CLI Runner and MongoDB logging
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Feature-selection GA")
    parser.add_argument('--N',      type=int, default=200)
    parser.add_argument('--mu',     type=float, default=0.01)
    parser.add_argument('--pc',     type=float, default=0.7)
    parser.add_argument('--elitism',type=int, default=1)
    parser.add_argument('--penalty',type=float, default=0.2)
    parser.add_argument('--patience',type=int,default=100)
    parser.add_argument('--rel_thr',type=float, default=0.001)
    parser.add_argument('--inject', type=float, default=0.02)
    parser.add_argument('--stag',   type=int,   default=10)
    parser.add_argument('--max_gens', type=int, default=1000)
    parser.add_argument('--sel',    choices=['roulette','rank','tournament'], default='tournament')
    parser.add_argument('--cross',  choices=['one_point','two_point','uniform'], default='uniform')
    parser.add_argument('--tag',    type=str, default='')
    parser.add_argument("--adaptive_elitism",type=bool,default=False)
    parser.add_argument("--injection",type=bool,default=False)
    args = parser.parse_args()

    X, y, model = load_data_and_model()

    ga = GeneticAlgorithm(
        X, y, model,
        population_size=args.N,
        mutation_rate=args.mu,
        crossover_rate=args.pc,
        elitism=args.elitism,
        penalty_coef=args.penalty,
        patience=args.patience,
        rel_thr=args.rel_thr,
        injection_rate=args.inject,
        adaptive_elitism=args.adaptive_elitism,
        injection=args.injection,
        stagnation_threshold=args.stag,
        max_generations=args.max_gens,
        selection_method=args.sel,
        crossover_method=args.cross
    )
    result = ga.run()

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    col = client['genetic_algos']['resultsv2']
    doc = result.copy()
    doc.update({'params': vars(args)})
    doc['best_mask'] = doc['best_mask'].tolist()
    col.insert_one(doc)

    print("\n|---- FINAL RESULTS ----|")
    pprint.pprint(result)
