import multiprocessing
from pprint import pprint
import random
import time

import numpy as np
from joblib import Parallel, delayed

from src.ewo_exceptions import *
from src.util import *

globalMinimum = 0.0
NUM_ITERATIONS = 500
RESULTS_FILE_PATH = '../results.pkl'


def set_minimum(func: str = "sphere"):
    global globalMinimum
    if func == "sphere":
        globalMinimum = 0.0
    if func == "easom":
        globalMinimum = -1.0
    if func == "beale":
        globalMinimum = 0.0


def eval_worm(worm: np.array, func: str = "sphere"):
    if func == "sphere":
        return np.sum(np.square(worm[0]))
    elif func == "easom":
        return -np.cos(worm[0]) * np.cos(worm[1]) * \
               np.exp(-np.square(worm[0] - np.pi) - np.square(worm[1] - np.pi))
    elif func == "beale":
        return np.square(1.5 - worm[0] + worm[0] * worm[1]) + \
               np.square(2.25 - worm[0] + worm[0] * np.square(worm[1])) + \
               np.square(2.625 - worm[0] + worm[0] * np.power(worm[1], 3))
    else:
        raise FunctionNotDefined


def fitness_sort(worms: np.array, func: str = "sphere"):
    """
    Sorts the worm array passed in based on the evaluation function.

    :param np.array worm: array of worms.
    """
    intermediary_list = list(worms)
    intermediary_list.sort(key=lambda worm: np.abs(eval_worm(worm, func) - globalMinimum))
    print([(x, np.abs(globalMinimum - eval_worm(x, func))) for x in intermediary_list])
    # print("sorted")
    # print(intermediary_list)
    return np.array(intermediary_list)


def roulette_wheel_2(worms: np.array, fitness_func: str):
    """

    :param worms: The full worms array used to select parents for reproduction.
    :param fitness_func: The name of the fitness function used for worm evaluation.
    :return:
    :rtype (np.array, np.array)
    """
    roulette_array = np.zeros(worms.shape[0])
    fitness_sum = 0.0
    for worm_index in range(worms.shape[0]):
        worm_fitness = 1 / (1 + np.abs(globalMinimum - eval_worm(worms[worm_index], fitness_func)))
        fitness_sum += worm_fitness
        roulette_array[worm_index] = worm_fitness
    roulette_array /= fitness_sum

    assert np.isclose(np.sum(roulette_array), 1), \
        roulette_array

    chosen_worm_indices = []
    for _ in range(0, 2):
        sample = np.random.ranf()
        fitness_index = 0
        sum_fitness = roulette_array[0]
        while sample > sum_fitness:
            fitness_index += 1
            sum_fitness += roulette_array[fitness_index]
        chosen_worm_indices.append(fitness_index)

    return worms[chosen_worm_indices[0]], worms[chosen_worm_indices[1]]


def reproduction1(worm: np.array, dim_bounds: (int, [float]), sim_factor: float):
    """

    :param np.array worm: Worm reproducing by method 1.
    :param (int, [float]) dim_bounds: Dimensional bounds represented by max-min tuple.
    :param float sim_factor: Parameter determining how far child will spawn from parent.
    """
    child_worm = np.zeros((worm.shape[0]))
    for dim in range(len(worm)):
        child_worm[dim] = dim_bounds[1][1] + dim_bounds[1][0] - sim_factor * worm[dim]

    return child_worm


def reproduction2(worms: np.array, fitness_func: str):
    """
    Reproduction method 2: uniform crossover with two parents and two intermediate children

    :param worms: The full worms array used to select parents for reproduction.
    :param fitness_func: The name of the fitness function used for worm evaluation.
    :return: Child worm produced by reproduction2.
    :rtype np.array:
    """
    parent1, parent2 = roulette_wheel_2(worms, fitness_func)
    intermediate_child1 = np.zeros(worms[0].shape[0])
    intermediate_child2 = np.zeros(worms[0].shape[0])
    for dim in range(parent1.shape[0]):
        if np.random.ranf() < 0.5:
            intermediate_child1[dim] = parent1[dim]
            intermediate_child2[dim] = parent2[dim]
        else:
            intermediate_child1[dim] = parent2[dim]
            intermediate_child2[dim] = parent2[dim]
    fitness1 = 1 / (1 + eval_worm(intermediate_child1, fitness_func))
    fitness2 = 1 / (1 + eval_worm(intermediate_child2, fitness_func))
    weight1 = fitness2 / (fitness1 + fitness2)
    weight2 = fitness1 / (fitness1 + fitness2)

    return weight1 * intermediate_child1 + weight2 * intermediate_child2


def random_other_worm(worms: np.array, worm_index: int):
    """
    Gets a random worm that differs from the worm in question.

    :param np.array worms: Array of worms represented by their position.
    :param int worm_index: Index of this worm, that we want to avoid.
    :return: A worm in the worms array different from the calling worm.
    :rtype np.array:
    """
    random_index = np.random.randint(len(worms))
    while random_index == worm_index:
        random_index = np.random.randint(len(worms))
    return worms[random_index]


def cauchy_mutate(worm: np.array, worms: np.array):
    """

    :param np.array worm: The worm to mutate via cauchy mutation.
    :param np.array worms: Worms used to calculate weight vector for cauchy mutation.
    :return: The worm after mutation.
    :rtype np.array:
    """
    weights = np.sum(worms, axis=0)
    weights /= len(worms)
    for x in range(0, len(worm)):
        cauchy_var = np.random.standard_cauchy()
        worm[x] += weights[x] * cauchy_var
    return worm


def EWO(worm_population: int, worms_kept: int, max_generations: int,
        dim_bounds: (int, [float]), fitness_func: str = "sphere", sim_factor: float = 0.98, cool_factor: float = 0.9):
    """

    :param int worm_population: Total number of worms in the population
    :param int worms_kept: The number of worms that do not undergo reproduction 2,
    and number that undergo cauchy mutation.
    :param int max_generations: Maximum number of worm generations before algorithm ends.
    :param (int, (float, float)) dim_bounds: Integer representing dimensions and (min, max) values of dimensions
    :param float sim_factor: Similarity factor for worm reproduction
    :param float cool_factor: Cooling factor
    :return:
    """
    set_minimum(fitness_func)
    worms = np.zeros((worm_population, dim_bounds[0]))
    prop_factor = 1.0
    for worm_index in range(worms.shape[0]):
        for dimension in range(dim_bounds[0]):
            worms[worm_index][dimension] = random.uniform(dim_bounds[1][0], dim_bounds[1][1])
    try:
        for generation in range(0, max_generations):
            for worm in worms:
                if eval_worm(worm) == globalMinimum:
                    raise MinimumReached
            worms = fitness_sort(worms, fitness_func)
            for worm_index in range(len(worms)):
                this_worm = worms[worm_index]
                child_worm1 = reproduction1(this_worm, dim_bounds, sim_factor)
                if worm_index > worms_kept:
                    child_worm2 = reproduction2(worms, fitness_func)
                else:
                    child_worm2 = random_other_worm(this_worm, worm_index)
                worms[worm_index] = prop_factor * child_worm1 + (1 - prop_factor) * child_worm2

            worms = fitness_sort(worms, fitness_func)
            # Cauchy mutation of worms
            for worm_index in range(worms_kept, len(worms)):
                worms[worm_index] = cauchy_mutate(worms[worm_index], worms)

            # Scale the proportion factor
            prop_factor *= cool_factor
        # print(worms[0])
        return eval_worm(worms[0], fitness_func)

    except (MinimumReached, BadDimension, FunctionNotDefined) as error:
        print(error)
        pass


def run_sphere():
    sphere_dims = (30, (-5.12, 5.12))
    return EWO(50, 10, 50, sphere_dims, fitness_func="sphere")


def run_easom():
    easom_dims = (2, (-100, 100))
    return EWO(50, 2, 50, easom_dims, fitness_func="easom")


def run_beale():
    beale_dims = (2, (-4.5, 4.5))
    return EWO(50, 2, 50, beale_dims, fitness_func="beale")


if __name__ == "__main__":

    start_time = time.time_ns()
    easom_dims = (2, (-100, 100))
    beale_dims = (2, (-4.5, 4.5))

    num_cores = multiprocessing.cpu_count()

    sphere_results = Parallel(n_jobs=num_cores)(delayed(run_sphere)() for _ in range(NUM_ITERATIONS))
    easom_results = Parallel(n_jobs=num_cores)(delayed(run_easom)() for _ in range(NUM_ITERATIONS))
    beale_results = Parallel(n_jobs=num_cores)(delayed(run_beale)() for _ in range(NUM_ITERATIONS))
    results = {
        'sphere': sphere_results,
        'easom': easom_results,
        'beale': beale_results
    }

    for key in results:
        results[key].sort()

    end_time = time.time_ns()
    pprint(results)
    print("Took: %.3fs" % ((end_time - start_time) / 1e9))

    # save result dict to file (using python pickle lib)
    save_result(results)
