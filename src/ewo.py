import numpy as np
import random

globalMinimum = 0.0



def eval_worm(worm: np.array, func: str="sphere"):
    if func == "sphere":
        print(np.sum(np.square(worm)))
        return np.sum(np.square(worm))
    elif func == "eggholder":
        return -(worm[1] + 47) * np.sin(np.sqrt(worm[1] + 0.5*worm[0] + 47)) - \
                worm[0]*np.sin(np.sqrt(worm[0] - (worm[1] + 47)))
    elif func == "beale":
        return np.square(1.5 - worm[0] + worm[0]*worm[1]) + \
                np.square(2.25 - worm[0] + worm[0] * np.square(worm[1])) + \
                np.square(2.625 - worm[0] + worm[0] * np.power(worm[1], 3))
    else:
        raise FunctionNotDefined



def fitness_sort(worms: np.array, func: str="sphere"):
    """
    Sorts the worm array passed in based on the evaluation function.

    :param np.array worm: array of worms.
    """
    intermediary_list = list(worms)
    intermediary_list.sort(key=lambda worm: eval_worm(worm, func))
    return np.array(intermediary_list)

def roulette_wheel_2(worms: np.array, fitness_func: str):
    roulette_array = np.zeros(worms.shape[0])
    fitness_sum = 0.0
    for worm_index in range(worms.shape[0]):
        worm_fitness = eval_worm(worms[worm_index])
        fitness_sum += worm_fitness
        roulette_array[worm_index] = worm_fitness
    roulette_array / fitness_sum

    chosen_worm_indices = []
    for _ in range(0,2):
        sample = np.random.ranf()
        fitness_index = 0
        sum_fitness = roulette_array[0]
        while sample > sum_fitness:
            i += 1
            sum_fitness += roulette_array[1]
        chosen_worm_indices.append(fitness_index)

    return (worms[chosen_worm_indices[0]], worms[chosen_worm_indices[1]])






def reproduction1(worm: np.array, dim_bounds: np.array, sim_factor: float):
    """

    :param np.array worm: Worm reproducing by method 1.
    :param np.array dim_bounds: Dimensional bounds represented by max-min tuples.
    :param float sim_factor: Parameter determining how far child will spawn from parent.
    """
    child_worm = np.zeros((worm.shape[0], worm.shape[1]))
    for dim in range(len(worm)):
        child_worm[dim] = dim_bounds[dim][1] + dim_bounds[dim][0] - sim_factor * worm[dim]
    return child_worm

def reproduction2(worms: np.array, fitness_func: str):
    """

    :param np.array worm1: First parent.
    :param np.array worm2: Second parent.
    """




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





def EWO(worm_population: int, worms_kept: int, max_generations: int,
        dim_bounds: np.array, sim_factor: float=0.95, cool_factor: float=0.9):
    """

    :param int worm_population:
    :param int worms_kept:
    :param int max_generations:
    :param np.array dim_bounds: Array of tuples representing (min, max) values of each dimension
    :param float sim_factor: Similarity factor for worm reproduction
    :param float cool_factor: Cooling factor
    :return:
    """
    worms = np.zeros((worm_population, dim_bounds.shape[0]))
    prop_factor = 1.0
    for worm_index in range(worms.shape[0]):
        for dimension in range(dim_bounds.shape[0]):
            worms[worm_index][dimension] = random.uniform(dim_bounds[dimension][0], dim_bounds[dimension][1])
    try:
        for generation in range(0, max_generations):
            for worm in worms:
                if eval_worm(worm) == globalMinimum:
                    raise MinimumReached
            worms = fitness_sort(worms)
            for worm_index in range(len(worms)):




    except (MinimumReached, BadDimension, FunctionNotDefined) as error:
        print(error)
        pass

if __name__ == '__main__':
    EWO(100, 20, 1, np.array([[-10, 10], [-10, 10]]))
