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

    :param np.array worm: array of worms
    """
    intermediary_list = list(worms)
    intermediary_list.sort(key=lambda worm: eval_worm(worm, func))
    return np.array(intermediary_list)


def EWO(worm_population: int, worms_kept: int, max_generations: int,
        dim: np.array, sim_factor: float=0.95, cool_factor: float=0.9):
    """

    :param int worm_population:
    :param int worms_kept:
    :param int max_generations:
    :param np.array dim: Array of tuples representing (min, max) values of each dimension
    :param float sim_factor: Similarity factor for worm reproduction
    :param float cool_factor: Cooling factor
    :return:
    """
    worms = np.zeros((worm_population, dim.shape[0]))
    prop_factor = 1.0
    for worm_index in range(worms.shape[0]):
        for dimension in range(dim.shape[0]):
            worms[worm_index][dimension] = random.uniform(dim[dimension][0], dim[dimension][1])
    try:
        for generation in range(0, max_generations):
            for worm in worms:
                if eval_worm(worm) == globalMinimum:
                    raise MinimumReached
            worms = fitness_sort(worms)
            for worm_index in range(worms.shape[0])


    except (MinimumReached, BadDimension, FunctionNotDefined) as error:
        print(error)
        pass

if __name__ == '__main__':
    EWO(100, 20, 1, np.array([[-10, 10], [-10, 10]]))
