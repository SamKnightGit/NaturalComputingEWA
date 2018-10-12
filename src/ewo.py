import numpy as np
import random

def eval_worm(worm: np.array):
    return

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
    for worm_index in worms.shape[0]:
        for dimension in dim.shape[0]:
            worms[worm_index][dimension] = random.uniform(dim[dimension][0], dim[dimension][1])
    for generation in range(0, max_generations):
        pass