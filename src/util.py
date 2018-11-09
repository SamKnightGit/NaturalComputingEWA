import pickle
from src.ewo import RESULTS_FILE_PATH

def save_result(results: dict):
    with open(RESULTS_FILE_PATH, 'wb') as f:
        pickle.dump(results, f)


def load_result():
    with open(RESULTS_FILE_PATH, 'rb') as f:
        return pickle.load(f)
