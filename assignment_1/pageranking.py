import numpy as np


def is_close(first_vector, second_vector, abs_tolerance=1e-5):
    abs_vector_difference = np.fabs(np.subtract(first_vector, second_vector))
    average_vector_difference = np.average(abs_vector_difference)
    if average_vector_difference < abs_tolerance:
        return True
    else:
        return False


def page_ranking(input_array):
    current_vector = np.random.rand(input_array.shape[0])

    while True:
        new_vector = np.dot(current_vector, transition_matrix)
        new_vector_norm = np.linalg.norm(new_vector)
        new_vector = new_vector / new_vector_norm

        if is_close(new_vector, current_vector):
            break
        else:
            current_vector = new_vector

    return current_vector


if __name__ == "__main__":
    transition_matrix = np.random.rand(5, 5)
    print(page_ranking(transition_matrix))
