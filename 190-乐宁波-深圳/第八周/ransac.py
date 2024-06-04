import numpy as np
import scipy.linalg as sl

def ransac(data, model, min_samples, max_iterations, threshold, min_inliers, debug=False, return_all=False):
    """
    Implements the RANSAC algorithm.

    Parameters:
        data (numpy.ndarray): Sample points.
        model (object): Hypothetical model to fit data.
        min_samples (int): Minimum number of samples required to fit the model.
        max_iterations (int): Maximum number of iterations.
        threshold (float): Threshold to determine if a point fits the model.
        min_inliers (int): Minimum number of inliers required for a valid model.
        debug (bool): If True, prints debug information.
        return_all (bool): If True, returns all data related to the fitting process.

    Returns:
        best_model (numpy.ndarray): The best fitting model parameters.
        Optional[dict]: A dictionary containing inliers if return_all is True.

    Raises:
        ValueError: If no acceptable model is found.
    """
    best_model = None
    lowest_error = np.inf
    best_inliers = None
    for iteration in range(max_iterations):
        sample_indices, test_indices = random_partition(min_samples, data.shape[0])
        potential_inliers = data[sample_indices]
        test_data = data[test_indices]

        trial_model = model.fit(potential_inliers)
        test_errors = model.get_error(test_data, trial_model)

        additional_inlier_indices = test_indices[test_errors < threshold]
        additional_inliers = data[additional_inlier_indices]

        if debug:
            print(f"Iteration {iteration}: Found {len(additional_inliers)} inliers.")

        if len(additional_inliers) > min_inliers:
            refined_data = np.concatenate((potential_inliers, additional_inliers))
            refined_model = model.fit(refined_data)
            refined_errors = model.get_error(refined_data, refined_model)
            current_error = np.mean(refined_errors)

            if current_error < lowest_error:
                best_model = refined_model
                lowest_error = current_error
                best_inliers = np.concatenate((sample_indices, additional_inlier_indices))

    if best_model is None:
        raise ValueError("Did not meet fit acceptance criteria.")

    if return_all:
        return best_model, {'inliers': best_inliers}
    else:
        return best_model

def random_partition(min_samples, total_samples):
    """
    Randomly partition indices into two groups: one with min_samples and the other with the rest.

    Parameters:
        min_samples (int): Number of samples required for the first group.
        total_samples (int): Total number of samples.

    Returns:
        tuple: Two arrays of indices, one for each group.
    """
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    return indices[:min_samples], indices[min_samples:]

class LinearLeastSquaresModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        model, _, _, _ = sl.lstsq(A, B)
        return model

    def get_error(self, data, model):
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        B_predicted = np.dot(A, model)
        return np.sum((B - B_predicted) ** 2, axis=1)

def test():
    np.random.seed(42)
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    n_outliers = 100
    all_indices = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_indices)
    outlier_indices = all_indices[:n_outliers]
    A_noisy[outlier_indices] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_indices] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=False)

    linear_fit, _, _, _ = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=False, return_all=True)

    import matplotlib.pyplot as plt

    sort_indices = np.argsort(A_exact[:, 0])
    A_sorted = A_exact[sort_indices]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    plt.plot(A_sorted[:, 0], np.dot(A_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    plt.plot(A_sorted[:, 0], np.dot(A_sorted, perfect_fit)[:, 0], label='exact system')
    plt.plot(A_sorted[:, 0], np.dot(A_sorted, linear_fit)[:, 0], label='linear fit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
