import pprint

import numpy as np
from numpy.linalg import lstsq
from typing import Tuple, Optional, Dict, Any


def ransac(data: np.ndarray, model: Any, min_samples: int, max_iterations: int, threshold: float, min_inliers: int, debug: bool = False, return_all: bool = False) -> Tuple[
    np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Implements the RANSAC algorithm.

    Parameters:
        data (np.ndarray): Sample points.
        model (object): Hypothetical model to fit data.
        min_samples (int): Minimum number of samples required to fit the model.
        max_iterations (int): Maximum number of iterations.
        threshold (float): Threshold to determine if a point fits the model.
        min_inliers (int): Minimum number of inliers required for a valid model.
        debug (bool): If True, prints debug information.
        return_all (bool): If True, returns all data related to the fitting process.

    Returns:
        Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]: The best fitting model parameters and a dictionary containing inliers if return_all is True.

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

        inliers_mask = test_errors < threshold
        additional_inliers = test_data[inliers_mask]

        if debug:
            print(f"Iteration {iteration}: Found {len(additional_inliers)} inliers.")

        if len(additional_inliers) > min_inliers:
            refined_data = np.vstack((potential_inliers, additional_inliers))
            refined_model = model.fit(refined_data)
            current_error = np.mean(model.get_error(refined_data, refined_model))

            if current_error < lowest_error:
                best_model = refined_model
                lowest_error = current_error
                best_inliers = np.concatenate((sample_indices, test_indices[inliers_mask]))

    if best_model is None:
        raise ValueError("Did not meet fit acceptance criteria.")

    if return_all:
        return best_model, {'inliers': best_inliers}
    else:
        return best_model


def random_partition(min_samples: int, total_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly partition indices into two groups: one with min_samples and the other with the rest.

    Parameters:
        min_samples (int): Number of samples required for the first group.
        total_samples (int): Total number of samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of indices, one for each group.
    """
    indices = np.random.permutation(total_samples)
    return indices[:min_samples], indices[min_samples:]


class LinearLeastSquaresModel:
    def __init__(self, input_columns: np.ndarray, output_columns: np.ndarray, debug: bool = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data: np.ndarray) -> np.ndarray:
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        model, _, _, _ = lstsq(A, B, rcond=None)
        return model

    def get_error(self, data: np.ndarray, model: np.ndarray) -> np.ndarray:
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        B_predicted = A @ model
        return np.sum((B - B_predicted) ** 2, axis=1)


def generate_data(n_samples, n_inputs, n_outputs):
    pass


def test():
    np.random.seed(42)
    n_samples = 500  # 正常样本数量
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 精确矩阵A
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = A_exact @ perfect_fit  # 点乘

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    n_outliers = 100  # 异常样本数量
    outlier_indices = np.random.choice(np.arange(A_noisy.shape[0]), n_outliers, replace=False)
    A_noisy[outlier_indices] = 20 * np.random.random((n_outliers, n_inputs))  # 均匀分布的噪声
    B_noisy[outlier_indices] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 正态分布的噪声

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = np.arange(n_inputs)
    output_columns = np.arange(n_inputs, n_inputs + n_outputs)

    linear_fit, _, _, _ = lstsq(all_data[:, input_columns], all_data[:, output_columns], rcond=None)  # 最小二乘法拟合，找到一组参数

    model = LinearLeastSquaresModel(input_columns, output_columns, debug=False)  # 实例化模型
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=False, return_all=True)  # 随机采样一致性算法进行拟合

    import matplotlib.pyplot as plt

    sorted_indices = np.argsort(A_exact[:, 0])
    A_sorted = A_exact[sorted_indices]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    plt.plot(A_sorted[:, 0], A_sorted @ ransac_fit, label='RANSAC fit')
    plt.plot(A_sorted[:, 0], A_sorted @ perfect_fit, label='exact system')
    plt.plot(A_sorted[:, 0], A_sorted @ linear_fit, label='linear fit')
    plt.legend()  # 图例
    plt.show()


if __name__ == "__main__":
    test()
