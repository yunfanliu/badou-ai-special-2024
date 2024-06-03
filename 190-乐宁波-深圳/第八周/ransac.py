import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(samples, fitting_model, min_samples_required, max_iterations, inlier_threshold, min_inliers_count, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）

    iterations = 0
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers
        }
        if (alsoinliers样本点数目 > d)
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    iterations = 0
    best_fit = None
    lowest_error = np.inf
    inlier_indices = None
    while iterations < max_iterations:
        # Randomly select min_samples_required indices as potential inliers
        sample_indices, test_indices = random_partition(min_samples_required, samples.shape[0])
        potential_inliers = samples[sample_indices, :]
        test_samples = samples[test_indices, :]

        # Fit model to potential inliers
        trial_model = fitting_model.fit(potential_inliers)
        test_errors = fitting_model.get_error(test_samples, trial_model)

        # Find additional inliers within the inlier threshold
        additional_inlier_indices = test_indices[test_errors < inlier_threshold]
        additional_inliers = samples[additional_inlier_indices, :]

        if debug:
            print(f"Iteration {iterations}: Found {len(additional_inliers)} inliers.")

        # If enough inliers are found, consider the model a good fit and attempt to refine it
        if len(additional_inliers) > min_inliers_count:
            refined_data = np.concatenate((potential_inliers, additional_inliers))
            refined_model = fitting_model.fit(refined_data)
            refined_errors = fitting_model.get_error(refined_data, refined_model)
            current_error = np.mean(refined_errors)

            if current_error < lowest_error:
                best_fit = refined_model
                lowest_error = current_error
                inlier_indices = np.concatenate((sample_indices, additional_inlier_indices))

        iterations += 1

    if best_fit is None:
        raise ValueError("Did not meet fit acceptance criteria.")

    if return_all:
        return best_fit, {'inliers': inlier_indices}
    else:
        return best_fit


def random_partition(min_samples_required, total_samples):
    """
    Randomly partition indices into two groups: one with min_samples_required and the other with the rest.

    Parameters:
        min_samples_required (int): Number of samples required for the first group.
        total_samples (int): Total number of samples.

    Returns:
        tuple: Two arrays of indices, one for each group.
    """
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    indices_group1 = all_indices[:min_samples_required]
    indices_group2 = all_indices[min_samples_required:]
    return indices_group1, indices_group2


class LinearLeastSquaresModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        model_coefficients, residuals, rank, s = sl.lstsq(A, B)
        return model_coefficients

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_predicted = np.dot(A, model)
        error_per_point = np.sum((B - B_predicted) ** 2, axis=1)
        return error_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
