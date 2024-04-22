# Based on https://github.com/aimagelab/mammoth
# and https://github.com/rahullabs/FIXR_Public.git
import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        li.append(results[i-1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


def average_incremental_accuracy(results):
    n_tasks = len(results)
    task_accuracies = []

    # 确保所有任务结果长度一致
    for i in range(n_tasks):
        results[i] += [0.0] * (n_tasks - len(results[i]))

    # 计算每个任务的平均性能
    for result in results:
        task_accuracies.append(np.mean(result))

    # 计算所有任务的平均性能
    aia = np.mean(task_accuracies)
    return aia
