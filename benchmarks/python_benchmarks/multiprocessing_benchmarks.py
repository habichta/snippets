from multiprocessing import Pool
from time import time

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


def square(number):
    return number * number


def identity(number):
    return number


def plot(x_values, y_values):
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])

    fig.update_layout(
        title="Benchmark for Multiprocessing",
        xaxis=dict(title="Experiment"),
        yaxis=dict(title="Times"),
    )

    fig.show()


def benchmark(num_complex_numbers, num_processes, nr_of_tasks):
    complex_array = np.random.rand(num_complex_numbers).astype(np.complex64)
    task_list = []
    with Pool(processes=num_processes) as pool:
        start = time()
        for _ in range(nr_of_tasks):
            task = pool.apply_async(identity, (complex_array,))
            task_list.append(task)

        _ = [task.get() for task in task_list]
        end = time()
        pool.close()
        pool.join()

        return end - start


if __name__ == "__main__":
    benchmarks = [
        (25000, 4, 8),
        (75000, 4, 8),
    ]

    all_result_times = []
    for _ in tqdm(range(500)):
        result_times = []
        for b in benchmarks:
            times = benchmark(*b)
            result_times.append(times)
        all_result_times.append(result_times)

    all_result_times = np.mean(np.array(all_result_times), axis=0)

    plot([str(b) for b in benchmarks], all_result_times)
