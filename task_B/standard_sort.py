import pandas as pd
import time

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


data = pd.read_csv("random_data_1000.csv", header=None).values.flatten()[1:]

compute_time = time.time()
sorted_data = quick_sort(data)
compute_time = time.time() - compute_time
print(f"Data sorted in {compute_time} seconds.")
