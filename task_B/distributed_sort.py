from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys
#sys.setrecursionlimit(10000)


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sorted_arrays(arrays):
    if not arrays:
        return []
    if len(arrays) == 1:
        return arrays[0]
    
    mid = len(arrays) // 2
    left = merge_sorted_arrays(arrays[:mid])
    right = merge_sorted_arrays(arrays[mid:])
    
    return merge_two_sorted_arrays(left, right)

# extend to work with n number of arrays
def merge_two_sorted_arrays(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chunks = None

if rank == 0:
    data = pd.read_csv("random_data_1000.csv", header=None).values.flatten()[1:]
    chunks = np.array_split(data, size)

total_compute_time = MPI.Wtime()


scatter_time = MPI.Wtime()
chunk = comm.scatter(chunks, root=0)
scatter_time = MPI.Wtime() - scatter_time
compute_time = MPI.Wtime()
chunk_sorted = quick_sort(chunk)
compute_time = MPI.Wtime() - compute_time

gather_time = MPI.Wtime()
sorted_chunks = comm.gather(chunk_sorted, root=0)
gather_time = MPI.Wtime() - gather_time


if rank == 0:
    final_merge_time = MPI.Wtime()
    #sorted_data = quick_sort(np.concatenate(sorted_chunks))
    sorted_data = merge_sorted_arrays(sorted_chunks)
    final_merge_time = MPI.Wtime() - final_merge_time
    total_compute_time = MPI.Wtime() - total_compute_time
    standard_sort_time = MPI.Wtime()
    data = quick_sort(data)
    standard_sort_time = MPI.Wtime() - standard_sort_time
    print(f"Standard sort time: {standard_sort_time} seconds.")
    print(f"Process {rank} sorted data in {compute_time} seconds.")
    print(f"Scatter time: {scatter_time} seconds.")
    print(f"Gather time: {gather_time} seconds.")
    print(f"Total compute time: {total_compute_time} seconds.")
    print(f"Final merge time: {final_merge_time} seconds.")
    # print("Sorted data:", sorted_data)
    # if sorted_data == sorted(sorted_data):
    #     print("The data is sorted correctly.")
    # else:
    #     print("The data is not sorted correctly.")

