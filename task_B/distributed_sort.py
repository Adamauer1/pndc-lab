from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys
import heapq
#sys.setrecursionlimit(10000)


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chunks = None

if rank == 0:
    data = pd.read_csv("random_data_200.csv", header=None).values.flatten()[1:]
    chunks = np.array_split(data, size)

total_compute_time = MPI.Wtime()


scatter_time = MPI.Wtime()
chunk = comm.scatter(chunks, root=0)
scatter_time = MPI.Wtime() - scatter_time
compute_time = MPI.Wtime()
chunk_sorted = quick_sort(chunk)
compute_time = MPI.Wtime() - compute_time
gather_time = MPI.Wtime()
max_local_sort_time = comm.reduce(compute_time, op=MPI.MAX, root=0)
sorted_chunks = comm.gather(chunk_sorted, root=0)
gather_time = MPI.Wtime() - gather_time


if rank == 0:
    final_merge_time = MPI.Wtime()
    #sorted_data = quick_sort(np.concatenate(sorted_chunks))
    #sorted_data = merge_sorted_arrays(sorted_chunks)
    sorted_data = list(heapq.merge(*sorted_chunks))
    final_merge_time = MPI.Wtime() - final_merge_time
    total_compute_time = MPI.Wtime() - total_compute_time
    print(max_local_sort_time + final_merge_time)
    print(total_compute_time - (scatter_time + gather_time))
    #print(sorted_data)
    #@standard_sort_time = MPI.Wtime()
    #data = quick_sort(data)
    #standard_sort_time = MPI.Wtime() - standard_sort_time
    #print(f"Standard sort time: {standard_sort_time} seconds.")
    #print(f"Process {rank} sorted data in {compute_time} seconds.")
    print(f"Scatter time: {scatter_time} seconds.")
    print(f"Gather time: {gather_time} seconds.")
    print(f"Total data transfer time: {scatter_time + gather_time} seconds.")
    print(f"Total compute time: {total_compute_time} seconds.")
    #print(f"Final merge time: {final_merge_time} seconds.")
    is_sorted = sorted_data == sorted(sorted_data)
    print(f"Is the data sorted? {'Yes' if is_sorted else 'No'}")
    #print to file5
    # with open("experiment_results/7p/data_200_results/experiment_4.txt", "w") as f:
    #     #f.write(f"Is the data sorted? {'Yes' if is_sorted else 'No'}\n")
    #     f.write(f"{scatter_time}\n")
    #     f.write(f"{gather_time}\n")
    #     f.write(f"{scatter_time + gather_time}\n")
    #     f.write(f"{total_compute_time}\n")
    #     #f.write(f"Final merge time: {final_merge_time} seconds.\n")


