from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = list(range(1, 101))
    np.random.shuffle(data)
    #print(data)
    chunks = np.array_split(data, size)
else:
    chunks = None

local_data = comm.scatter(chunks, root=0)

local_sort = sorted(local_data)

print(f"End of local sort for process {rank}")

all_sort = comm.gather(local_sort, root=0)
#print(all_sort)

if rank == 0:
    total_sort = sorted(all_sort[0])
    #print()
    #print(total_sort)