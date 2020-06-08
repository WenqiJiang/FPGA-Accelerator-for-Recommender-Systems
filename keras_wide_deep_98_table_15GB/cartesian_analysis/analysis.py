import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dim_size = np.load("embedding_dimension.npy") # (dimension_of_each_vec, total_entries)
    print(dim_size)

    total_params = np.sum([d * s for d, s in dim_size])

    memory_access_reduction_list = []
    storage_overhead_list = []

    for i in range(1, int(len(dim_size) / 2) + 1):
        # cartesian product the first 2 * i embedding tables
        # product size = #entry1 * # entry2 * (dim1 + dim2)
        # principle -> the very beginning multiply the very last
        prod_size = 0
        top_arr = dim_size[:2 * i]
        for j in range(i):
            table1 = top_arr[j]
            table2 = top_arr[-1 -j]
            prod_size += table1[1] * table2[1] * (table1[0] + table2[0])
        memory_access_reduction = (len(dim_size) - i) / len(dim_size)
        storage_overhead = (total_params + prod_size) / total_params
        memory_access_reduction_list.append(memory_access_reduction)
        storage_overhead_list.append(storage_overhead)
        print("Top {}:\t{}M params\tMemory Access: {:.2f}\tStorage: {:.2f}".format(
            2 * i, prod_size/1e6, memory_access_reduction, storage_overhead))

    x = storage_overhead_list
    y = memory_access_reduction_list
    plt.plot(x, y)
    plt.xlabel('Storage Consumption (compared with no Cartesian product')
    plt.ylabel('Random DRAM access ratio')
    plt.title("Storage-Memory Access Trade-off")
    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, '({:.2f}, {:.2f})'.format(i_x, i_y))

    # plt.xscale("log")
    plt.show()