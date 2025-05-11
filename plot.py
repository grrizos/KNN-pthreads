import matplotlib.pyplot as plt

num_of_CANDQ = [1000, 5000, 10000, 50000, 100000]
times = [0.01013, 1.08483, 3.7479, 80.4492 ,321.925]  # replace with your own results

plt.plot(num_of_CANDQ, times, marker='o')
plt.xlabel('Number of C/Q')
plt.ylabel('Execution Time (seconds)')
plt.title('KNN Algorithm Time vs Number of C/Q')
plt.grid(True)
plt.savefig("knn_speed_plot_search.png")
plt.show()
