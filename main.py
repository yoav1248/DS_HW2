from data import *
from clustering import *

FILE_FORMAT = "pdf"
CSV_PATH = "london.csv"

print("Part A: ")
df = load_data(CSV_PATH)
df = add_new_columns(df)
data_analysis(df)

print()
print("Part B: ")
df = load_data(CSV_PATH)
data = transform_data(df, ["cnt", "t1"])
for k in [2, 3, 5]:
    labels, centroids = kmeans(data, k)

    print(f"k = {k}")
    print(np.array_str(centroids, precision=3, suppress_small=True))
    print()

    visualize_results(data, labels, centroids, f"plot_{k}.{FILE_FORMAT}")
