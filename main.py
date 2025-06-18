from data import *
from clustering import *

IS_SAMPLE = False
FILE_FORMAT = "pdf"
CSV_PATH = "london_sample_500.csv" if IS_SAMPLE else "london.csv"

print("Part A: ")
df = load_data(CSV_PATH)
df = add_new_columns(df)
data_analysis(df)

print()
print("Part B: ")
df = load_data(CSV_PATH)
data = transform_data(df, ["cnt", "t1"])

ks = [2, 3, 5]
for i, k in enumerate(ks):
    labels, centroids = kmeans(data, k)

    print(f"k = {k}")
    print(np.array_str(centroids, precision=3, suppress_small=True))
    if i != len(ks) - 1:
        print()

    visualize_results(data, labels, centroids, f"plot_{k}{'_sample' if IS_SAMPLE else ''}.{FILE_FORMAT}")
