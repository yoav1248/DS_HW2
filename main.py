from data import *
from clustering import *

print("Part A: ")
df = load_data("london_sample_500.csv")
df = add_new_columns(df)
data_analysis(df)

print()
print("Part B: ")
df = load_data("london_sample_500.csv")
data = transform_data(df, ["cnt", "t1"])
for k in [2, 3, 5]:
    labels, centroids = kmeans(data, k)
    print(f"k = {k}")
    print(centroids.round(decimals = 3))
    print()