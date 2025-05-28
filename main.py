from data import *
from clustering import *

print("Part A: ")
df = load_data("london_sample_500.csv")
df = add_new_columns(df)
data_analysis(df)

print()
print("Part B: ")
df = load_data("london_sample_500.csv")
print(transform_data(df, ["cnt", "t1"]))