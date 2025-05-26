from data import *

print("Part A: ")
df = load_data("london_sample_500.csv")
df = add_new_columns(df)
data_analysis(df)