import pandas as pd
from datetime import datetime

def load_data(path):
    return pd.read_csv(path)

def add_new_columns(df):
    # override df with a copy so original dataframe is not modified
    df = df.copy()

    season_names = ['spring', 'summer', 'fall', 'winter']
    get_season_name = lambda i: season_names[i]
    df['season_name'] = df['season'].apply(get_season_name)

    datetime_series = (df["timestamp"].apply
                       (lambda timestamp_str: datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M")))
    df['Hour'] = datetime_series.apply(lambda datetime_obj: datetime_obj.hour)
    df['Day'] = datetime_series.apply(lambda datetime_obj: datetime_obj.day)
    df['Month'] = datetime_series.apply(lambda datetime_obj: datetime_obj.month)
    df['Year'] = datetime_series.apply(lambda datetime_obj: datetime_obj.year)

    df['is_weekend_holiday'] = df.apply(lambda record: record['is_holiday'] * 2 + record['is_weekend'] + 1,
                                        axis = 1)

    df['t_diff'] = df.apply(lambda record: record['t2'] - record['t1'], axis=1)

    return df

def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()


    feature_names = corr.columns

    pair_to_abs_corr_dict = {}

    for i, feature_name1 in enumerate(feature_names):
        for j, feature_name2 in enumerate(feature_names[i+1:]):
            pair_to_abs_corr_dict[(feature_name1, feature_name2)] = abs(corr.loc[feature_name1, feature_name2])

    items_by_abs_corr_desc = sorted(pair_to_abs_corr_dict.items(),
                               key = lambda item: abs(item[1]),
                               reverse = True)

    print("Highest correlated are: ")
    for i, (feature_pair, abs_corr) in enumerate(items_by_abs_corr_desc[:5]):
        print(f"{i+1}. {feature_pair} with {abs_corr:.6f}")
    print()

    print("Lowest correlated are: ")
    for i, (feature_pair, abs_corr) in enumerate(reversed(items_by_abs_corr_desc[-5:])):
        print(f"{i+1}. {feature_pair} with {abs_corr:.6f}")
    print()

    season_t_diff_means = df.groupby(["season_name"])["t_diff"].mean()
    season_t_diff_means["All"] = df["t_diff"].mean()
    for season_name, avg in season_t_diff_means.items():
        print(f"{season_name} average t_diff is {avg:.2f}")

