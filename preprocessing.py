import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dateutil import parser
import torch # Added torch import as it's used in the original notebook before preprocessing

def parse_mixed_date(date_str):
    try:
        return parser.parse(date_str, dayfirst=True)
    except Exception:
        return pd.NaT

def preprocess_timeseries_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.drop(labels=['Sprint'], axis=1)
    df['Scan date'] = df['Scan date'].astype(str).apply(parse_mixed_date) # Use the parse_mixed_date function
    df['Scan date'] = pd.to_datetime(df['Scan date']) # Ensure datetime type after parsing

    categorical_columns = ['Time to complete each version', 'Commit frequency', 'Type of environment', 'Repo']
    group_sum_col = 'Change code line number'
    target_col = 'Evaluate'
    date_col = 'Scan date'

    all_dfs = []
    for repo, group in df.groupby('Repo'):
        # group by date to avoid duplicates before reindexing
        group = group.groupby(date_col).agg({
            group_sum_col: 'sum',
            'Number of vulnerable modules': 'mean',
            'Number of people involved in development': 'mean',
            'Number of libraries detected errors': 'mean',
            'Number of potential weaknesses': 'mean',
            'Severity of the threat': 'mean',
            'Number of environmental configuration vulnerabilities': 'mean',
            target_col: 'first',
            **{col: 'first' for col in categorical_columns}
        }).sort_index()

        # create full date index and reindex
        all_dates = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
        group = group.reindex(all_dates, method='ffill')
        group[date_col] = group.index
        group['Repo'] = repo

        # identify filled-in rows and set code line to 0
        original_dates = df[df['Repo'] == repo]['Scan date'].unique()
        group[group_sum_col] = group[group_sum_col].where(group.index.isin(original_dates), 0)

        all_dfs.append(group)

    df_filled = pd.concat(all_dfs).reset_index(drop=True)

    # Ensure the target column is integer type
    df_filled[target_col] = df_filled[target_col].astype(int)

    # encode categoricals
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_filled[col] = le.fit_transform(df_filled[col])
        label_encoders[col] = le

    # normalize features (excluding Scan date, Repo, Evaluate)
    feature_cols = df_filled.columns.difference([date_col, 'Repo', target_col])
    scaler = StandardScaler()
    df_filled[feature_cols] = scaler.fit_transform(df_filled[feature_cols])

    # final outputs
    X = df_filled.drop(columns=[date_col, 'Repo', target_col])
    y = df_filled[target_col]
    meta = df_filled[[date_col, 'Repo']]

    return X, y, meta, label_encoders, scaler

# Add the initial data loading and mapping from the notebook for consistency
if __name__ == "__main__":
    df = pd.read_csv('resources/data_v3.csv')

    # Apply the parse_mixed_date function and convert to datetime
    df['Scan date'] = df['Scan date'].astype(str).apply(parse_mixed_date)
    df['Scan date'] = pd.to_datetime(df['Scan date'], errors='coerce')

    df['Commit frequency'] = df['Commit frequency'].str.capitalize()

    label_map = {
        'Very low': 0,
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Very high': 4
    }
    df['Evaluate'] = df["Evaluate"].map(label_map)

    X, y, meta, encoders, scaler = preprocess_timeseries_dataframe(df)

    print("Preprocessing complete. X, y, meta, encoders, and scaler are available.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("meta shape:", meta.shape)
