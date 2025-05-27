from flask import Flask, render_template, request
import pandas as pd
import torch
import pickle
import os
import plotly.graph_objs as go
from plotly.offline import plot
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


from model import MyModel
from preprocessing import preprocess_timeseries_dataframe, parse_mixed_date
from prediction import predict_with_uncertainty, bayesian_predictive_entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'resources', 'data_v3.csv')


try:
    input_features = 10
    timesteps = 5
    num_classes = 5

    model = MyModel(input_features=input_features, timesteps=timesteps, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("Model, encoders, and scaler loaded successfully.")

except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    model = None
    encoders = None
    scaler = None

@app.route('/')
def index():
    try:
        df_full = pd.read_csv(DATA_PATH)

        df_full['Scan date'] = df_full['Scan date'].astype(str).apply(parse_mixed_date)
        df_full['Scan date'] = pd.to_datetime(df_full['Scan date'], errors='coerce')
        df_full['Commit frequency'] = df_full['Commit frequency'].str.capitalize()
        label_map = { 'Very low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very high': 4 }
        df_full['Evaluate'] = df_full["Evaluate"].map(label_map)


        if 'Repo' in encoders:
            temp_repo_encoder = LabelEncoder()
            temp_repo_encoder.fit(df_full['Repo'])
            repo_names = temp_repo_encoder.classes_.tolist()
        else:
            repo_names = df_full['Repo'].unique().tolist()

        repo_names.sort()
    except Exception as e:
        print(f"Error loading data or getting repo names: {e}")
        repo_names = []

    return render_template('index.html', repo_names=repo_names)

@app.route('/assess', methods=['POST'])
def assess():
    if model is None or encoders is None or scaler is None:
        return "Error: Model or preprocessing objects not loaded.", 500

    repo_name = request.form.get('repo_name')
    if not repo_name:
        return "Error: Repository name not provided.", 400

    try:
        df_full = pd.read_csv(DATA_PATH)

        df_full['Scan date'] = df_full['Scan date'].astype(str).apply(parse_mixed_date)
        df_full['Scan date'] = pd.to_datetime(df_full['Scan date'], errors='coerce')
        df_full['Commit frequency'] = df_full['Commit frequency'].str.capitalize()
        label_map = { 'Very low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very high': 4 }
        df_full['Evaluate'] = df_full["Evaluate"].map(label_map)

        df_repo = df_full[df_full['Repo'] == repo_name].copy()

        if df_repo.empty:
            return render_template('index.html', repo_names=[], error_message=f"No data found for repository: {repo_name}")

        # Extract repository information
        num_commits = len(df_repo)
        date_range_start = df_repo['Scan date'].min().strftime('%Y-%m-%d') if not df_repo.empty else 'N/A'
        date_range_end = df_repo['Scan date'].max().strftime('%Y-%m-%d') if not df_repo.empty else 'N/A'

        # Get unique environment types and decode them
        unique_environments_encoded = df_repo['Type of environment'].unique()
        # Need to handle cases where 'Type of environment' might not be in the encoder classes
        # This might happen if a repo only has data points with environment types not seen in training
        # A safer approach is to use the original df_repo before any encoding attempts
        original_environments = df_repo['Type of environment'].unique().tolist()
        type_of_environment = ", ".join(map(str, original_environments)) # Display all unique original environment types


        date_col = 'Scan date'
        group_sum_col = 'Change code line number'
        target_col = 'Evaluate'
        categorical_columns = ['Time to complete each version', 'Commit frequency', 'Type of environment', 'Repo']


        group = df_repo.groupby(date_col).agg({
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

        all_dates = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
        group = group.reindex(all_dates, method='ffill')
        group[date_col] = group.index
        group['Repo'] = repo_name

        original_dates = df_repo[date_col].unique()
        group[group_sum_col] = group[group_sum_col].where(group.index.isin(original_dates), 0)

        df_processed_repo = group.reset_index(drop=True)

        df_processed_repo[target_col] = df_processed_repo[target_col].astype(int)


        for col in categorical_columns:
            try:
                df_processed_repo[col] = encoders[col].transform(df_processed_repo[col])
            except ValueError as e:
                print(f"Warning: Category not seen during training for column {col}: {e}")


        feature_cols = df_processed_repo.columns.difference([date_col, 'Repo', target_col])
        df_processed_repo[feature_cols] = scaler.transform(df_processed_repo[feature_cols])

        timesteps = 5
        X_repo = df_processed_repo.drop(columns=[date_col, 'Repo', target_col])
        y_repo = df_processed_repo[target_col]
        meta_repo = df_processed_repo[[date_col, 'Repo', target_col]]

        X_sequences = []
        y_sequences = []
        meta_sequences = []

        for i in range(len(X_repo) - timesteps + 1):
            window_idx = range(i, i + timesteps)
            x_seq = torch.tensor(X_repo.iloc[window_idx].values, dtype=torch.float32)
            y_seq = torch.tensor(y_repo.iloc[window_idx].values, dtype=torch.long)
            meta_seq = meta_repo.iloc[window_idx].iloc[-1].to_dict()
            X_sequences.append(x_seq)
            y_sequences.append(y_seq)
            meta_sequences.append(meta_seq)

        if not X_sequences:
             return render_template('index.html', repo_names=[], error_message=f"Not enough data points ({len(df_repo)} commits) for repository: {repo_name} to form sequences of length {timesteps}.")

        X_batch = torch.stack(X_sequences)

        mean_preds, std_preds = predict_with_uncertainty(model, X_batch, runs=50)
        mean_probs = mean_preds[:, -1, :]
        std_probs = std_preds[:, -1, :]

        _, entropy = bayesian_predictive_entropy(model, X_batch, runs=50)
        entropy_last_timestep = entropy[:, -1]


        prediction_dates = [meta['Scan date'] for meta in meta_sequences]
        actual_risk_levels = [meta['Evaluate'] for meta in meta_sequences]

        results = []
        class_names = ["Very low", "Low", "Medium", "High", "Very high"]
        for i in range(len(mean_probs)):
            predicted_class_index = mean_probs[i].argmax()
            predicted_risk = class_names[predicted_class_index]
            confidence = mean_probs[i][predicted_class_index]
            uncertainty = std_probs[i][predicted_class_index]
            entropy_val = entropy_last_timestep[i]
            date = prediction_dates[i].strftime('%Y-%m-%d')
            actual_risk = class_names[actual_risk_levels[i]]

            results.append({
                'date': date,
                'predicted_risk': predicted_risk,
                'confidence': f"{confidence:.4f}",
                'uncertainty': f"{uncertainty:.4f}",
                'entropy': f"{entropy_val:.4f}",
                'actual_risk': actual_risk
            })


        risk_trace = go.Scatter(
            x=[r['date'] for r in results],
            y=[class_names.index(r['predicted_risk']) for r in results],
            mode='lines+markers',
            name='Predicted Risk Level',
            line=dict(color='blue')
        )
        actual_risk_trace = go.Scatter(
            x=[r['date'] for r in results],
            y=[class_names.index(r['actual_risk']) for r in results],
            mode='markers',
            name='Actual Risk Level',
            marker=dict(color='red', symbol='x')
        )
        risk_layout = go.Layout(
            title=f'Predicted Risk Level Over Time for {repo_name}',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Risk Level', tickvals=list(range(len(class_names))), ticktext=class_names),
            hovermode='closest'
        )
        risk_fig = go.Figure(data=[risk_trace, actual_risk_trace], layout=risk_layout)
        risk_plot_div = plot(risk_fig, output_type='div', include_plotlyjs=False)

        entropy_trace = go.Scatter(
            x=[r['date'] for r in results],
            y=[float(r['entropy']) for r in results],
            mode='lines+markers',
            name='Predictive Entropy',
            line=dict(color='purple')
        )
        entropy_layout = go.Layout(
            title=f'Predictive Uncertainty (Entropy) Over Time for {repo_name}',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Entropy'),
            hovermode='closest'
        )
        entropy_fig = go.Figure(data=[entropy_trace], layout=entropy_layout)
        entropy_plot_div = plot(entropy_fig, output_type='div', include_plotlyjs=False)


        df_full_for_dropdown = pd.read_csv(DATA_PATH)
        df_full_for_dropdown['Scan date'] = df_full_for_dropdown['Scan date'].astype(str).apply(parse_mixed_date)
        df_full_for_dropdown['Scan date'] = pd.to_datetime(df_full_for_dropdown['Scan date'], errors='coerce')
        df_full_for_dropdown['Commit frequency'] = df_full_for_dropdown['Commit frequency'].str.capitalize()
        df_full_for_dropdown['Evaluate'] = df_full_for_dropdown["Evaluate"].map(label_map)

        if 'Repo' in encoders:
            temp_repo_encoder = LabelEncoder()
            temp_repo_encoder.fit(df_full_for_dropdown['Repo'])
            repo_names_for_dropdown = temp_repo_encoder.classes_.tolist()
        else:
            repo_names_for_dropdown = df_full_for_dropdown['Repo'].unique().tolist()
        repo_names_for_dropdown.sort()


        return render_template(
            'index.html',
            repo_name=repo_name,
            results=results,
            risk_plot_div=risk_plot_div,
            entropy_plot_div=entropy_plot_div,
            repo_names=repo_names_for_dropdown,
            num_commits=num_commits,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            type_of_environment=type_of_environment
        )

    except Exception as e:
        print(f"Error during assessment: {e}")
        try:
            df_full_for_dropdown = pd.read_csv(DATA_PATH)
            df_full_for_dropdown['Scan date'] = df_full_for_dropdown['Scan date'].astype(str).apply(parse_mixed_date)
            df_full_for_dropdown['Scan date'] = pd.to_datetime(df_full_for_dropdown['Scan date'], errors='coerce')
            df_full_for_dropdown['Commit frequency'] = df_full_for_dropdown['Commit frequency'].str.capitalize()
            df_full_for_dropdown['Evaluate'] = df_full_for_dropdown["Evaluate"].map(label_map)

            if 'Repo' in encoders:
                temp_repo_encoder = LabelEncoder()
                temp_repo_encoder.fit(df_full_for_dropdown['Repo'])
                repo_names_for_dropdown = temp_repo_encoder.classes_.tolist()
            else:
                repo_names_for_dropdown = df_full_for_dropdown['Repo'].unique().tolist()
            repo_names_for_dropdown.sort()
        except Exception as e_inner:
             print(f"Error loading data for dropdown in error handler: {e_inner}")
             repo_names_for_dropdown = []


        return render_template('index.html', repo_names=repo_names_for_dropdown, error_message=f"An error occurred during assessment: {e}")


if __name__ == '__main__':
    app.run(debug=True)
