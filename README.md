# Machine Learning for Security Risk Detection in DevOps Pipelines

## Overview

This project develops a machine learning-based solution to detect security risks in DevOps pipelines, focusing on continuous integration and deployment (CI/CD) processes. By leveraging time-series data from DevOps pipelines, the system combines Long Short-Term Memory (LSTM) neural networks and Bayesian Neural Networks (BNN) to model security behavior patterns and quantify prediction uncertainty. A Flask-based web dashboard enables users to visualize risk levels and uncertainty for selected repositories, aiding DevSecOps engineers in identifying and addressing potential threats early.

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/thanh309/project-2-it3930e.git
    cd project-2-it3930e
    ```

2. **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    # Create a virtual environment (optional but recommended)
    conda create -n risk_dashboard_env
    conda activate risk_dashboard_env

    # Install packages
    pip install -r requirements.txt
    ```

3. **Prepare the model:**
    Run the preparation script to preprocess the data, train the model, and save the necessary files (`model.pth`, `encoders.pkl`, `scaler.pkl`) in the `model/` directory.

    ```bash
    python3 prepare_model.py
    ```

    Ensure you have the data file in the correct location before running this script.

## Running the Flask Application

1. **Navigate to the application directory:**

    ```bash
    cd risk_dashboard_app
    ```

2. **Set the Flask application file:**

    ```bash
    export FLASK_APP=app.py
    ```

    *Note: On Windows, use `set FLASK_APP=app.py`.*

3. **Run the Flask development server:**

    ```bash
    flask run
    ```

4. **Access the dashboard:**
    Open your web browser and go to the address provided by the `flask run` command (usually `http://127.0.0.1:5000/`).

Select a repository from the dropdown and click "Assess Risk" to view the predicted risk level, uncertainty, and repository information.

## Acknowledgments

This project was developed under the guidance of Dr. Vũ Thị Hương Giang at Hanoi University of Science and Technology.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
