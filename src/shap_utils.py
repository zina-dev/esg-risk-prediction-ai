# shap_utils.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_model_with_shap(model, X_train_scaled, feature_names, sample_size=1000):
    """
    Generates a SHAP summary plot to explain the model's predictions.
    Optimized to run faster by sampling a subset of data.

    Args:
        model: The trained model (e.g., Random Forest or XGBoost).
        X_train_scaled: The scaled training data (NumPy array).
        feature_names: List of feature names.
        sample_size: Number of rows to use for SHAP (default = 1000).
    """

    print("\n--- Starting Optimized SHAP Analysis ---")
    print(f"Sampling up to {sample_size} rows for SHAP computation...")

    # 1. Convert to DataFrame with feature names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)

    # 2. Sample a smaller subset for faster SHAP computation
    if len(X_train_df) > sample_size:
        X_sample = X_train_df.sample(sample_size, random_state=42)
    else:
        X_sample = X_train_df.copy()

    # 3. Initialize TreeExplainer (optimized for tree-based models)
    explainer = shap.TreeExplainer(model)

    # 4. Calculate SHAP values on the sample
    print("Calculating SHAP values (this will take ~10–30 seconds)...")
    shap_values = explainer.shap_values(X_sample)

    # 5. Plot summary
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)

    plt.title("SHAP Summary Plot: Understanding Feature Impact (Sampled)")
    plt.tight_layout()
    plt.show()



# ... (your analyze_model_with_shap function is above this) ...

def analyze_deep_model_with_shap(model, X_train_scaled, X_test_scaled, feature_names):
    """
    Analyzes a deep learning (MLP) model with SHAP's KernelExplainer.
    
    WARNING: This is a model-agnostic method and is EXTREMELY SLOW.
    It works by creating a summary of the data and running
    thousands of predictions.
    """
    print("\n--- Starting SHAP Analysis for Deep Learning (MLP) Model ---")
    print("WARNING: This will be VERY SLOW (5-15+ minutes)...")
    
    # 1. Create a summary of the training data (background data)
    # KernelExplainer needs a summary of the data to build
    # its baseline. We'll use 100 summary points.
    print("  - Creating K-Means summary of 100 data points...")
    X_train_summary = shap.kmeans(X_train_scaled, 100)
    
    # 2. Use KernelExplainer
    # We pass the model's prediction function (model.predict)
    # and the background data summary.
    print("  - Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(model.predict, X_train_summary)
    
    # 3. Get a small sample of the *test data* to explain.
    # Running on all 2200+ test samples would take hours.
    # We'll take a random 100-sample from the test set.
    print("  - Taking 100 random samples from the test set to explain...")
    sample_indices = np.random.choice(X_test_scaled.shape[0], 100, replace=False)
    X_test_sample = X_test_scaled[sample_indices]

    # 4. Calculate SHAP values for the test sample.
    # This is the step that will take several minutes.
    print("  - Calculating SHAP values... Please be patient...")
    shap_values = explainer.shap_values(X_test_sample)
    
    print("SHAP values calculated. Generating plot...")

    # 5. Convert the test sample to a DataFrame for plotting
    X_test_sample_df = pd.DataFrame(X_test_sample, columns=feature_names)

    # 6. Generate the summary plot
    shap.summary_plot(shap_values, X_test_sample_df, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot for MLP Model (Sampled)")
    plt.show()