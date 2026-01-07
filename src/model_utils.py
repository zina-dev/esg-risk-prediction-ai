# model_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# REPLACE your old function with this new one
def prepare_data_for_modeling(df, target_column, cols_to_drop):
    """
    Prepares the data by:
    1. Splitting into features (X) and target (y).
    2. Splitting into training and testing sets.
    3. Scaling the features (X) using StandardScaler.
    """
    print(f"\nPreparing data for modeling...")
    
    # 1. Define X and y
    y = df[target_column]
    cols_to_drop_full = cols_to_drop + [target_column]
    X = df.drop(columns=cols_to_drop_full)
    
    print(f"  - Target (y): {target_column}")
    print(f"  - Features (X): {list(X.columns)}")
    
    # 2. Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. --- NEW: SCALE THE DATA ---
    # This is required for the Deep Learning model to work well
    print("  - Scaling data using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training data ONLY
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data (using the fit from training)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  - Training set size: {X_train_scaled.shape[0]} samples")
    print(f"  - Testing set size:  {X_test_scaled.shape[0]} samples")
    
    # Return the SCALED data
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model on the training data.
    """
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  - Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints key metrics.
    Also generates a plot of actual vs. predicted values.
    """
    print("\nEvaluating model performance...")
    
    # 1. Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # 2. Calculate metrics
    # R-squared: What % of the variance in y is explained by X? (Closer to 1.0 is better)
    r2 = r2_score(y_test, y_pred)
    
    # RMSE: On average, how many "points" off is our prediction? (Lower is better)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  - R-squared (R²):   {r2:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # 3. Plot Actual vs. Predicted
    print("  - Generating Actual vs. Predicted plot...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    
    # Add a "perfect prediction" line
    perfect_line = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(perfect_line, perfect_line, 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual ESG_Overall Score")
    plt.ylabel("Predicted ESG_Overall Score")
    plt.title("Model Evaluation: Actual vs. Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()




    # model_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # <-- 1. ADD THIS IMPORT
from sklearn.metrics import r2_score, mean_squared_error

# ... (your existing prepare_data_for_modeling, train_linear_regression,
#      and evaluate_model functions are here) ...


# 2. ADD THIS NEW FUNCTION AT THE END
def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Regressor model on the training data.
    """
    print("\nTraining Random Forest Regressor model...")
    
    # n_estimators = number of "trees" in the forest
    # n_jobs = -1 uses all your computer's CPU cores to speed up training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    print("  - Model training complete.")
    return model



# ... (near other sklearn imports)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ... (after your train_random_forest function) ...

def train_deep_learning_model(X_train, y_train):
    """
    Trains a tuned Deep Learning (MLP Regressor) model
    based on the user's expert suggestions.
    """
    print("\nTraining Tuned Deep Learning (MLP) model...")
    
    # These are the new, better parameters you suggested
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        learning_rate_init=0.0008,
        alpha=1e-4, # L2 regularization
        batch_size=256,
        max_iter=1200,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  - Model training complete.")
    return model

def train_xgboost_model(X_train, y_train):
    """
    Trains an XGBoost Regressor model.
    """
    print("\nTraining XGBoost Regressor model...")

    model = XGBRegressor(
        n_estimators=100,   # number of trees
        learning_rate=0.1,  # learning rate
        max_depth=6,        # add depth for control
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("  - Model training complete.")
    return model  # ✅ must return the trained model





# ... (all your other model functions are above this) ...

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for a trained tree-based model
    (like Random Forest or XGBoost).
    """
    print("\nGenerating Feature Importance plot...")
    
    # Check if the model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("This model does not have 'feature_importances_'.")
        print("Please use a model like Random Forest or XGBoost.")
        return

    # Create a DataFrame of features and their importance scores
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Display the top 10 features
    print("Top 10 Most Important Features:")
    print(feature_df.head(10))
    
    # Create the bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title(f"Feature Importance for {type(model).__name__}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()