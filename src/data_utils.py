import pandas as pd
import os

def load_esg_data(file_path):
    """
    Loads data from CSV or Excel files based on the file extension.
    """
    try:
        # Check the file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print(f"ERROR: Unsupported file format for: {file_path}")
            return None
            
        print(f"Successfully loaded data from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None

# ... (your load_esg_data, check_duplicates, visualize_outliers, 
#      and perform_eda_visualizations functions are above this) ...

# REPLACE your old save_to_desktop function with this:
def save_data(df, output_path):
    """
    Saves a DataFrame to a specified full path.
    Saves as either CSV or Excel based on the file extension.
    """
    if df is None:
        print("No data to save.")
        return

    try:
        # Save based on the extension in the output_path
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        else:
            print(f"ERROR: Unsupported save format for: {output_path}. Please use .csv or .xlsx")
            return

        print(f"Successfully saved file to: {output_path}")
    
    except Exception as e:
        print(f"An error occurred during saving: {e}")





import pandas as pd
import os
import matplotlib.pyplot as plt  # <-- ADD THIS IMPORT
import seaborn as sns            # <-- ADD THIS IMPORT

# ... (your existing load_esg_data and save_to_desktop functions) ...


def check_duplicates(df):
    """
    Checks for and reports the number of duplicate rows.
    """
    if df is not None:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate rows in the data.")
        else:
            print("Data Check: No duplicate rows found.")
    else:
        print("No data to check for duplicates.")

def visualize_outliers(df, columns):
    """
    Generates box plots for a list of numeric columns to identify outliers.
    """
    if df is not None:
        print("\nGenerating Box Plots for Outlier Analysis...")
        
        # Set the style for the plots
        sns.set(style="whitegrid")
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(10, 2)) # Make the plot wide and short
                sns.boxplot(x=df[col])
                plt.title(f"Box Plot for {col}")
                plt.show()
            else:
                print(f"Skipping '{col}': Not a numeric column or not found.")
                
        print("Box plot generation complete.")
    else:
        print("No data to visualize.")



        # ... (your existing load, save, check_duplicates, and visualize_outliers functions) ...


def perform_eda_visualizations(df, numeric_cols, categorical_cols):
    """
    Generates a suite of EDA visualizations:
    1. A correlation heatmap for numeric columns.
    2. Histograms for numeric columns.
    3. Count plots for categorical columns.
    """
    if df is None:
        print("No data to visualize.")
        return

    sns.set(style="whitegrid")
    print("\nGenerating EDA Visualizations...")

    # --- 1. Correlation Heatmap ---
    print("\n--- Correlation Heatmap ---")
    print("This shows how strongly numeric features are related to each other (1.0 = perfect positive, -1.0 = perfect negative).")
    
    plt.figure(figsize=(14, 10))
    # Create a correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create the heatmap
    sns.heatmap(corr_matrix, 
                annot=True,     # Show the numbers in the squares
                fmt=".2f",      # Format numbers to 2 decimal places
                cmap="coolwarm", # Use a blue-to-red color map
                linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

    # --- 2. Histograms (with KDE) ---
    print("\n--- Histograms (Distributions) ---")
    print("These show the 'shape' of your numeric data.")
    
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, bins=30) # kde=True adds the smooth line
        plt.title(f"Distribution of {col}")
        plt.show()

    # --- 3. Count Plots (Bar Charts) ---
    print("\n--- Count Plots (Categorical Data) ---")
    print("These show the counts for each category.")
    
    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        # Use order to sort the bars by count
        sns.countplot(y=df[col], 
                      order=df[col].value_counts().index) 
        plt.title(f"Count of Companies by {col}")
        plt.xlabel("Number of Companies")
        plt.ylabel(col)
        plt.show()
        
    print("EDA visualization complete.")