import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

#===============================================================
# 1. DATA LOADING

exam_df = pd.read_csv('Exam_Score_Prediction.csv')
wine_df = pd.read_csv('Wine_Quality.csv')

#===============================================================
# 2. DATA CLEANING & PREPARATION

# Remove non-informative columns from Exam dataset
exam_df = exam_df.drop(columns=['student_id'])

# Handle missing values in Wine dataset
wine_df = wine_df.dropna()

def preprocess_and_analyze(df, label_col, title):
    print(f"\n=== Analysis for {title} ===")
    
    # ENCODE Categorical Features
    df_encoded = df.copy()
    le = LabelEncoder()       #Using LabelEncoder for simplicity                 
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Calculate MEAN and VARIANCE
    print("Mean:\n", df_encoded.mean())
    print("\nVariance:\n", df_encoded.var())
    
    # Correlation Matrix
    corr = df_encoded.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Correlation Matrix - {title}')
    plt.show()
    
    # Calculate PCA
    X = df_encoded.drop(columns=[label_col])
    y = df_encoded[label_col]
    
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # Plot Scree Plot (Cumulative Explained Variance)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title(f'Explained Variance by PCs - {title}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.grid()
    plt.show()
    
    # 2D PCA Visualization
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title(f'PCA 2D Projection - {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label=label_col)
    plt.show()

# Run for both datasets
preprocess_and_analyze(exam_df, 'exam_score', "Exam Score Prediction")
preprocess_and_analyze(wine_df, 'quality', "Wine Quality")