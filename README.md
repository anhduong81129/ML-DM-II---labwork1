# ML-DM-II---labwork1
📊 Exploratory Data Analysis & PCA Visualization
📌 Project Overview
This project performs data preprocessing, statistical analysis, correlation analysis, and Principal Component Analysis (PCA) on two real-world datasets:
    - Exam Score Prediction Dataset
    - Wine Quality Dataset


The goal is to:
    + Clean and preprocess data
    + Analyze statistical properties (mean & variance)
    + Visualize feature correlations
    + Reduce dimensionality using PCA
    + Visualize data structure in 2D using principal components



📁 Project Structure
📂 Project Folder
│
├── plot.py                     # Main analysis & visualization script
├── Exam_Score_Prediction.csv   # Student exam score dataset
├── Wine_Quality.csv            # Wine quality dataset
└── README.md                   # Project documentation


📊 Datasets Description
1️⃣ Exam Score Prediction Dataset
This dataset contains student-related features used to analyze factors influencing exam performance.
    - Preprocessing steps applied:
    - Removed non-informative column (student_id)
    - Encoded categorical variables
    - Statistical analysis (mean & variance)
    - Correlation matrix visualization
    - PCA-based dimensionality reduction
    - Target variable:
    - exam_score



2️⃣ Wine Quality Dataset
This dataset contains physicochemical properties of wine samples.
Preprocessing steps applied:
    - Removed missing values
    - Encoded categorical variables (if any)
    - Statistical analysis (mean & variance)
    - Correlation matrix visualization
    - PCA-based dimensionality reduction
    - Target variable:
    - quality



⚙️ Technologies & Libraries Used
    - Python 3
    - Pandas – data manipulation
    - NumPy – numerical computation
    - Matplotlib – data visualization
    - Seaborn – correlation heatmaps
    - Scikit-learn
    - StandardScaler
    - LabelEncoder
    - PCA





🔍 Analysis Workflow
The analysis is automated using a reusable function:
🔹 Data Processing
    Encode categorical features using LabelEncoder
    Standardize numerical features using StandardScaler

🔹 Statistical Analysis
    Compute mean and variance for all features

🔹 Correlation Analysis
    Generate correlation matrices
    Visualize using heatmaps

🔹 Principal Component Analysis (PCA)
    Compute explained variance
    Generate scree plots
    Reduce data to 2 principal components
    Visualize data distribution using color-coded scatter plots



▶️ How to Run the Project
1️⃣ Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

2️⃣ Run the Script
python plot.py

The script will:
Print statistical results to the console
Display correlation heatmaps
Show PCA explained variance plots
Visualize 2D PCA projections for both datasets



📈 Output Visualizations
Correlation heatmaps for each dataset
Scree plots showing cumulative explained variance
2D PCA scatter plots colored by target labels



🎯 Learning Outcomes
Understand feature relationships through correlation analysis
Apply PCA for dimensionality reduction
Visualize high-dimensional data effectively
Practice reusable data analysis pipelines



👤 Author
Ánh Dương

If you want, I can also:
Add screenshots section
Write a GitHub-ready version
Simplify it for course submission
Add theoretical explanation of PCA