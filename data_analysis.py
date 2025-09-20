"""
Data Analysis with Pandas and Visualization with Matplotlib
Assignment: Analyzing the Iris Dataset

This script demonstrates:
1. Loading and exploring datasets with pandas
2. Basic data analysis and statistics
3. Data visualization with matplotlib
4. Error handling and data cleaning
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_dataset():
    """
    Task 1: Load and Explore the Dataset
    Load the Iris dataset and perform initial exploration
    """
    try:
        print("=" * 60)
        print("TASK 1: LOADING AND EXPLORING THE DATASET")
        print("=" * 60)
        
        # Load the Iris dataset
        iris_data = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['species'] = iris_data.target
        df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("✓ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\n1. First 5 rows of the dataset:")
        print(df.head())
        
        print("\n2. Dataset information:")
        print(df.info())
        
        print("\n3. Data types:")
        print(df.dtypes)
        
        print("\n4. Checking for missing values:")
        missing_values = df.isnull().sum()
        print(missing_values)
        
        if missing_values.sum() == 0:
            print("✓ No missing values found!")
        else:
            print("⚠ Missing values detected - cleaning required")
            # Fill missing values with median for numerical columns
            df.fillna(df.median(), inplace=True)
            print("✓ Missing values filled with median")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    """
    Task 2: Basic Data Analysis
    Compute statistics and perform groupings
    """
    try:
        print("\n" + "=" * 60)
        print("TASK 2: BASIC DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("1. Basic statistics for numerical columns:")
        print(df.describe())
        
        # Group by species and compute means
        print("\n2. Average measurements by species:")
        species_means = df.groupby('species_name').mean()
        print(species_means)
        
        # Additional analysis
        print("\n3. Species distribution:")
        species_counts = df['species_name'].value_counts()
        print(species_counts)
        
        # Correlation analysis
        print("\n4. Correlation matrix:")
        numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix)
        
        # Key findings
        print("\n5. Key Findings:")
        print("• All species have equal representation (50 samples each)")
        print("• Petal length and petal width are highly correlated (0.96)")
        print("• Virginica has the largest average measurements")
        print("• Setosa has the smallest petal measurements")
        
        return species_means, correlation_matrix
        
    except Exception as e:
        print(f"❌ Error in data analysis: {e}")
        return None, None

def create_visualizations(df, species_means, correlation_matrix):
    """
    Task 3: Data Visualization
    Create four different types of visualizations
    """
    try:
        print("\n" + "=" * 60)
        print("TASK 3: DATA VISUALIZATION")
        print("=" * 60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iris Dataset Analysis - Four Visualization Types', fontsize=16, fontweight='bold')
        
        # 1. Line Chart - Trends across species (simulated time series)
        ax1 = axes[0, 0]
        species_order = ['setosa', 'versicolor', 'virginica']
        measurements = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        for measurement in measurements:
            values = [species_means.loc[species, measurement] for species in species_order]
            ax1.plot(species_order, values, marker='o', linewidth=2, label=measurement)
        
        ax1.set_title('1. Line Chart: Average Measurements Across Species', fontweight='bold')
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Measurement (cm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar Chart - Comparison across categories
        ax2 = axes[0, 1]
        x_pos = np.arange(len(species_order))
        petal_lengths = [species_means.loc[species, 'petal length (cm)'] for species in species_order]
        
        bars = ax2.bar(x_pos, petal_lengths, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax2.set_title('2. Bar Chart: Average Petal Length by Species', fontweight='bold')
        ax2.set_xlabel('Species')
        ax2.set_ylabel('Petal Length (cm)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(species_order)
        
        # Add value labels on bars
        for bar, value in zip(bars, petal_lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Histogram - Distribution of a numerical column
        ax3 = axes[1, 0]
        ax3.hist(df['sepal length (cm)'], bins=20, color='#96CEB4', alpha=0.7, edgecolor='black')
        ax3.set_title('3. Histogram: Distribution of Sepal Length', fontweight='bold')
        ax3.set_xlabel('Sepal Length (cm)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics to histogram
        mean_val = df['sepal length (cm)'].mean()
        ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax3.legend()
        
        # 4. Scatter Plot - Relationship between two numerical columns
        ax4 = axes[1, 1]
        colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        
        for species in df['species_name'].unique():
            species_data = df[df['species_name'] == species]
            ax4.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                       c=colors[species], label=species, alpha=0.7, s=50)
        
        ax4.set_title('4. Scatter Plot: Sepal Length vs Petal Length', fontweight='bold')
        ax4.set_xlabel('Sepal Length (cm)')
        ax4.set_ylabel('Petal Length (cm)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: Correlation heatmap
        plt.figure(figsize=(10, 8))
        numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Heatmap of Iris Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("✓ All visualizations created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def main():
    """
    Main function to execute the complete data analysis workflow
    """
    print("IRIS DATASET ANALYSIS WITH PANDAS AND MATPLOTLIB ")
    print("This script demonstrates comprehensive data analysis techniques\n")
    
    # Task 1: Load and explore dataset
    df = load_and_explore_dataset()
    if df is None:
        return
    
    # Task 2: Basic data analysis
    species_means, correlation_matrix = basic_data_analysis(df)
    if species_means is None:
        return
    
    # Task 3: Create visualizations
    create_visualizations(df, species_means, correlation_matrix)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Summary of findings:")
    print("• Dataset contains 150 samples across 3 iris species")
    print("• No missing values detected")
    print("• Strong correlation between petal measurements")
    print("• Clear species separation based on petal characteristics")
    print("• Virginica species has largest overall measurements")

if __name__ == "__main__":
    main()
