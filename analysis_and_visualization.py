import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os


def load_and_clean_data(filepath):
    """
    Loads the Global Superstore dataset, cleans column names, converts data types,
    handles missing values, and engineers new features for analysis.
    """
    print("Step 1: Loading and Cleaning Data...")
    # Load the dataset using 'latin1' encoding for better compatibility
    df = pd.read_csv(filepath, encoding='latin1')

    # Standardize column names: make lowercase and replace special characters
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Convert date columns to datetime objects, assuming day-first format (e.g., 01-01-2021)
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    df['ship_date'] = pd.to_datetime(df['ship_date'], dayfirst=True)

    # Handle missing Postal Code values: fill with 0 and convert to integer
    df['postal_code'] = df['postal_code'].fillna(0).astype(int)

    # Create a Profit Margin column for easier analysis
    # Use np.where to avoid division-by-zero errors
    df['profit_margin'] = np.where(df['sales'] != 0, df['profit'] / df['sales'], 0)

    # Extract year and month for temporal analysis
    df['order_year_month'] = df['order_date'].dt.to_period('M')

    print("Data cleaning complete.")
    return df


def analyze_geographical_performance(df):
    """Analyzes sales and profit by country and saves visualizations."""
    print("Step 2: Analyzing Geographical Performance...")
    country_performance = df.groupby('country').agg(
        total_sales=('sales', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()

    # Correctly calculate profit margin for the aggregated data
    country_performance['profit_margin'] = np.where(
        country_performance['total_sales'] != 0,
        country_performance['total_profit'] / country_performance['total_sales'],
        0
    )

    top_10_sales = country_performance.sort_values(by='total_sales', ascending=False).head(10)
    top_10_profit = country_performance.sort_values(by='total_profit', ascending=False).head(10)

    # Visualization: Top 10 Countries by Sales
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_10_sales, x='total_sales', y='country', palette='viridis')
    plt.title('Top 10 Countries by Total Sales', fontsize=16)
    plt.xlabel('Total Sales ($)')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('charts/geography_top_10_sales.png')
    plt.close()

    # Visualization: Top 10 Countries by Profit
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_10_profit, x='total_profit', y='country', palette='plasma')
    plt.title('Top 10 Countries by Total Profit', fontsize=16)
    plt.xlabel('Total Profit ($)')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('charts/geography_top_10_profit.png')
    plt.close()
    print("Geographical charts saved.")


def analyze_temporal_trends(df):
    """Analyzes monthly sales and profit and saves the visualization."""
    print("Step 3: Analyzing Temporal Trends...")
    monthly_trends = df.groupby('order_year_month').agg(
        total_sales=('sales', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()
    monthly_trends['order_year_month'] = monthly_trends['order_year_month'].dt.to_timestamp()

    # Visualization: Monthly Sales and Profit
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(monthly_trends['order_year_month'], monthly_trends['total_sales'], color='dodgerblue', marker='o',
             linestyle='-', label='Sales')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Sales ($)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')

    ax2 = ax1.twinx()
    ax2.plot(monthly_trends['order_year_month'], monthly_trends['total_profit'], color='seagreen', marker='x',
             linestyle='--', label='Profit')
    ax2.set_ylabel('Total Profit ($)', color='seagreen')
    ax2.tick_params(axis='y', labelcolor='seagreen')

    plt.title('Monthly Sales and Profit Trends (2011-2014)', fontsize=16)
    fig.tight_layout()
    plt.savefig('charts/temporal_monthly_trends.png')
    plt.close()
    print("Temporal trends chart saved.")


def analyze_product_performance(df):
    """Analyzes profit by sub-category and saves the visualization."""
    print("Step 4: Analyzing Product Performance...")
    subcategory_profit = df.groupby(['category', 'sub_category']).agg(
        total_profit=('profit', 'sum')
    ).reset_index().sort_values(by='total_profit', ascending=False)

    # Visualization: Profit by Sub-Category
    plt.figure(figsize=(14, 10))
    sns.barplot(data=subcategory_profit, x='total_profit', y='sub_category', hue='category', dodge=False,
                palette='coolwarm')
    plt.title('Total Profit by Product Sub-Category', fontsize=16)
    plt.xlabel('Total Profit ($)')
    plt.ylabel('Sub-Category')
    plt.axvline(x=0, color='black', linestyle='--')  # Add a line at zero for reference
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig('charts/product_profit_by_subcategory.png')
    plt.close()
    print("Product performance chart saved.")


def analyze_discount_impact(df):
    """Analyzes the relationship between discount and profit margin."""
    print("Step 5: Analyzing Discount Impact...")
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='discount', y='profit_margin',
                scatter_kws={'alpha': 0.1, 'color': 'gray'}, line_kws={'color': 'red'})
    plt.title('Profit Margin vs. Discount Level', fontsize=16)
    plt.xlabel('Discount')
    plt.ylabel('Profit Margin')
    # Format Y-axis as percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('charts/impact_discount_vs_margin.png')
    plt.close()
    print("Discount impact chart saved.")


def main():
    """Main function to run the entire analysis pipeline."""
    # Create a directory for charts if it doesn't exist
    if not os.path.exists('charts'):
        os.makedirs('charts')

    filepath = 'Global_Superstore2.csv'
    try:
        df_cleaned = load_and_clean_data(filepath)
        analyze_geographical_performance(df_cleaned)
        analyze_temporal_trends(df_cleaned)
        analyze_product_performance(df_cleaned)
        analyze_discount_impact(df_cleaned)
        print("\nAnalysis complete. All charts have been saved to the 'charts' directory.")
    except FileNotFoundError:
        print(
            f"FATAL ERROR: The file '{filepath}' was not found. Please ensure it is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

