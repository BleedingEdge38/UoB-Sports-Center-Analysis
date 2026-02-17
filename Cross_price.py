import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from itertools import combinations

def run_analysis():
    """
    Main function to run the complete price elasticity analysis.
    """
    # Step 1: Load Data
    # Correctly load all required sheets from the attached Excel files
    financial_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name=['S&F', 'Tiverton'])
    financial_data = pd.concat(financial_sheets.values(), ignore_index=True)

    bookings_sheets = pd.read_excel('Bookings_data.xlsx', sheet_name=['Squash', 'Classes'])
    bookings_data = pd.concat(bookings_sheets.values(), ignore_index=True)

    # Step 2: Preprocess Data
    # Convert date and numeric columns, handling potential errors
    financial_data['Sales Detail Participation Date'] = pd.to_datetime(financial_data['Sales Detail Participation Date'], errors='coerce')
    financial_data['Sales Detail Gross Amount'] = pd.to_numeric(financial_data['Sales Detail Gross Amount'], errors='coerce')
    financial_data['Sales Detail Quantity'] = pd.to_numeric(financial_data['Sales Detail Quantity'], errors='coerce')
    
    # Define a robust function for service categorization
    def map_service(product_name):
        if pd.isna(product_name):
            return 'other'
        product_name = str(product_name).lower()
        if 'gym' in product_name or 'inclusive' in product_name:
            return 'gym'
        if 'swim' in product_name:
            return 'swim'
        if 'class' in product_name:
            return 'classes'
        if 'squash' in product_name or 'court' in product_name:
            return 'courts'
        return 'other'

    financial_data['product_canonical'] = financial_data['Product Hierarchy Product'].apply(map_service)
    
    # Create a 'period' column for monthly aggregation
    financial_data['period'] = financial_data['Sales Detail Participation Date'].dt.to_period('M')

    # Step 3: Aggregate Data for Elasticity Analysis
    # Aggregate data to get monthly price and quantity for each service
    agg_data = financial_data[financial_data['product_canonical'] != 'other'].groupby(['period', 'product_canonical']).agg(
        total_revenue=('Sales Detail Gross Amount', 'sum'),
        total_quantity=('Sales Detail Quantity', 'sum')
    ).reset_index()

    # Calculate average price and handle divisions by zero
    agg_data = agg_data[agg_data['total_quantity'] > 0]
    agg_data['avg_price'] = agg_data['total_revenue'] / agg_data['total_quantity']
    
    # Filter out negative or zero prices which are not valid for log transformation
    agg_data = agg_data[agg_data['avg_price'] > 0]

    # Create log-transformed columns for the regression model
    agg_data['log_price'] = np.log(agg_data['avg_price'])
    agg_data['log_quantity'] = np.log(agg_data['total_quantity'])
    
    services = ['gym', 'swim', 'classes', 'courts']
    
    # Step 4: Calculate Own-Price Elasticity
    own_price_elasticities = {}
    for service in services:
        service_data = agg_data[agg_data['product_canonical'] == service]
        if len(service_data) > 20: # Ensure sufficient data for regression
            X = service_data[['log_price']]
            y = service_data['log_quantity']
            model = LinearRegression().fit(X, y)
            own_price_elasticities[service] = model.coef_[0]
        else:
            own_price_elasticities[service] = np.nan

    # Step 5: Calculate Cross-Price Elasticity
    cross_elasticities = {}
    for service_i, service_j in combinations(services, 2):
        data_i = agg_data[agg_data['product_canonical'] == service_i][['period', 'log_price']]
        data_j = agg_data[agg_data['product_canonical'] == service_j][['period', 'log_quantity']]
        merged_data = pd.merge(data_i, data_j, on='period', suffixes=('_i', '_j'))
        
        if len(merged_data) > 20:
            X = merged_data[['log_price_i']]
            y = merged_data['log_quantity_j']
            model = LinearRegression().fit(X, y)
            cross_elasticities[f"{service_i}_to_{service_j}"] = model.coef_
        else:
            cross_elasticities[f"{service_i}_to_{service_j}"] = np.nan

    # Step 6: Build and Visualize the Substitution Matrix
    matrix = pd.DataFrame(index=services, columns=services, dtype=float)
    
    # Fill diagonal with own-price elasticities
    for service, elasticity in own_price_elasticities.items():
        matrix.loc[service, service] = elasticity
        
    # Fill off-diagonal with cross-price elasticities, assuming symmetry
    for key, elasticity in cross_elasticities.items():
        s_i, s_j = key.split('_to_')
        matrix.loc[s_i, s_j] = elasticity
        matrix.loc[s_j, s_i] = elasticity # Assume symmetry

    # Step 7: Generate Charts
    # Chart 1: Own-Price Elasticity
    plt.figure(figsize=(10, 6))
    own_elasticity_df = pd.Series(own_price_elasticities).dropna()
    bars = sns.barplot(x=own_elasticity_df.index, y=own_elasticity_df.values, palette='viridis')
    plt.title('Own-Price Elasticity of Demand by Service', fontsize=16)
    plt.ylabel('Elasticity Coefficient')
    plt.xlabel('Service')
    plt.axhline(0, color='grey', linestyle='--')
    plt.axhline(-1, color='red', linestyle='--', label='Unitary Elasticity')
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                 ha='center', va='bottom' if bar.get_height() > 0 else 'top', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Chart 2: Cross-Price Elasticity Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix.astype(float), annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=.5)
    plt.title('Cross-Price Elasticity and Substitution Matrix', fontsize=16)
    plt.xlabel('Service Whose Price Changes')
    plt.ylabel('Service Whose Demand is Affected')
    plt.tight_layout()
    plt.show()

    return matrix

# Execute the analysis
final_matrix = run_analysis()
print("Cross-Price Elasticity Analysis Complete.")
print("Substitution Matrix:")
print(final_matrix)

