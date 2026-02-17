import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path

class CrossPriceElasticityAnalyzer:
    def __init__(self, cache_dir='./data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.financial_data = None
        self.own_price_elasticities = {}
    
    def load_data(self):
        """Load financial data from the attached Excel files with similar logic to SentimentPriceElasticityIntegration"""
        financial_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name=['S&F', 'Tiverton'])
        self.financial_data = pd.concat([financial_sheets['S&F'], financial_sheets['Tiverton']], ignore_index=True)
        
        # Preprocess dates
        if 'Sales Detail Participation Date' in self.financial_data.columns:
            self.financial_data['Sales Detail Participation Date'] = pd.to_datetime(self.financial_data['Sales Detail Participation Date'], errors='coerce')
        
        # Process product categories into simple canonical product names
        self.financial_data['product_canonical'] = self.financial_data['Product Hierarchy Second Language Description'].str.lower().str.strip()
        
        # Only keep relevant products for services substitution analysis
        valid_services = ['gym', 'swim', 'classes', 'courts']
        # Map product_canonical to main service categories
        def map_to_service(prod):
            if pd.isna(prod):
                return None
            prod = prod.lower()
            if 'gym' in prod:
                return 'gym'
            elif 'swim' in prod:
                return 'swim'
            elif 'class' in prod:
                return 'classes'
            elif 'court' in prod or 'squash' in prod or 'badminton' in prod:
                return 'courts'
            else:
                return None
        self.financial_data['product_canonical'] = self.financial_data['product_canonical'].apply(map_to_service)
        
        # Filter data for relevant services
        self.financial_data = self.financial_data[self.financial_data['product_canonical'].isin(valid_services)]
        
        # Create analysis period field (month-year) for aggregation
        self.financial_data['period'] = self.financial_data['Sales Detail Participation Date'].dt.to_period('M').astype(str)

    def prepare_data_for_elasticity(self):
        """Aggregate data by product and period to get average price and demand quantity"""
        # Calculate avg price and total quantity per service per period
        df = self.financial_data.copy()
        # Net amount is used as price proxy here
        grouped = df.groupby(['period', 'product_canonical']).agg(
            avg_price=('Sales Detail Gross Amount', 'mean'),  # mean price
            quantity=('Sales Detail Quantity', 'sum')  # total quantity booked
        ).reset_index()
        
        # Log transform to stabilize variance
        grouped['log_price'] = np.log(grouped['avg_price'].replace(0, np.nan))
        grouped['log_quantity'] = np.log(grouped['quantity'].replace(0, np.nan))
        grouped = grouped.dropna(subset=['log_price', 'log_quantity'])
        return grouped

    def calculate_own_price_elasticity(self, data):
        """Calculate own-price elasticity for each service"""
        elasticity_results = {}
        for service in data['product_canonical'].unique():
            service_data = data[data['product_canonical'] == service].sort_values('period')
            if len(service_data) > 20:
                # Calculate pct change using log differences as approximation
                service_data['price_change'] = service_data['log_price'].diff()
                service_data['quantity_change'] = service_data['log_quantity'].diff()
                valid = service_data.dropna(subset=['price_change', 'quantity_change'])
                if not valid.empty:
                    model = LinearRegression().fit(valid['price_change'].values.reshape(-1, 1), valid['quantity_change'].values)
                    elasticity = model.coef_[0]
                    elasticity_results[service] = elasticity
        self.own_price_elasticities = elasticity_results
        return elasticity_results

    def calculate_cross_price_elasticity(self, data):
        """
        Calculate how price changes in one service affect demand for others
        """
        from itertools import combinations
        cross_elasticities = {}
        services = data['product_canonical'].unique()
        services = [s for s in ['gym','swim','classes','courts'] if s in services]

        for service_i, service_j in combinations(services, 2):
            data_i = data[data['product_canonical'] == service_i]
            data_j = data[data['product_canonical'] == service_j]
            merged_data = data_i.merge(data_j, on='period', suffixes=('_i', '_j'))
            if len(merged_data) > 20:
                X = merged_data[['log_price_i']].values.reshape(-1, 1)
                y = merged_data['log_quantity_j'].values
                model = LinearRegression().fit(X, y)
                cross_elasticity = model.coef_
                cross_elasticities[f"{service_i}_to_{service_j}"] = cross_elasticity
        return cross_elasticities

    def create_substitution_matrix(self, cross_elasticities):
        """
        Visualize service substitution relationships
        """
        services = ['gym', 'swim', 'classes', 'courts']
        matrix = pd.DataFrame(index=services, columns=services, dtype=float)
        # Fill diagonal with own-price elasticities
        for service in services:
            matrix.loc[service, service] = self.own_price_elasticities.get(service, np.nan)
        # Fill off-diagonal with cross-price elasticities
        for key, value in cross_elasticities.items():
            service_i, service_j = key.split('_to_')
            matrix.loc[service_i, service_j] = value
            matrix.loc[service_j, service_i] = value  # Assume symmetry
        # Plot heatmap
        for service in services:
            plt.figure(figsize=(6, 5))
            sns.heatmap(matrix.astype(float), annot=True, cmap='RdBu_r', center=0)
            plt.title(f'Substitution Matrix - Highlighting {service}')
            plt.show()
        return matrix

# Instantiate and run the analysis
analyzer = CrossPriceElasticityAnalyzer()
analyzer.load_data()
data_for_analysis = analyzer.prepare_data_for_elasticity()
own_price_elasticities = analyzer.calculate_own_price_elasticity(data_for_analysis)
cross_price_elasticities = analyzer.calculate_cross_price_elasticity(data_for_analysis)
substitution_matrix = analyzer.create_substitution_matrix(cross_price_elasticities)

