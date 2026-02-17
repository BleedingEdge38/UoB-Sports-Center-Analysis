# enhanced_revenue_pricing_analytics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RevenuePricingAnalytics:
    def __init__(self, data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        """Initialize the revenue and pricing analytics class"""
        self.data_dir = data_dir
        self.load_prepared_data()
        self.prepare_revenue_data()

    def load_prepared_data(self):
        """Load prepared datasets from files"""
        print("📂 Loading prepared data for revenue analysis...")
        try:
            # Load main datasets
            with open(f'{self.data_dir}/attendance.pkl', 'rb') as f:
                self.attendance = pickle.load(f)
            
            with open(f'{self.data_dir}/financial.pkl', 'rb') as f:
                self.financial = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_squash.pkl', 'rb') as f:
                self.bookings_squash = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_classes.pkl', 'rb') as f:
                self.bookings_classes = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_alt_sessions.pkl', 'rb') as f:
                self.bookings_alt_sessions = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_other_activities.pkl', 'rb') as f:
                self.bookings_other_activities = pickle.load(f)
            
            print("✅ All prepared data loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading prepared data: {e}")
            print("🔄 Please run the data preparation script first!")
            raise

    def prepare_revenue_data(self):
        """Prepare and clean revenue data for analysis"""
        print("🔧 Preparing revenue data...")
        
        # Clean financial data
        self.financial['gross_amount'] = pd.to_numeric(self.financial['sales_detail_gross_amount'], errors='coerce')
        self.financial['participation_date'] = pd.to_datetime(self.financial['sales_detail_participation_date'], errors='coerce')
        self.financial['raised_date'] = pd.to_datetime(self.financial['sales_detail_raised_date'], errors='coerce')
        
        # Extract relevant columns for analysis
        revenue_columns = [
            'unique_key', 'gross_amount', 'participation_date', 'raised_date',
            'sales_detail_price_level', 'sales_detail_payment_method',
            'sales_detail_status', 'sales_detail_product_id',
            'contacts_detail_age', 'contacts_detail_gender'
        ]
        
        # Filter valid financial records
        self.financial_clean = self.financial[revenue_columns].copy()
        self.financial_clean = self.financial_clean[
            (self.financial_clean['gross_amount'] > 0) &
            (self.financial_clean['sales_detail_status'] == 'Paid')
        ].copy()
        
        # Add time-based features
        self.financial_clean['year'] = self.financial_clean['participation_date'].dt.year
        self.financial_clean['month'] = self.financial_clean['participation_date'].dt.month
        self.financial_clean['quarter'] = self.financial_clean['participation_date'].dt.quarter
        self.financial_clean['day_of_week'] = self.financial_clean['participation_date'].dt.dayofweek
        
        # Clean attendance data
        self.attendance['date'] = pd.to_datetime(self.attendance['date'], errors='coerce')
        self.attendance['age'] = pd.to_numeric(self.attendance['age'], errors='coerce')
        
        print("✅ Revenue data prepared successfully!")

    def revenue_concentration_analysis(self):
        """Analyze revenue concentration - identify top 20% customers contributing to 80% of revenue"""
        print("💰 Analyzing revenue concentration...")
        
        # Calculate total revenue per customer
        customer_revenue = self.financial_clean.groupby('unique_key').agg({
            'gross_amount': 'sum',
            'participation_date': ['count', 'min', 'max'],
            'sales_detail_price_level': 'first'
        }).reset_index()
        
        customer_revenue.columns = ['unique_key', 'total_revenue', 'transaction_count', 'first_transaction', 'last_transaction', 'price_level']
        customer_revenue['customer_lifetime_days'] = (customer_revenue['last_transaction'] - customer_revenue['first_transaction']).dt.days
        
        # Sort by revenue and calculate cumulative percentage
        customer_revenue = customer_revenue.sort_values('total_revenue', ascending=False)
        customer_revenue['cumulative_revenue'] = customer_revenue['total_revenue'].cumsum()
        customer_revenue['cumulative_revenue_pct'] = (customer_revenue['cumulative_revenue'] / customer_revenue['total_revenue'].sum()) * 100
        customer_revenue['customer_rank_pct'] = (np.arange(len(customer_revenue)) + 1) / len(customer_revenue) * 100
        
        # Identify top 20% customers
        top_20_pct_customers = customer_revenue[customer_revenue['customer_rank_pct'] <= 20]
        revenue_by_top_20 = top_20_pct_customers['total_revenue'].sum()
        total_revenue = customer_revenue['total_revenue'].sum()
        
        # Find customers contributing to 80% of revenue
        customers_80_revenue = customer_revenue[customer_revenue['cumulative_revenue_pct'] <= 80]
        
        return {
            'customer_revenue': customer_revenue,
            'top_20_pct_customers': top_20_pct_customers,
            'customers_80_revenue': customers_80_revenue,
            'revenue_by_top_20': revenue_by_top_20,
            'total_revenue': total_revenue,
            'pareto_stats': {
                'top_20_pct_revenue_contribution': (revenue_by_top_20 / total_revenue) * 100,
                'customers_for_80_revenue': len(customers_80_revenue),
                'total_customers': len(customer_revenue)
            }
        }

    def arpu_analysis(self):
        """Calculate Average Revenue Per User by facility, membership type, and time period"""
        print("📊 Calculating ARPU analysis...")
        
        # ARPU by membership type
        arpu_by_membership = self.financial_clean.groupby('sales_detail_price_level').agg({
            'gross_amount': ['sum', 'mean', 'count'],
            'unique_key': 'nunique'
        }).reset_index()
        arpu_by_membership.columns = ['membership_type', 'total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
        arpu_by_membership['arpu'] = arpu_by_membership['total_revenue'] / arpu_by_membership['unique_customers']
        
        # ARPU by time period (monthly)
        arpu_by_month = self.financial_clean.groupby(['year', 'month']).agg({
            'gross_amount': 'sum',
            'unique_key': 'nunique'
        }).reset_index()
        arpu_by_month['arpu'] = arpu_by_month['gross_amount'] / arpu_by_month['unique_key']
        arpu_by_month['period'] = pd.to_datetime(arpu_by_month[['year', 'month']].assign(day=1))
        
        # ARPU by facility (based on attendance data correlation)
        facility_revenue = self.attendance.merge(
            self.financial_clean[['unique_key', 'gross_amount', 'participation_date']],
            on='unique_key',
            how='inner'
        )
        
        arpu_by_facility = facility_revenue.groupby('facility_type').agg({
            'gross_amount': ['sum', 'mean'],
            'unique_key': 'nunique'
        }).reset_index()
        arpu_by_facility.columns = ['facility_type', 'total_revenue', 'avg_transaction', 'unique_customers']
        arpu_by_facility['arpu'] = arpu_by_facility['total_revenue'] / arpu_by_facility['unique_customers']
        
        return {
            'arpu_by_membership': arpu_by_membership,
            'arpu_by_month': arpu_by_month,
            'arpu_by_facility': arpu_by_facility
        }

    def revenue_seasonality_analysis(self):
        """Analyze revenue seasonality patterns"""
        print("🗓️ Analyzing revenue seasonality...")
        
        # Monthly revenue patterns
        monthly_revenue = self.financial_clean.groupby(['year', 'month']).agg({
            'gross_amount': 'sum',
            'unique_key': 'nunique',
            'participation_date': 'count'
        }).reset_index()
        monthly_revenue['period'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
        monthly_revenue.columns = ['year', 'month', 'total_revenue', 'unique_customers', 'transactions', 'period']
        
        # Quarterly revenue patterns
        quarterly_revenue = self.financial_clean.groupby(['year', 'quarter']).agg({
            'gross_amount': 'sum',
            'unique_key': 'nunique',
            'participation_date': 'count'
        }).reset_index()
        quarterly_revenue.columns = ['year', 'quarter', 'total_revenue', 'unique_customers', 'transactions']
        
        # Day of week patterns
        dow_revenue = self.financial_clean.groupby('day_of_week').agg({
            'gross_amount': ['sum', 'mean'],
            'unique_key': 'nunique'
        }).reset_index()
        dow_revenue.columns = ['day_of_week', 'total_revenue', 'avg_transaction', 'unique_customers']
        dow_revenue['day_name'] = dow_revenue['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        # Calculate seasonality indices
        monthly_avg = monthly_revenue['total_revenue'].mean()
        monthly_revenue['seasonality_index'] = (monthly_revenue['total_revenue'] / monthly_avg) * 100
        
        return {
            'monthly_revenue': monthly_revenue,
            'quarterly_revenue': quarterly_revenue,
            'dow_revenue': dow_revenue
        }

    def payment_method_analysis(self):
        """Analyze payment method preferences and correlation with customer behavior"""
        print("💳 Analyzing payment method preferences...")
        
        # Payment method distribution
        payment_methods = self.financial_clean.groupby('sales_detail_payment_method').agg({
            'gross_amount': ['sum', 'mean', 'count'],
            'unique_key': 'nunique'
        }).reset_index()
        payment_methods.columns = ['payment_method', 'total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
        payment_methods['revenue_share'] = (payment_methods['total_revenue'] / payment_methods['total_revenue'].sum()) * 100
        
        # Payment method by membership type
        payment_by_membership = self.financial_clean.groupby(['sales_detail_price_level', 'sales_detail_payment_method']).agg({
            'gross_amount': 'sum',
            'unique_key': 'nunique'
        }).reset_index()
        payment_by_membership.columns = ['membership_type', 'payment_method', 'total_revenue', 'unique_customers']
        
        # Payment method trends over time
        payment_trends = self.financial_clean.groupby(['year', 'month', 'sales_detail_payment_method']).agg({
            'gross_amount': 'sum'
        }).reset_index()
        payment_trends['period'] = pd.to_datetime(payment_trends[['year', 'month']].assign(day=1))
        
        return {
            'payment_methods': payment_methods,
            'payment_by_membership': payment_by_membership,
            'payment_trends': payment_trends
        }

    def demand_response_analysis(self):
        """Analyze demand response to pricing changes"""
        print("📈 Analyzing demand response to pricing...")
        
        # Analyze booking frequency by price level
        booking_datasets = []
        
        # Combine booking data
        for df, activity in [(self.bookings_squash, 'Squash'), (self.bookings_classes, 'Classes'), (self.bookings_alt_sessions, 'Alternative'),
                           (self.bookings_other_activities, 'Other Activities')]:
            if not df.empty:
                df_copy = df.copy()
                df_copy['activity_type'] = activity
                if 'Unique' in df_copy.columns:
                    df_copy.rename(columns={'Unique': 'unique_key'}, inplace=True)
                if 'Bookings Detail Start Date' in df_copy.columns:
                    df_copy['booking_date'] = pd.to_datetime(df_copy['Bookings Detail Start Date'])
                booking_datasets.append(df_copy)
        
        if booking_datasets:
            bookings_combined = pd.concat(booking_datasets, ignore_index=True)
            
            # Merge with financial data to get pricing info
            bookings_with_price = bookings_combined.merge(
                self.financial_clean[['unique_key', 'sales_detail_price_level', 'gross_amount']],
                on='unique_key',
                how='left'
            )
            
            # Analyze booking frequency by price level
            booking_frequency = bookings_with_price.groupby(['sales_detail_price_level', 'activity_type']).size().reset_index(name='booking_count')
            
            # Calculate average price by membership type
            avg_price_by_membership = self.financial_clean.groupby('sales_detail_price_level')['gross_amount'].mean().reset_index()
            avg_price_by_membership.columns = ['sales_detail_price_level', 'avg_price']
            
            # Merge booking frequency with pricing
            demand_response = booking_frequency.merge(avg_price_by_membership, on='sales_detail_price_level', how='left')
            
            return {
                'demand_response': demand_response,
                'bookings_with_price': bookings_with_price,
                'booking_frequency': booking_frequency
            }
        
        return {'demand_response': pd.DataFrame(), 'bookings_with_price': pd.DataFrame(), 'booking_frequency': pd.DataFrame()}

    def membership_tier_changes_analysis(self):
        """Analyze membership upgrade/downgrade patterns"""
        print("🔄 Analyzing membership tier changes...")
        
        # Track customers with multiple price levels over time
        customer_tiers = self.financial_clean.groupby(['unique_key', 'sales_detail_price_level']).agg({
            'participation_date': ['min', 'max'],
            'gross_amount': 'sum'
        }).reset_index()
        customer_tiers.columns = ['unique_key', 'price_level', 'first_date', 'last_date', 'total_spent']
        
        # Find customers with tier changes
        tier_changes = customer_tiers.groupby('unique_key').agg({
            'price_level': ['count', 'first', 'last'],
            'first_date': 'min',
            'last_date': 'max',
            'total_spent': 'sum'
        }).reset_index()
        tier_changes.columns = ['unique_key', 'tier_count', 'first_tier', 'last_tier', 'first_date', 'last_date', 'total_spent']
        tier_changes['tier_changed'] = tier_changes['first_tier'] != tier_changes['last_tier']
        tier_changes['customer_lifetime_days'] = (tier_changes['last_date'] - tier_changes['first_date']).dt.days
        
        # Analyze tier transition patterns
        tier_transitions = tier_changes[tier_changes['tier_changed']]
        transition_matrix = tier_transitions.groupby(['first_tier', 'last_tier']).size().reset_index(name='transition_count')
        
        return {
            'tier_changes': tier_changes,
            'tier_transitions': tier_transitions,
            'transition_matrix': transition_matrix
        }

    # SEPARATE VISUALIZATION FUNCTIONS

    def create_pareto_chart(self, analysis_results):
        """Create Pareto chart for revenue concentration"""
        print("Creating Chart 1: Revenue Concentration (Pareto Analysis)")
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        customer_revenue = analysis_results['customer_revenue'].head(50)  # Top 50 for visibility
        
        bars = ax1.bar(range(len(customer_revenue)), customer_revenue['total_revenue'],
                      color='#2E8B57', alpha=0.7, label='Revenue')
        
        ax1_twin = ax1.twinx()
        line = ax1_twin.plot(range(len(customer_revenue)), customer_revenue['cumulative_revenue_pct'],
                           color='#DC143C', marker='o', linewidth=2, markersize=4, label='Cumulative %')
        
        ax1.set_title('Customer Revenue Concentration (Pareto Analysis)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Customer Rank', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Revenue (£)', fontsize=12, fontweight='bold', color='#2E8B57')
        ax1_twin.set_ylabel('Cumulative Revenue %', fontsize=12, fontweight='bold', color='#DC143C')
        
        # Add 80% line
        ax1_twin.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
        
        # Add legends
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_revenue_distribution_pie(self, analysis_results):
        """Create pie chart for revenue distribution"""
        print("Creating Chart 2: Revenue Distribution by Customer Segments")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        pareto_stats = analysis_results['pareto_stats']
        
        sizes = [pareto_stats['top_20_pct_revenue_contribution'],
                100 - pareto_stats['top_20_pct_revenue_contribution']]
        labels = ['Top 20% Customers', 'Remaining 80% Customers']
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 12})
        
        ax.set_title('Revenue Distribution by Customer Segments', fontsize=16, fontweight='bold', pad=20)
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()

    def create_arpu_membership_chart(self, analysis_results):
        """Create ARPU by membership type chart"""
        print("Creating Chart 3: ARPU by Membership Type")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        arpu_membership = analysis_results['arpu_by_membership'].sort_values('arpu', ascending=False)
        
        bars = ax.bar(range(len(arpu_membership)), arpu_membership['arpu'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(arpu_membership))), alpha=0.8)
        
        ax.set_title('ARPU by Membership Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Membership Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('ARPU (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(arpu_membership)))
        ax.set_xticklabels(arpu_membership['membership_type'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_arpu_trends_chart(self, analysis_results):
        """Create ARPU trends over time chart"""
        print("Creating Chart 4: ARPU Trends Over Time")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        arpu_month = analysis_results['arpu_by_month']
        
        ax.plot(arpu_month['period'], arpu_month['arpu'], marker='o', linewidth=3,
               markersize=8, color='#2E8B57')
        ax.fill_between(arpu_month['period'], arpu_month['arpu'], alpha=0.3, color='#2E8B57')
        
        ax.set_title('ARPU Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('ARPU (£)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_arpu_facility_chart(self, analysis_results):
        """Create ARPU by facility chart"""
        print("Creating Chart 5: ARPU by Facility Type")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        arpu_facility = analysis_results['arpu_by_facility']
        
        bars = ax.bar(range(len(arpu_facility)), arpu_facility['arpu'],
                     color=plt.cm.Set3(np.linspace(0, 1, len(arpu_facility))), alpha=0.8)
        
        ax.set_title('ARPU by Facility Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Facility Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('ARPU (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(arpu_facility)))
        ax.set_xticklabels(arpu_facility['facility_type'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_monthly_revenue_chart(self, analysis_results):
        """Create monthly revenue patterns chart"""
        print("Creating Chart 6: Monthly Revenue Patterns")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        monthly_revenue = analysis_results['monthly_revenue']
        
        ax.plot(monthly_revenue['period'], monthly_revenue['total_revenue'],
               marker='o', linewidth=3, markersize=8, color='#2E8B57')
        ax.fill_between(monthly_revenue['period'], monthly_revenue['total_revenue'],
                       alpha=0.3, color='#2E8B57')
        
        ax.set_title('Monthly Revenue Patterns', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Revenue (£)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(monthly_revenue)), monthly_revenue['total_revenue'], 1)
        p = np.poly1d(z)
        ax.plot(monthly_revenue['period'], p(range(len(monthly_revenue))),
               linestyle='--', color='red', alpha=0.7, linewidth=2, label='Trend')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def create_quarterly_revenue_chart(self, analysis_results):
        """Create quarterly revenue chart"""
        print("Creating Chart 7: Quarterly Revenue Distribution")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        quarterly_revenue = analysis_results['quarterly_revenue']
        quarters = quarterly_revenue.groupby('quarter')['total_revenue'].sum().sort_index()
        
        bars = ax.bar(range(len(quarters)), quarters.values,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
        
        ax.set_title('Quarterly Revenue Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Revenue (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels([f'Q{i}' for i in quarters.index])
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'£{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_dow_revenue_chart(self, analysis_results):
        """Create day of week revenue chart"""
        print("Creating Chart 8: Revenue by Day of Week")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        dow_revenue = analysis_results['dow_revenue']
        
        bars = ax.bar(range(len(dow_revenue)), dow_revenue['total_revenue'],
                     color=plt.cm.Set2(np.linspace(0, 1, len(dow_revenue))), alpha=0.8)
        
        ax.set_title('Revenue by Day of Week', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Revenue (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(dow_revenue)))
        ax.set_xticklabels(dow_revenue['day_name'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'£{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_seasonality_index_chart(self, analysis_results):
        """Create seasonality index chart"""
        print("Creating Chart 9: Revenue Seasonality Index (2022-2025)")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        monthly_revenue = analysis_results['monthly_revenue']
        
        # **MODIFIED CODE: Filter data for years 2022-2025**
        filtered_monthly_revenue = monthly_revenue[
            (monthly_revenue['year'] >= 2022) & (monthly_revenue['year'] <= 2025)
        ]
        
        if 'seasonality_index' in filtered_monthly_revenue.columns and not filtered_monthly_revenue.empty:
            bars = ax.bar(range(len(filtered_monthly_revenue)), filtered_monthly_revenue['seasonality_index'],
                         color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(filtered_monthly_revenue))), alpha=0.8)
            
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Average (100)')
            
            ax.set_title('Revenue Seasonality Index (2022-2025)', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_ylabel('Seasonality Index', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(filtered_monthly_revenue)))
            ax.set_xticklabels([f"{int(row['year'])}-{int(row['month']):02d}"
                               for _, row in filtered_monthly_revenue.iterrows()], rotation=45, ha='right')
            
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available for 2022-2025 period', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Revenue Seasonality Index (2022-2025)', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()

    def create_payment_method_pie(self, analysis_results):
        """Create payment method distribution pie chart"""
        print("Creating Chart 10: Payment Method Revenue Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        payment_methods = analysis_results['payment_methods'].sort_values('total_revenue', ascending=False)
        
        # Pie chart for payment method distribution
        sizes = payment_methods['revenue_share'].values
        labels = payment_methods['payment_method'].values
        colors = plt.cm.Set3(np.linspace(0, 1, len(payment_methods)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 10})
        
        ax.set_title('Payment Method Revenue Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()

    def create_payment_transaction_chart(self, analysis_results):
        """Create average transaction value by payment method chart"""
        print("Creating Chart 11: Average Transaction Value by Payment Method")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        payment_methods = analysis_results['payment_methods'].sort_values('total_revenue', ascending=False)
        
        bars = ax.bar(range(len(payment_methods)), payment_methods['avg_transaction'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(payment_methods))), alpha=0.8)
        
        ax.set_title('Average Transaction Value by Payment Method', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Payment Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Transaction (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(payment_methods)))
        ax.set_xticklabels(payment_methods['payment_method'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_payment_trends_chart(self, analysis_results):
        """Create payment method trends chart"""
        print("Creating Chart 12: Payment Method Trends Over Time")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        payment_trends = analysis_results['payment_trends']
        payment_methods = analysis_results['payment_methods']
        
        if not payment_trends.empty:
            # Plot trends for top payment methods
            top_methods = payment_methods.head(4)['payment_method'].tolist()
            for method in top_methods:
                method_data = payment_trends[payment_trends['sales_detail_payment_method'] == method]
                if not method_data.empty:
                    ax.plot(method_data['period'], method_data['gross_amount'],
                           marker='o', linewidth=2, label=method, markersize=4)
        
        ax.set_title('Payment Method Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue (£)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_demand_price_scatter(self, analysis_results):
        """Create price vs demand scatter plot"""
        print("Creating Chart 13: Price vs Demand Relationship")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        demand_response = analysis_results['demand_response']
        
        if not demand_response.empty:
            scatter = ax.scatter(demand_response['avg_price'], demand_response['booking_count'],
                               c=demand_response.index, cmap='viridis', s=100, alpha=0.7)
            
            ax.set_title('Price vs Demand Relationship', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Average Price (£)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Booking Count', fontsize=12, fontweight='bold')
            
            # Add trend line
            if len(demand_response) > 1:
                z = np.polyfit(demand_response['avg_price'], demand_response['booking_count'], 1)
                p = np.poly1d(z)
                ax.plot(demand_response['avg_price'], p(demand_response['avg_price']),
                       "r--", alpha=0.8, linewidth=2, label='Trend')
                ax.legend()
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_demand_activity_chart(self, analysis_results):
        """Create demand by activity type chart"""
        print("Creating Chart 14: Demand by Activity Type")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        demand_response = analysis_results['demand_response']
        
        if not demand_response.empty:
            activity_demand = demand_response.groupby('activity_type')['booking_count'].sum().sort_values(ascending=False)
            
            bars = ax.bar(range(len(activity_demand)), activity_demand.values,
                         color=plt.cm.Set2(np.linspace(0, 1, len(activity_demand))), alpha=0.8)
            
            ax.set_title('Demand by Activity Type', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Activity Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Bookings', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(activity_demand)))
            ax.set_xticklabels(activity_demand.index, rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_tier_change_distribution(self, analysis_results):
        """Create tier change distribution pie chart"""
        print("Creating Chart 15: Membership Tier Change Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        tier_changes = analysis_results['tier_changes']
        change_dist = tier_changes['tier_changed'].value_counts()
        
        colors = ['#4ECDC4', '#FF6B6B']
        labels = ['No Change', 'Tier Changed']
        
        wedges, texts, autotexts = ax.pie(change_dist.values, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        
        ax.set_title('Membership Tier Change Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()

    def create_clv_tier_changes(self, analysis_results):
        """Create CLV by tier changes chart"""
        print("Creating Chart 16: Customer Lifetime Value by Tier Change Status")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        tier_changes = analysis_results['tier_changes']
        clv_by_change = tier_changes.groupby('tier_changed')['total_spent'].agg(['mean', 'median']).reset_index()
        
        x = np.arange(len(clv_by_change))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, clv_by_change['mean'], width, label='Mean',
                      color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x + width/2, clv_by_change['median'], width, label='Median',
                      color='#4682B4', alpha=0.8)
        
        ax.set_title('Customer Lifetime Value by Tier Change Status', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tier Change Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Customer Lifetime Value (£)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No Change', 'Tier Changed'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def create_tenure_distribution(self, analysis_results):
        """Create customer tenure distribution chart"""
        print("Creating Chart 17: Customer Lifetime Distribution (Tier Changers)")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        tier_changes = analysis_results['tier_changes']
        tier_changes_only = tier_changes[tier_changes['tier_changed'] == True]
        
        if not tier_changes_only.empty:
            # Histogram of customer lifetime for tier changers
            ax.hist(tier_changes_only['customer_lifetime_days'], bins=20,
                   color='#FF6B6B', alpha=0.7, edgecolor='black')
            
            ax.set_title('Customer Lifetime Distribution (Tier Changers)', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Customer Lifetime (Days)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean line
            mean_lifetime = tier_changes_only['customer_lifetime_days'].mean()
            ax.axvline(x=mean_lifetime, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_lifetime:.0f} days')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def generate_insights_report(self, all_results):
        """Generate comprehensive insights report"""
        print("Generating Revenue and Pricing Insights Report...")
        
        insights = {
            'revenue_concentration': {},
            'arpu_insights': {},
            'seasonality_insights': {},
            'payment_insights': {},
            'demand_insights': {},
            'tier_change_insights': {}
        }
        
        # Revenue Concentration Insights
        pareto_stats = all_results['revenue_concentration']['pareto_stats']
        insights['revenue_concentration'] = {
            'top_20_pct_contribution': f"{pareto_stats['top_20_pct_revenue_contribution']:.1f}%",
            'customers_for_80_revenue': f"{pareto_stats['customers_for_80_revenue']} customers",
            'total_revenue': f"£{all_results['revenue_concentration']['total_revenue']:,.2f}"
        }
        
        # ARPU Insights
        arpu_data = all_results['arpu_analysis']
        top_arpu_membership = arpu_data['arpu_by_membership'].loc[arpu_data['arpu_by_membership']['arpu'].idxmax()]
        insights['arpu_insights'] = {
            'highest_arpu_membership': f"{top_arpu_membership['membership_type']} (£{top_arpu_membership['arpu']:.0f})",
            'overall_arpu': f"£{arpu_data['arpu_by_membership']['arpu'].mean():.0f}"
        }
        
        # Seasonality Insights
        seasonality_data = all_results['seasonality_analysis']
        peak_month = seasonality_data['monthly_revenue'].loc[seasonality_data['monthly_revenue']['total_revenue'].idxmax()]
        insights['seasonality_insights'] = {
            'peak_month': f"{peak_month['year']}-{peak_month['month']:02d}",
            'peak_revenue': f"£{peak_month['total_revenue']:,.2f}"
        }
        
        # Payment Method Insights
        payment_data = all_results['payment_analysis']
        top_payment_method = payment_data['payment_methods'].loc[payment_data['payment_methods']['total_revenue'].idxmax()]
        insights['payment_insights'] = {
            'top_payment_method': f"{top_payment_method['payment_method']} ({top_payment_method['revenue_share']:.1f}%)",
            'avg_transaction_value': f"£{payment_data['payment_methods']['avg_transaction'].mean():.0f}"
        }
        
        # Tier Change Insights
        tier_data = all_results['tier_changes']
        tier_change_rate = (tier_data['tier_changes']['tier_changed'].sum() / len(tier_data['tier_changes'])) * 100
        insights['tier_change_insights'] = {
            'tier_change_rate': f"{tier_change_rate:.1f}%",
            'total_customers_analyzed': f"{len(tier_data['tier_changes'])}"
        }
        
        return insights

    def run_complete_analysis(self):
        """Run the complete revenue and pricing analysis with separate charts"""
        print("Running Complete Revenue and Pricing Analysis...")
        print("=" * 80)
        
        # Perform all analyses
        print("\n1️⃣ Analyzing Revenue Concentration...")
        revenue_concentration = self.revenue_concentration_analysis()
        
        print("\n2️⃣ Calculating ARPU Analysis...")
        arpu_analysis = self.arpu_analysis()
        
        print("\n3️⃣ Analyzing Revenue Seasonality...")
        seasonality_analysis = self.revenue_seasonality_analysis()
        
        print("\n4️⃣ Analyzing Payment Methods...")
        payment_analysis = self.payment_method_analysis()
        
        print("\n5️⃣ Analyzing Demand Response...")
        demand_response = self.demand_response_analysis()
        
        print("\n6️⃣ Analyzing Membership Tier Changes...")
        tier_changes = self.membership_tier_changes_analysis()
        
        # Combine all results
        all_results = {
            'revenue_concentration': revenue_concentration,
            'arpu_analysis': arpu_analysis,
            'seasonality_analysis': seasonality_analysis,
            'payment_analysis': payment_analysis,
            'demand_response': demand_response,
            'tier_changes': tier_changes
        }
        
        # Create all visualizations as separate charts
        print("\n" + "="*80)
        print("📊 CREATING SEPARATE REVENUE AND PRICING VISUALIZATIONS")
        print("="*80)
        
        # Revenue Concentration Charts
        self.create_pareto_chart(revenue_concentration)
        self.create_revenue_distribution_pie(revenue_concentration)
        
        # ARPU Charts
        self.create_arpu_membership_chart(arpu_analysis)
        self.create_arpu_trends_chart(arpu_analysis)
        self.create_arpu_facility_chart(arpu_analysis)
        
        # Seasonality Charts
        self.create_monthly_revenue_chart(seasonality_analysis)
        self.create_quarterly_revenue_chart(seasonality_analysis)
        self.create_dow_revenue_chart(seasonality_analysis)
        self.create_seasonality_index_chart(seasonality_analysis)
        
        # Payment Method Charts
        self.create_payment_method_pie(payment_analysis)
        self.create_payment_transaction_chart(payment_analysis)
        self.create_payment_trends_chart(payment_analysis)
        
        # Demand Response Charts
        self.create_demand_price_scatter(demand_response)
        self.create_demand_activity_chart(demand_response)
        
        # Tier Changes Charts
        self.create_tier_change_distribution(tier_changes)
        self.create_clv_tier_changes(tier_changes)
        self.create_tenure_distribution(tier_changes)
        
        # Generate insights
        insights = self.generate_insights_report(all_results)
        
        # Print key insights
        print("\n" + "="*80)
        print("🎯 KEY REVENUE AND PRICING INSIGHTS")
        print("="*80)
        
        for category, insight_dict in insights.items():
            print(f"\n📈 {category.upper().replace('_', ' ')}:")
            for key, value in insight_dict.items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*80)
        print("✅ REVENUE AND PRICING ANALYSIS COMPLETE!")
        print("📊 All 17 visualizations displayed as separate, clearly visible charts")
        print("💡 Comprehensive insights generated successfully")
        print("="*80)
        
        return all_results, insights


# Run the analytics
if __name__ == "__main__":
    # Check if prepared data exists
    import os
    
    if not os.path.exists('prepared_data'):
        print("❌ Prepared data directory not found!")
        print("🔄 Please run the data preparation script first:")
        print("   python data_preparation.py")
        exit(1)
    
    # Create analytics instance
    analytics = RevenuePricingAnalytics()
    
    # Run complete analysis
    results, insights = analytics.run_complete_analysis()
    
    print("\n🎉 Enhanced Revenue and Pricing Analytics completed successfully!")
    print("📊 All 17 visualizations are now displayed as separate, clearly visible charts!")
    print("💰 Revenue optimization insights generated for strategic decision making!")
