# churn_retention_analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set a professional and clear plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('rocket')

class ChurnRetentionAnalytics:
    """
    A class to perform and visualize churn and retention analysis
    on the sports centre data.
    """
    def __init__(self, data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        """Initialize the analytics class by loading and preparing data."""
        self.data_dir = data_dir
        self.attendance = None
        self.financial = None
        self.cancellations = None
        self.customer_data = None # A merged dataset for comprehensive analysis
        
        self._load_prepared_data()
        self._prepare_data_for_analysis()

    def _load_prepared_data(self):
        """Load all prepared datasets from the specified directory."""
        print("📂 Loading prepared data for Churn and Retention analysis...")
        try:
            with open(os.path.join(self.data_dir, 'attendance.pkl'), 'rb') as f:
                self.attendance = pickle.load(f)
            print("  - Attendance data loaded.")
            with open(os.path.join(self.data_dir, 'financial.pkl'), 'rb') as f:
                self.financial = pickle.load(f)
            print("  - Financial data loaded.")
            with open(os.path.join(self.data_dir, 'cancellations.pkl'), 'rb') as f:
                self.cancellations = pickle.load(f)
            print("  - Cancellation data loaded.")
        except Exception as e:
            print(f"❌ An error occurred while loading data: {e}")
            raise

    def _prepare_data_for_analysis(self):
        """Clean and merge data to create a comprehensive customer view."""
        print("🔧 Preparing and cleaning data...")
        
        # --- Prepare Attendance Data ---
        if self.attendance is not None and not self.attendance.empty:
            self.attendance['date'] = pd.to_datetime(self.attendance['date'], errors='coerce')
            
            # Create a summary of each customer's first and last visit
            customer_activity = self.attendance.groupby('unique_key').agg(
                first_visit=('date', 'min'),
                last_visit=('date', 'max'),
                total_visits=('date', 'count'),
                facilities_used=('facility_type', 'nunique')
            ).reset_index()
            customer_activity['tenure_days'] = (customer_activity['last_visit'] - customer_activity['first_visit']).dt.days
        else:
            print("   -> Warning: Attendance data not found or empty.")
            return

        # --- Prepare Cancellation Data ---
        if self.cancellations is not None and not self.cancellations.empty:
            # Clean column names
           # Before this line
# self.cancellations.dropna(subset=['unique_key', 'effective_from'], inplace=True)
# Add this:
            self.cancellations.columns = (
                self.cancellations.columns
                .str.strip().str.lower().str.replace(' ', '_')
                )
            if 'unique_id' in self.cancellations.columns and 'unique_key' not in self.cancellations.columns:
                self.cancellations.rename(columns={'unique_id': 'unique_key'}, inplace=True)
            if not {'unique_key', 'effective_from'}.issubset(self.cancellations.columns):
                print("   -> SKIPPED: Required fields ['unique_key', 'effective_from'] not available in cancellations data.")
                self.cancellations = pd.DataFrame(columns=['unique_key', 'effective_from', 'reason', 'churned'])
            else:
                # Original cleaning logic
                self.cancellations['effective_from'] = pd.to_datetime(self.cancellations['effective_from'], errors='coerce')
                self.cancellations.dropna(subset=['unique_key', 'effective_from'], inplace=True)
                self.cancellations['unique_key'] = self.cancellations['unique_key'].astype(int)
                self.cancellations['churned'] = 1

        else:
             print("   -> Note: Cancellation data not found or empty.")
             self.cancellations = pd.DataFrame(columns=['unique_key', 'effective_from', 'churned'])
        
        if not self.cancellations.empty:
    # Standardize col names
            self.cancellations.columns = (
            self.cancellations.columns.str.strip().str.lower().str.replace(' ', '_')
            )
    # Coverage for common possible columns:
        possible_reason_cols = [c for c in self.cancellations.columns if 'reason' in c]
        if possible_reason_cols:
        # Always create a column named 'reason' for your logic to use
            self.cancellations['reason'] = self.cancellations[possible_reason_cols[0]]
        else:
            print("[DEBUG] No cancellation reason column found. Columns:", self.cancellations.columns.tolist())
   


        # --- Create a Master Customer DataFrame ---
        # Merge activity with cancellation status
        self.customer_data = pd.merge(customer_activity, self.cancellations[['unique_key', 'effective_from', 'churned']], on='unique_key', how='left')
        self.customer_data['churned'].fillna(0, inplace=True)
        self.customer_data['churned'] = self.customer_data['churned'].astype(int)
        
        # Add join date from financial data if available
        if self.financial is not None and 'Contacts Detail Join Date' in self.financial.columns:
            join_dates = self.financial.groupby('unique_key')['Contacts Detail Join Date'].first().reset_index()
            join_dates['Contacts Detail Join Date'] = pd.to_datetime(join_dates['Contacts Detail Join Date'], errors='coerce')
            self.customer_data = pd.merge(self.customer_data, join_dates, on='unique_key', how='left')
        
        print("✅ Data preparation complete.")

    # --- Cancellation Pattern Analysis ---
    def analyze_cancellation_reasons(self):
        """Categorize and visualize cancellation reasons."""
        print("\n1. Analyzing Cancellation Reasons...")
        if self.cancellations.empty or 'reason' not in self.cancellations.columns:
            print("   -> SKIPPED: Cancellation reason data is not available.")
            return
            
        # Clean and categorize reasons
        def categorize_reason(reason_str):
            if not isinstance(reason_str, str): return 'Other/Unknown'
            reason_str = reason_str.lower()
            if 'relocation' in reason_str or 'moving' in reason_str: return 'Relocation'
            if 'cost' in reason_str or 'financial' in reason_str: return 'Cost'
            if 'lack of use' in reason_str or 'not using' in reason_str: return 'Lack of Use'
            if 'health' in reason_str or 'injury' in reason_str or 'medical' in reason_str: return 'Health/Injury'
            if 'facilities' in reason_str or 'equipment' in reason_str: return 'Facilities/Services'
            if 'left uni' in reason_str or 'employment' in reason_str: return 'Left University/Job'
            if 'personal circumstances' in reason_str: return 'Personal Circumstances'
            if 'cooling off' in reason_str: return 'Cooling Off Period'
            return 'Other/Unknown'

        self.cancellations['reason_category'] = self.cancellations['reason'].apply(categorize_reason)
        reason_counts = self.cancellations['reason_category'].value_counts()
        
        plt.figure(figsize=(14, 8))
        reason_counts.plot(kind='barh', color=sns.color_palette('magma_r', len(reason_counts)))
        plt.title('Top Reasons for Membership Cancellation', fontsize=18, fontweight='bold')
        plt.xlabel('Number of Cancellations')
        plt.ylabel('Reason Category')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def analyze_time_to_churn(self):
        """Analyze the average membership duration before churn."""
        print("\n2. Analyzing Time-to-Churn...")
        if self.customer_data is None or 'Contacts Detail Join Date' not in self.customer_data.columns:
            print("   -> SKIPPED: Join date information is not available for this analysis.")
            return
            
        churned_customers = self.customer_data[self.customer_data['churned'] == 1].dropna(subset=['Contacts Detail Join Date', 'effective_from'])
        churned_customers['time_to_churn'] = (churned_customers['effective_from'] - churned_customers['Contacts Detail Join Date']).dt.days

        # Filter out nonsensical data (e.g., negative duration)
        churned_customers = churned_customers[churned_customers['time_to_churn'] >= 0]
        
        plt.figure(figsize=(14, 7))
        sns.histplot(churned_customers['time_to_churn'], bins=50, kde=True, color='indianred')
        plt.title('Distribution of Membership Duration Before Churn', fontsize=18, fontweight='bold')
        plt.xlabel('Membership Duration (Days)')
        plt.ylabel('Number of Churned Members')
        plt.xlim(0, churned_customers['time_to_churn'].quantile(0.95)) # Focus on 95% of data
        plt.show()

    def analyze_seasonal_churn(self):
        """Identify seasonal patterns in member churn."""
        print("\n3. Analyzing Seasonal Churn Patterns...")
        if self.cancellations.empty or 'effective_from' not in self.cancellations.columns:
            print("   -> SKIPPED: Cancellation data with effective dates is not available.")
            return

        self.cancellations['churn_month'] = self.cancellations['effective_from'].dt.month_name()
        monthly_churn = self.cancellations['churn_month'].value_counts()
        
        # Order months correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_churn = monthly_churn.reindex(month_order)
        
        plt.figure(figsize=(14, 7))
        monthly_churn.plot(kind='bar', color=sns.color_palette('crest', 12))
        plt.title('Total Cancellations by Month', fontsize=18, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Cancellations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_churn_indicators(self):
        """Analyze usage patterns leading up to cancellation."""
        print("\n4. Analyzing Predictive Churn Indicators (Usage Decline)...")
        if self.customer_data is None or 'effective_from' not in self.customer_data.columns:
            print("   -> SKIPPED: Cannot analyze churn indicators without cancellation dates.")
            return

        churned_users_with_date = self.customer_data[self.customer_data['churned'] == 1].dropna(subset=['effective_from'])
        
        usage_change = []
        for index, row in churned_users_with_date.iterrows():
            user_id = row['unique_key']
            churn_date = row['effective_from']
            user_attendance = self.attendance[self.attendance['unique_key'] == user_id]
            
            # Visits in 90 days before churn vs. 91-180 days before
            visits_before_churn = user_attendance[(user_attendance['date'] < churn_date) & 
                                                  (user_attendance['date'] >= churn_date - pd.DateOffset(days=90))].shape[0]
            visits_long_before = user_attendance[(user_attendance['date'] < churn_date - pd.DateOffset(days=90)) & 
                                                  (user_attendance['date'] >= churn_date - pd.DateOffset(days=180))].shape[0]
            
            if visits_long_before > 0: # Only include users with a baseline
                usage_change.append({'unique_key': user_id, 'change': visits_before_churn - visits_long_before})
        
        if not usage_change:
            print("   -> SKIPPED: Not enough historical data for churn indicator analysis.")
            return

        usage_change_df = pd.DataFrame(usage_change)
        
        plt.figure(figsize=(14, 7))
        sns.histplot(usage_change_df['change'], bins=20, kde=True, color='salmon')
        plt.title('Change in Visit Frequency in 90 Days Before Cancellation', fontsize=18, fontweight='bold')
        plt.xlabel('Change in Number of Visits (Last 90 days vs Previous 90 days)')
        plt.ylabel('Number of Churned Members')
        plt.axvline(0, color='black', linestyle='--')
        plt.show()

    # --- Retention Driver Analysis ---
    def profile_high_value_customers(self):
        """Profile the characteristics of the most loyal customers."""
        print("\n5. Profiling High-Value (Loyal) Customers...")
        if self.customer_data is None:
            print("   -> SKIPPED: No customer data to analyze.")
            return
            
        # Define loyal customers: long tenure, not churned, and recently active
        snapshot_date = self.customer_data['last_visit'].max()
        loyal_customers = self.customer_data[
            (self.customer_data['churned'] == 0) & 
            (self.customer_data['tenure_days'] > 365) & # Tenure over a year
            (self.customer_data['last_visit'] > snapshot_date - pd.DateOffset(days=90)) # Active in last 3 months
        ]
        
        if loyal_customers.empty:
            print("   -> SKIPPED: No customers fit the 'loyal' criteria.")
            return
            
        # Merge with demographic data
        if 'price_level' in self.attendance.columns:
            demographics = self.attendance.groupby('unique_key').agg(
                price_level=('price_level', 'first'),
                age=('age', 'first'),
                gender=('gender', 'first')
            ).reset_index()
            loyal_profile = pd.merge(loyal_customers, demographics, on='unique_key', how='left')
        else:
            loyal_profile = loyal_customers
        
        print(f"   -> Found {len(loyal_profile)} loyal customers to profile.")
        
        # Visualize Price Level
        if 'price_level' in loyal_profile.columns:
            plt.figure(figsize=(14, 7))
            loyal_profile['price_level'].value_counts().head(10).plot(kind='bar', color=sns.color_palette('summer'))
            plt.title('Membership Tiers of Loyal Customers', fontsize=18, fontweight='bold')
            plt.ylabel('Number of Loyal Customers')
            plt.xticks(rotation=45, ha='right')
            plt.show()

        # Visualize Age Distribution
        if 'age' in loyal_profile.columns:
            plt.figure(figsize=(12, 7))
            sns.histplot(loyal_profile['age'].dropna(), bins=20, kde=True, color='green')
            plt.title('Age Distribution of Loyal Customers', fontsize=18, fontweight='bold')
            plt.xlabel('Age')
            plt.show()

    def analyze_facility_usage_retention(self):
        """Compare facility usage between loyal and churned customers."""
        print("\n6. Correlating Facility Usage with Retention...")
        if self.customer_data is None:
            print("   -> SKIPPED: No customer data to analyze.")
            return

        # Get unique keys for churned vs. loyal customers
        churned_ids = self.customer_data[self.customer_data['churned'] == 1]['unique_key'].unique()
        loyal_ids = self.customer_data[self.customer_data['churned'] == 0]['unique_key'].unique()

        # Get attendance for each group
        churned_attendance = self.attendance[self.attendance['unique_key'].isin(churned_ids)]
        loyal_attendance = self.attendance[self.attendance['unique_key'].isin(loyal_ids)]

        # Calculate average visits per facility
        churned_facility_usage = churned_attendance.groupby('facility_type').size() / len(churned_ids) if len(churned_ids) > 0 else pd.Series()
        loyal_facility_usage = loyal_attendance.groupby('facility_type').size() / len(loyal_ids) if len(loyal_ids) > 0 else pd.Series()
        
        if churned_facility_usage.empty and loyal_facility_usage.empty:
            print("   -> SKIPPED: Not enough data to compare facility usage.")
            return
            
        usage_comparison = pd.DataFrame({'Loyal': loyal_facility_usage, 'Churned': churned_facility_usage}).fillna(0)
        
        usage_comparison.plot(kind='bar', figsize=(14, 8), color={'Loyal': 'darkcyan', 'Churned': 'orangered'})
        plt.title('Average Visits per Facility: Loyal vs. Churned Customers', fontsize=18, fontweight='bold')
        plt.ylabel('Average Number of Visits per Customer')
        plt.xlabel('Facility Type')
        plt.xticks(rotation=0)
        plt.show()

    def develop_engagement_score(self):
        """Develop and analyze a customer engagement score."""
        print("\n7. Developing a Customer Engagement Score...")
        if self.customer_data is None:
            print("   -> SKIPPED: No customer data available for this analysis.")
            return

        # Use Recency, Frequency, and Visit Variety (facilities_used)
        engagement_df = self.customer_data[['unique_key', 'churned', 'last_visit', 'total_visits', 'facilities_used']].copy()
        snapshot_date = engagement_df['last_visit'].max() + pd.DateOffset(days=1)
        engagement_df['recency'] = (snapshot_date - engagement_df['last_visit']).dt.days
        
        # Rename for clarity
        engagement_df.rename(columns={'total_visits': 'frequency', 'facilities_used': 'variety'}, inplace=True)
        
        # Select features for scoring and scale them
        features = engagement_df[['recency', 'frequency', 'variety']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create engagement score (invert recency as lower is better)
        engagement_df['engagement_score'] = features_scaled[:, 1] + features_scaled[:, 2] - features_scaled[:, 0]
        
        # Visualize the score distribution for churned vs. retained
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=engagement_df, x='churned', y='engagement_score', palette='pastel')
        plt.title('Engagement Score Distribution: Retained vs. Churned Customers', fontsize=18, fontweight='bold')
        plt.xlabel('Customer Has Churned')
        plt.ylabel('Engagement Score')
        plt.xticks([0, 1], ['No (Retained)', 'Yes (Churned)'])
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    analytics = ChurnRetentionAnalytics()
    
    # Run Cancellation Pattern Analysis
    analytics.analyze_cancellation_reasons()
    analytics.analyze_time_to_churn()
    analytics.analyze_seasonal_churn()
    analytics.analyze_churn_indicators()
    
    # Run Retention Driver Analysis
    analytics.profile_high_value_customers()
    analytics.analyze_facility_usage_retention()
    analytics.develop_engagement_score()
    
    print("\n\n✅ Churn and Retention Analysis is Complete!")
