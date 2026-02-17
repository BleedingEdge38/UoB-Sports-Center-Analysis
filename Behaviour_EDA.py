# temporal_behavioral_analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

class TemporalBehavioralAnalytics:
    def __init__(self, data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        """Initialize the analytics class."""
        self.data_dir = data_dir
        self.attendance = None
        self.bookings_combined = None
        self.cancellations = None
        self.qualitative_feedback = None
        self.load_prepared_data()
        self.prepare_data_for_analysis()

    def load_prepared_data(self):
        """Load prepared datasets from files."""
        print("📂 Loading prepared data...")
        try:
            with open(f'{self.data_dir}/attendance.pkl', 'rb') as f:
                self.attendance = pickle.load(f)
            print("  - Attendance data loaded.")
            
            # Combine all booking datasets if they exist
            booking_files = ['bookings_squash.pkl', 'bookings_classes.pkl', 'bookings_alt_sessions.pkl', 'bookings__other_activities.pkl']
            all_bookings = []
            for bf in booking_files:
                try:
                    with open(f'{self.data_dir}/{bf}', 'rb') as f:
                        all_bookings.append(pickle.load(f))
                except FileNotFoundError:
                    print(f"  - Warning: Booking file {bf} not found.")
            if all_bookings:
                self.bookings_combined = pd.concat(all_bookings, ignore_index=True)
                print("  - All available booking data loaded and combined.")
            else:
                self.bookings_combined = pd.DataFrame()

            # Load cancellation data
            try:
                with open(f'{self.data_dir}/cancellations.pkl', 'rb') as f:
                    self.cancellations = pickle.load(f)
                print("  - Cancellation data loaded.")
            except FileNotFoundError:
                print("  - Warning: Cancellation data not found.")
                self.cancellations = pd.DataFrame()

            # Load qualitative feedback
            try:
                self.qualitative_feedback = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')
                print("  - Qualitative feedback data loaded.")
            except FileNotFoundError:
                print("  - Warning: Qualitative feedback file not found.")
                self.qualitative_feedback = pd.DataFrame()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def prepare_data_for_analysis(self):
        """Prepare and clean data for temporal and behavioral analysis."""
        print("🔧 Preparing data for analysis...")
        if self.attendance is not None:
            self.attendance['date'] = pd.to_datetime(self.attendance['date'])
            self.attendance['hour'] = pd.to_datetime(self.attendance['date_time']).dt.hour
            self.attendance['day_of_week'] = self.attendance['date'].dt.dayofweek
            self.attendance['day_name'] = self.attendance['date'].dt.day_name()
            print("  - Attendance data prepared.")

        if not self.bookings_combined.empty:
            self.bookings_combined['booked_date'] = pd.to_datetime(self.bookings_combined['Bookings Detail Booked Date'], errors='coerce')
            self.bookings_combined['start_date'] = pd.to_datetime(self.bookings_combined['Bookings Detail Start Date'], errors='coerce')
            # Calculate lead time in days
            if 'booked_date' in self.bookings_combined and 'start_date' in self.bookings_combined:
                self.bookings_combined['lead_time_days'] = (self.bookings_combined['start_date'] - self.bookings_combined['booked_date']).dt.days
            print("  - Bookings data prepared.")

    # --- Time Series Analysis ---
    def seasonality_decomposition_analysis(self):
        """Perform and visualize seasonality decomposition."""
        print("\n1. Decomposing Time Series...")
        if self.attendance is None or self.attendance.empty:
            print("   -> Skipping: Attendance data not available.")
            return

        # Resample to daily attendance count
        daily_attendance = self.attendance.set_index('date').resample('D')['unique_key'].count()
        
        # We need at least 2 full periods for decomposition, so check data length
        if len(daily_attendance) < 14:
            print("   -> Skipping: Not enough daily data for meaningful seasonal decomposition.")
            return

        # Decompose the time series
        decomposition = seasonal_decompose(daily_attendance, model='additive', period=7)

        # Plot the decomposed components
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        ax1.set_title('Time Series Decomposition of Daily Attendance', fontsize=16, fontweight='bold')
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

    def usage_heatmaps(self):
        """Create and display usage heatmaps."""
        print("\n2. Generating Usage Heatmaps...")
        if self.attendance is None or self.attendance.empty:
            print("   -> Skipping: Attendance data not available.")
            return
            
        # Create a pivot table for day-of-week and hour-of-day
        heatmap_data = self.attendance.pivot_table(index='day_name', columns='hour', values='unique_key', aggfunc='count')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)

        plt.figure(figsize=(18, 8))
        sns.heatmap(heatmap_data, cmap='inferno', linewidths=.5)
        plt.title('Heatmap of Facility Usage by Day and Hour', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.show()

    def booking_lead_time_analysis(self):
        """Analyze and visualize booking lead times."""
        print("\n3. Analyzing Booking Lead Time...")
        if self.bookings_combined.empty or 'lead_time_days' not in self.bookings_combined.columns:
            print("   -> Skipping: Booking data with lead time not available.")
            return

        # Filter out negative lead times (bookings made after start date)
        lead_time_data = self.bookings_combined[self.bookings_combined['lead_time_days'] >= 0]
        
        plt.figure(figsize=(12, 7))
        sns.histplot(lead_time_data['lead_time_days'], bins=30, kde=True)
        plt.title('Distribution of Booking Lead Time (in days)', fontsize=16, fontweight='bold')
        plt.xlabel('Lead Time (Days)')
        plt.ylabel('Number of Bookings')
        plt.xlim(0, 30) # Focus on the most common range
        plt.show()
        
        # Analyze lead time by activity
        if 'Bookings Detail Activity2' in lead_time_data.columns:
            avg_lead_time = lead_time_data.groupby('Bookings Detail Activity2')['lead_time_days'].mean().sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(14, 8))
            avg_lead_time.plot(kind='barh', color=sns.color_palette('magma', len(avg_lead_time)))
            plt.title('Average Booking Lead Time by Top 10 Activities', fontsize=16, fontweight='bold')
            plt.xlabel('Average Lead Time (Days)')
            plt.ylabel('Activity')
            plt.gca().invert_yaxis()
            plt.show()

    # --- Customer Behavior Clustering ---
    def usage_intensity_segmentation(self):
        """Segment customers based on usage intensity (RFM-style)."""
        print("\n4. Segmenting Customers by Usage Intensity...")
        if self.attendance is None or self.attendance.empty:
            print("   -> Skipping: Attendance data not available.")
            return

        # Calculate Recency, Frequency
        snapshot_date = self.attendance['date'].max() + pd.DateOffset(days=1)
        rfm_data = self.attendance.groupby('unique_key').agg({
            'date': lambda date: (snapshot_date - date.max()).days,
            'unique_key': 'count'
        }).rename(columns={'date': 'Recency', 'unique_key': 'Frequency'})
        
        # Normalize the data
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency']])
        
        # Find the optimal number of clusters using the Elbow Method
        sse = {}
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            sse[k] = kmeans.inertia_
        
        # Plot Elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(list(sse.keys()), list(sse.values()), 'o-')
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE")
        plt.title("Elbow Method for Optimal K")
        plt.show()
        
        # Apply K-Means with the optimal K (e.g., 4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # Analyze the clusters
        cluster_summary = rfm_data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean'
        }).reset_index()
        print("\nCluster Summary:")
        print(cluster_summary)

        # Visualize the clusters
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=rfm_data, x='Recency', y='Frequency', hue='Cluster', palette='bright', s=100)
        plt.title('Customer Segments based on Usage Intensity', fontsize=16, fontweight='bold')
        plt.xlabel('Recency (Days since last visit)')
        plt.ylabel('Frequency (Total visits)')
        plt.gca().invert_xaxis()
        plt.show()

    def activity_preference_clustering(self):
        """Group customers by their preferred activities."""
        print("\n5. Clustering Customers by Activity Preference...")
        if self.attendance is None or self.attendance.empty:
            print("   -> Skipping: Attendance data not available.")
            return

        # Create a user-facility interaction matrix
        user_facility_matrix = self.attendance.pivot_table(index='unique_key', columns='facility_type', aggfunc='size', fill_value=0)
        
        # Normalize the matrix
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(user_facility_matrix)

        # Apply K-Means (e.g., 3 clusters for simplicity)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        user_facility_matrix['Cluster'] = kmeans.fit_predict(matrix_scaled)
        
        # Analyze cluster preferences
        cluster_prefs = user_facility_matrix.groupby('Cluster').mean()
        print("\nActivity Preference Cluster Summary (Avg Visits):")
        print(cluster_prefs)

        # Visualize the cluster preferences
        cluster_prefs.plot(kind='bar', figsize=(14, 8), stacked=False)
        plt.title('Average Facility Visits by Customer Cluster', fontsize=16, fontweight='bold')
        plt.ylabel('Average Number of Visits')
        plt.xlabel('Customer Cluster')
        plt.xticks(rotation=0)
        plt.show()

    def churn_risk_segmentation(self):
        """Identify customers at risk of churning."""
        print("\n6. Identifying Churn Risk...")
        if self.attendance is None or self.attendance.empty:
            print("   -> Skipping: Attendance data not available for churn analysis.")
            return

        # --- Churn Indicator 1: Declining Visit Frequency ---
        # Get usage in last 90 days vs. previous 90 days
        end_date = self.attendance['date'].max()
        last_90_days = self.attendance[(self.attendance['date'] > end_date - pd.DateOffset(days=90))]
        prev_90_days = self.attendance[(self.attendance['date'] <= end_date - pd.DateOffset(days=90)) & 
                                        (self.attendance['date'] > end_date - pd.DateOffset(days=180))]

        last_90_counts = last_90_days.groupby('unique_key').size().rename('last_90_visits')
        prev_90_counts = prev_90_days.groupby('unique_key').size().rename('prev_90_visits')

        churn_risk = pd.concat([last_90_counts, prev_90_counts], axis=1).fillna(0)
        churn_risk['visit_change'] = churn_risk['last_90_visits'] - churn_risk['prev_90_visits']
        
        at_risk_customers = churn_risk[churn_risk['visit_change'] < -5] # Example threshold: dropped by more than 5 visits
        print(f"Identified {len(at_risk_customers)} customers with significant drop in visit frequency.")

        # --- Churn Indicator 2: Match with Cancellation Data ---
        if self.cancellations is not None and not self.cancellations.empty and 'Unique ID' in self.cancellations.columns:
            cancelled_ids = self.cancellations['Unique ID'].dropna().astype(int)
            churn_risk['has_cancelled'] = churn_risk.index.isin(cancelled_ids)
            
            # Visualize
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=churn_risk, x='has_cancelled', y='visit_change')
            plt.title('Change in Visit Frequency for Cancelled vs. Active Customers', fontsize=16, fontweight='bold')
            plt.xlabel('Has Cancelled')
            plt.ylabel('Change in Visits (Last 90 days vs. Previous 90 days)')
            plt.show()
        else:
            print("   -> Note: Cancellation data not available to validate churn indicators.")


# --- Main execution ---
if __name__ == "__main__":
    analytics = TemporalBehavioralAnalytics()
    
    # Run Time Series Analysis
    analytics.seasonality_decomposition_analysis()
    analytics.usage_heatmaps()
    analytics.booking_lead_time_analysis()
    
    # Run Behavioral Clustering
    analytics.usage_intensity_segmentation()
    analytics.activity_preference_clustering()
    analytics.churn_risk_segmentation()
    
    print("\n\n✅ Temporal and Behavioral Analysis Complete!")
