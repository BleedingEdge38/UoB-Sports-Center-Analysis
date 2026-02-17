import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def drop_duplicate_columns(df):
    """Drop duplicate columns, keeping first occurrence."""
    v = pd.Series(df.columns)
    dups = v[v.duplicated()].unique()
    if len(dups):
        print(f"❗ Duplicate columns detected: {list(dups)}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print(f"✅ Duplicates dropped. Columns now: {list(df.columns)}")
    return df

def canonicalize_financial_columns(df):
    """Canonicalize column names to standard format for consistent processing"""
    if df is None or df.empty:
        return df
    
    column_mapping = {
        'unique_key': 'unique_id',
        'sales_detail_participation_date': 'date',
        'sales_detail_gross_amount': 'price',
        'sales_detail_net_amount': 'price',
        'sales_detail_quantity': 'quantity',
        'product_hierarchy_product': 'product',
        'sales_detail_price_level': 'customer_type',
        'contacts_detail_price_level': 'customer_type',
        'attendance_detail_date': 'date',
        'attendance_detail_date_time': 'date'
    }
    
    lower_col_map = {k.lower(): v for k, v in column_mapping.items()}
    new_cols = {}
    
    for col in df.columns:
        canonical_name = lower_col_map.get(col.lower())
        if canonical_name:
            new_cols[col] = canonical_name
    
    if new_cols:
        df = df.rename(columns=new_cols)
        print(f"✅ Mapped columns: {new_cols}")
    
    return drop_duplicate_columns(df)

class AdvancedCustomerSegmentation:
    def __init__(self, cache_dir='./data_cache', pkl_data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pkl_data_dir = pkl_data_dir
        self.financial_data = None
        self.attendance_data = None
        self.user_data = None
        self.customer_segments = None
        
        # Primary member profiles to focus on
        self.primary_profiles = [
            'Student Member', 'Staff Member', 'Alumni Member',
            'Community Member', 'Associate Member',
            'Community Over 65 Member', 'Staff Non Member'
        ]
    
    def load_pkl_datasets(self):
        """Load pickle datasets"""
        datasets = {}
        pkl_files = {
            'financial': 'financial.pkl',
            'attendance': 'attendance.pkl',
        }
        
        for name, filename in pkl_files.items():
            try:
                if os.path.exists(f'{self.pkl_data_dir}/{filename}'):
                    with open(f'{self.pkl_data_dir}/{filename}', 'rb') as f:
                        datasets[name] = pickle.load(f)
                    print(f"✅ Loaded {name} data from {filename}")
                else:
                    print(f"⚠️ {filename} not found, will try alternative loading...")
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
        
        return datasets
    
    def load_excel_fallback(self, dataset_name):
        """Load data from Excel files as fallback"""
        try:
            if dataset_name == 'financial':
                financial_sheets = pd.read_excel('Financial_Data.xlsx', sheet_name=['S&F', 'Tiverton'])
                df = pd.concat([financial_sheets['S&F'], financial_sheets['Tiverton']], ignore_index=True)
                return drop_duplicate_columns(df)
            elif dataset_name == 'attendance':
                attendance_sheets = pd.read_excel('Att_data.xlsx', 
                    sheet_name=['Gym', 'Gym 2','Reception Barrier 1','Reception Barrier 2', 'Recpetion Barrier 3', 'Reception Barrier 4'])
                df = pd.concat([sheet for sheet in attendance_sheets.values()], ignore_index=True)
                return drop_duplicate_columns(df)
            elif dataset_name == 'user':
                df = pd.read_excel('User_data.xlsx')
                return drop_duplicate_columns(df)
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} from Excel: {e}")
            return None
    
    def load_and_preprocess_data(self):
        """Load and preprocess all required data"""
        print("\n🔄 Starting data preprocessing for customer segmentation...")
        
        try:
            # Try to load from pickle first
            pkl_datasets = self.load_pkl_datasets()
            
            # Load financial data
            if 'financial' in pkl_datasets and pkl_datasets['financial'] is not None:
                self.financial_data = pkl_datasets['financial']
                if isinstance(self.financial_data, dict):
                    self.financial_data = pd.concat([
                        self.financial_data['S&F'], 
                        self.financial_data['Tiverton']
                    ], ignore_index=True)
            else:
                self.financial_data = self.load_excel_fallback('financial')
            
            # Load attendance data
            if 'attendance' in pkl_datasets and pkl_datasets['attendance'] is not None:
                self.attendance_data = pkl_datasets['attendance']
                if isinstance(self.attendance_data, dict):
                    all_attendance = []
                    for sheet_data in self.attendance_data.values():
                        if sheet_data is not None:
                            all_attendance.append(sheet_data)
                    self.attendance_data = pd.concat(all_attendance, ignore_index=True) if all_attendance else pd.DataFrame()
            else:
                self.attendance_data = self.load_excel_fallback('attendance')
            
            # Load user data
            self.user_data = self.load_excel_fallback('user')
            
            # Process and canonicalize columns
            if self.financial_data is not None:
                self.financial_data = canonicalize_financial_columns(self.financial_data)
                self.financial_data = drop_duplicate_columns(self.financial_data)
            
            if self.attendance_data is not None:
                self.attendance_data = canonicalize_financial_columns(self.attendance_data)
                self.attendance_data = drop_duplicate_columns(self.attendance_data)
            
            if self.user_data is not None:
                self.user_data = drop_duplicate_columns(self.user_data)
            
            # Filter for primary profiles
            self._filter_primary_profiles()
            
            # Preprocess dates
            self._preprocess_dates()
            
            print("✅ Data loaded and preprocessed successfully!")
            
        except Exception as e:
            print(f"❌ Error during data loading: {e}")
            raise
    
    def _filter_primary_profiles(self):
        """Filter data for primary member profiles only"""
        print("🎯 Filtering data for primary member profiles...")
        
        if self.financial_data is not None and 'customer_type' in self.financial_data.columns:
            initial_count = len(self.financial_data)
            self.financial_data = self.financial_data[
                self.financial_data['customer_type'].isin(self.primary_profiles)
            ]
            self.financial_data = self.financial_data[self.financial_data['price'] > 0]
            print(f" Financial data: {initial_count} → {len(self.financial_data)} records")
        
        if self.user_data is not None:
            price_level_col = None
            for col in self.user_data.columns:
                if 'price level' in col.lower():
                    price_level_col = col
                    break
            
            if price_level_col:
                initial_count = len(self.user_data)
                self.user_data = self.user_data[
                    self.user_data[price_level_col].isin(self.primary_profiles)
                ]
                print(f" User data: {initial_count} → {len(self.user_data)} records")
    
    def _preprocess_dates(self):
        """Preprocess date columns"""
        try:
            if self.financial_data is not None and 'date' in self.financial_data.columns:
                self.financial_data['date'] = pd.to_datetime(self.financial_data['date'], errors='coerce')
            
            if self.attendance_data is not None and 'date' in self.attendance_data.columns:
                self.attendance_data['date'] = pd.to_datetime(self.attendance_data['date'], errors='coerce')
                
        except Exception as e:
            print(f"⚠️ Warning during date preprocessing: {e}")
    
    def create_behavioral_features(self):
        """Create behavioral features for segmentation"""
        print("📊 Creating behavioral features...")
        
        if self.financial_data is None:
            print("❌ No financial data available")
            return None
        
        # Create customer behavioral features
        customer_features = self.financial_data.groupby('unique_id').agg({
            'quantity': ['sum', 'mean', 'std'],
            'price': ['mean', 'std', 'sum'],
            'date': ['count', 'nunique'],
            'customer_type': 'first'
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'customer_id', 'total_quantity', 'avg_quantity', 'quantity_std',
            'avg_price', 'price_std', 'total_revenue', 'total_transactions', 
            'active_months', 'customer_type'
        ]
        
        # Fill NaN values
        customer_features['quantity_std'] = customer_features['quantity_std'].fillna(0)
        customer_features['price_std'] = customer_features['price_std'].fillna(0)
        
        # Add attendance data if available
        if self.attendance_data is not None and 'unique_id' in self.attendance_data.columns:
            attendance_features = self.attendance_data.groupby('unique_id').agg({
                'date': 'count'
            }).reset_index()
            attendance_features.columns = ['customer_id', 'total_visits']
            
            customer_features = customer_features.merge(
                attendance_features, on='customer_id', how='left'
            )
            customer_features['total_visits'] = customer_features['total_visits'].fillna(0)
        else:
            customer_features['total_visits'] = customer_features['total_transactions']
        
        # Calculate additional behavioral metrics
        customer_features['avg_transaction_value'] = (
            customer_features['total_revenue'] / customer_features['total_transactions']
        ).fillna(0)
        
        customer_features['visit_frequency'] = (
            customer_features['total_visits'] / customer_features['active_months']
        ).fillna(0)
        
        # Calculate price sensitivity (simplified version)
        customer_features['price_sensitivity'] = (
            customer_features['price_std'] / customer_features['avg_price']
        ).fillna(0)
        customer_features['price_sensitivity'] = customer_features['price_sensitivity'].replace([np.inf, -np.inf], 0)
        
        print(f"✅ Created behavioral features for {len(customer_features)} customers")
        return customer_features
    
    def behavioral_customer_segmentation(self, customer_features):
        """Create behaviorally-driven customer segments"""
        print("🎯 Performing behavioral customer segmentation...")
        
        # Select features for clustering
        feature_cols = [
            'total_visits', 'avg_quantity', 'avg_price', 'total_revenue',
            'visit_frequency', 'price_sensitivity', 'avg_transaction_value'
        ]
        
        # Prepare features
        features_df = customer_features[feature_cols].copy()
        
        # Handle any remaining infinite or NaN values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(3, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            score = silhouette_score(scaled_features, labels)
            silhouette_scores.append(score)
        
        # Select optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"📈 Optimal number of clusters: {optimal_k}")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        customer_features['behavioral_segment'] = kmeans.fit_predict(scaled_features)
        
        # Analyze segments and assign meaningful labels
        segment_analysis = customer_features.groupby('behavioral_segment').agg({
            'total_visits': 'mean',
            'avg_price': 'mean',
            'total_revenue': 'mean',
            'visit_frequency': 'mean',
            'price_sensitivity': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        # Create segment labels based on characteristics
        segment_labels = self._create_segment_labels(segment_analysis)
        
        customer_features['segment_name'] = customer_features['behavioral_segment'].map(segment_labels)
        
        # Store results
        self.customer_segments = customer_features
        self.segment_analysis = segment_analysis
        self.scaler = scaler
        self.kmeans = kmeans
        
        print(f"✅ Successfully segmented {len(customer_features)} customers into {optimal_k} behavioral segments")
        return customer_features
    
    def _create_segment_labels(self, segment_analysis):
        """Create meaningful labels for segments based on their characteristics"""
        labels = {}
        
        for segment in segment_analysis.index:
            row = segment_analysis.loc[segment]
            
            # Determine segment characteristics
            if row['avg_price'] > segment_analysis['avg_price'].median():
                price_category = "Premium"
            else:
                price_category = "Value"
            
            if row['visit_frequency'] > segment_analysis['visit_frequency'].median():
                frequency_category = "Regular"
            else:
                frequency_category = "Occasional"
            
            if row['total_revenue'] > segment_analysis['total_revenue'].median():
                revenue_category = "High-Value"
            else:
                revenue_category = "Standard"
            
            # Combine characteristics for label
            if price_category == "Premium" and frequency_category == "Regular":
                labels[segment] = "Premium Regular Users"
            elif price_category == "Value" and frequency_category == "Regular":
                labels[segment] = "Value-Conscious Regulars"
            elif price_category == "Premium" and frequency_category == "Occasional":
                labels[segment] = "Premium Occasionals"
            elif revenue_category == "High-Value":
                labels[segment] = "High-Value Members"
            else:
                labels[segment] = "Standard Members"
        
        return labels
    
    def create_segment_profile_visualization(self):
        """Create comprehensive segment profile visualization"""
        if self.customer_segments is None:
            print("❌ No segmentation results available")
            return
        
        # Create a comprehensive profile chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Behavioral Customer Segments - Profile Analysis\nUoB Sports & Fitness Centre', 
                     fontsize=16, fontweight='bold')
        
        # Segment size distribution
        segment_counts = self.customer_segments['segment_name'].value_counts()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(segment_counts))))
        axes[0, 0].set_title('Segment Size Distribution', fontweight='bold')
        
        # Average price by segment
        price_by_segment = self.customer_segments.groupby('segment_name')['avg_price'].mean().sort_values(ascending=True)
        axes[0, 1].barh(price_by_segment.index, price_by_segment.values, color='skyblue')
        axes[0, 1].set_title('Average Price by Segment', fontweight='bold')
        axes[0, 1].set_xlabel('Average Price (£)')
        
        # Total visits by segment
        visits_by_segment = self.customer_segments.groupby('segment_name')['total_visits'].mean().sort_values(ascending=True)
        axes[0, 2].barh(visits_by_segment.index, visits_by_segment.values, color='lightgreen')
        axes[0, 2].set_title('Average Visits by Segment', fontweight='bold')
        axes[0, 2].set_xlabel('Average Total Visits')
        
        # Total revenue by segment
        revenue_by_segment = self.customer_segments.groupby('segment_name')['total_revenue'].mean().sort_values(ascending=True)
        axes[1, 0].barh(revenue_by_segment.index, revenue_by_segment.values, color='salmon')
        axes[1, 0].set_title('Average Revenue by Segment', fontweight='bold')
        axes[1, 0].set_xlabel('Average Total Revenue (£)')
        
        # Visit frequency by segment
        frequency_by_segment = self.customer_segments.groupby('segment_name')['visit_frequency'].mean().sort_values(ascending=True)
        axes[1, 1].barh(frequency_by_segment.index, frequency_by_segment.values, color='gold')
        axes[1, 1].set_title('Visit Frequency by Segment', fontweight='bold')
        axes[1, 1].set_xlabel('Visits per Month')
        
        # Price sensitivity by segment
        sensitivity_by_segment = self.customer_segments.groupby('segment_name')['price_sensitivity'].mean().sort_values(ascending=True)
        axes[1, 2].barh(sensitivity_by_segment.index, sensitivity_by_segment.values, color='mediumpurple')
        axes[1, 2].set_title('Price Sensitivity by Segment', fontweight='bold')
        axes[1, 2].set_xlabel('Price Sensitivity Score')
        
        plt.tight_layout()
        plt.savefig('customer_segments_profile.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def create_segment_scatter_analysis(self):
        """Create scatter plot analysis of segments"""
        if self.customer_segments is None:
            print("❌ No segmentation results available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Behavioral Customer Segments - Relationship Analysis\nUoB Sports & Fitness Centre', 
                     fontsize=16, fontweight='bold')
        
        # Price vs Visits
        for segment in self.customer_segments['segment_name'].unique():
            segment_data = self.customer_segments[self.customer_segments['segment_name'] == segment]
            axes[0, 0].scatter(segment_data['avg_price'], segment_data['total_visits'], 
                              label=segment, alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Average Price (£)')
        axes[0, 0].set_ylabel('Total Visits')
        axes[0, 0].set_title('Price vs Total Visits')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Revenue vs Frequency
        for segment in self.customer_segments['segment_name'].unique():
            segment_data = self.customer_segments[self.customer_segments['segment_name'] == segment]
            axes[0, 1].scatter(segment_data['total_revenue'], segment_data['visit_frequency'], 
                              label=segment, alpha=0.7, s=60)
        axes[0, 1].set_xlabel('Total Revenue (£)')
        axes[0, 1].set_ylabel('Visit Frequency (visits/month)')
        axes[0, 1].set_title('Revenue vs Visit Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price Sensitivity vs Average Price
        for segment in self.customer_segments['segment_name'].unique():
            segment_data = self.customer_segments[self.customer_segments['segment_name'] == segment]
            axes[1, 0].scatter(segment_data['price_sensitivity'], segment_data['avg_price'], 
                              label=segment, alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Price Sensitivity')
        axes[1, 0].set_ylabel('Average Price (£)')
        axes[1, 0].set_title('Price Sensitivity vs Average Price')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Transaction Value vs Visits
        for segment in self.customer_segments['segment_name'].unique():
            segment_data = self.customer_segments[self.customer_segments['segment_name'] == segment]
            axes[1, 1].scatter(segment_data['avg_transaction_value'], segment_data['total_visits'], 
                              label=segment, alpha=0.7, s=60)
        axes[1, 1].set_xlabel('Average Transaction Value (£)')
        axes[1, 1].set_ylabel('Total Visits')
        axes[1, 1].set_title('Transaction Value vs Total Visits')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customer_segments_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def create_segment_heatmap(self):
        """Create heatmap showing segment characteristics"""
        if self.customer_segments is None:
            print("❌ No segmentation results available")
            return
        
        # Create segment summary
        segment_summary = self.customer_segments.groupby('segment_name').agg({
            'total_visits': 'mean',
            'avg_price': 'mean',
            'total_revenue': 'mean',
            'visit_frequency': 'mean',
            'price_sensitivity': 'mean',
            'avg_transaction_value': 'mean'
        }).round(2)
        
        # Normalize for better visualization
        segment_summary_norm = segment_summary.div(segment_summary.max())
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(segment_summary_norm.T, annot=segment_summary.T, fmt='.2f', 
                    cmap='YlOrRd', cbar_kws={'label': 'Normalized Score'})
        plt.title('Customer Segment Characteristics Heatmap\nUoB Sports & Fitness Centre', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Customer Segments', fontweight='bold')
        plt.ylabel('Behavioral Metrics', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('customer_segments_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def create_customer_type_distribution(self):
        """Create distribution of customer types within segments"""
        if self.customer_segments is None:
            print("❌ No segmentation results available")
            return

        # Create crosstab of segments vs customer types
        segment_customer_type = pd.crosstab(
            self.customer_segments['segment_name'],
            self.customer_segments['customer_type'],
            normalize='index'
        ) * 100

        # Define distinct colors for each customer type
        import matplotlib.colors as mcolors
        
        # Get all unique customer types
        customer_types = segment_customer_type.columns
        
        # Create a distinct color palette
        # Using a combination of different color maps to ensure distinction
        colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#FFA07A',  # Light Salmon
            '#98D8C8',  # Mint
            '#F7DC6F',  # Light Yellow
            '#BB8FCE'   # Light Purple
        ]
        
        # Ensure we have enough colors
        if len(customer_types) > len(colors):
            # Generate additional colors using colormap
            additional_colors = plt.cm.Set3(np.linspace(0, 1, len(customer_types) - len(colors)))
            colors.extend([mcolors.rgb2hex(color) for color in additional_colors])
        
        # Create color mapping
        color_map = {customer_type: colors[i] for i, customer_type in enumerate(customer_types)}
        
        plt.figure(figsize=(14, 8))
        
        # Create the stacked bar plot with custom colors
        ax = segment_customer_type.plot(
            kind='bar', 
            stacked=True, 
            ax=plt.gca(),
            color=[color_map[col] for col in segment_customer_type.columns]
        )
        
        plt.title('Customer Type Distribution within Behavioral Segments\nUoB Sports & Fitness Centre',
                fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Behavioral Segments', fontweight='bold')
        plt.ylabel('Percentage (%)', fontweight='bold')
        plt.legend(title='Customer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('customer_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    
    def generate_segment_insights(self):
        """Generate strategic insights for each segment"""
        if self.customer_segments is None:
            print("❌ No segmentation results available")
            return

        print("\n" + "="*80)
        print("🎯 BEHAVIORAL CUSTOMER SEGMENTATION INSIGHTS")
        print("🎯 UoB SPORTS & FITNESS CENTRE")
        print("="*80)

        for segment in self.customer_segments['segment_name'].unique():
            segment_data = self.customer_segments[self.customer_segments['segment_name'] == segment]
            print(f"\n📊 {segment} ({len(segment_data)} customers - {len(segment_data)/len(self.customer_segments)*100:.1f}%):")
            print(f"   Average Revenue: £{segment_data['total_revenue'].mean():.2f}")
            print(f"   Average Price: £{segment_data['avg_price'].mean():.2f}")
            print(f"   Average Visits: {segment_data['total_visits'].mean():.1f}")
            print(f"   Visit Frequency: {segment_data['visit_frequency'].mean():.2f} visits/month")
            print(f"   Price Sensitivity: {segment_data['price_sensitivity'].mean():.3f}")

            # Top customer types in this segment
            top_customer_types = segment_data['customer_type'].value_counts().head(3)
            print(f"   Top Customer Types: {', '.join([f'{k} ({v})' for k, v in top_customer_types.items()])}")

            # Strategic recommendations
            recommendations = self._get_segment_recommendations(segment, segment_data)
            print(f"   💡 Strategy: {recommendations}")

        # Overall insights
        total_customers = len(self.customer_segments)
        total_revenue = self.customer_segments['total_revenue'].sum()
        print(f"\n🎯 OVERALL INSIGHTS:")
        print(f"   Total Customers Analyzed: {total_customers:,}")
        print(f"   Total Revenue: £{total_revenue:,.2f}")
        print(f"   Average Revenue per Customer: £{total_revenue/total_customers:.2f}")

        # Identify high-value segments - FIX HERE
        high_value_segments = self.customer_segments.groupby('segment_name')['total_revenue'].mean().sort_values(ascending=False)
        highest_value_segment = high_value_segments.index[0]
        highest_value_amount = high_value_segments.iloc  # Get the actual value
        print(f"   Highest Value Segment: {highest_value_segment} (£{highest_value_amount:.2f} avg)")
        
        largest_segment_name = self.customer_segments['segment_name'].value_counts().index
        largest_segment_count = self.customer_segments['segment_name'].value_counts().iloc
        print(f"   Largest Segment: {largest_segment_name} ({largest_segment_count} customers)")

        print("\n" + "="*80)

    
    def _get_segment_recommendations(self, segment_name, segment_data):
        """Generate strategic recommendations for each segment"""
        avg_revenue = segment_data['total_revenue'].mean()
        avg_frequency = segment_data['visit_frequency'].mean()
        price_sensitivity = segment_data['price_sensitivity'].mean()
        
        if "Premium" in segment_name:
            if avg_frequency > 10:
                return "Loyalty rewards, exclusive services, premium facilities access"
            else:
                return "Engagement campaigns, exclusive events, value-added services"
        elif "Value" in segment_name or price_sensitivity > 0.3:
            return "Cost-effective packages, bulk discounts, off-peak promotions"
        elif "High-Value" in segment_name:
            return "VIP treatment, personalized services, premium upgrades"
        elif avg_frequency > 15:
            return "Retention focus, referral programs, community building"
        else:
            return "Activation campaigns, usage incentives, onboarding improvements"
    
    def run_complete_analysis(self):
        """Run the complete customer segmentation analysis"""
        print("🚀 Starting Advanced Customer Segmentation Analysis")
        print("📊 UoB Sports & Fitness Centre")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create behavioral features
        customer_features = self.create_behavioral_features()
        
        if customer_features is None:
            print("❌ Cannot proceed without customer features")
            return
        
        # Perform segmentation
        self.behavioral_customer_segmentation(customer_features)
        
        # Generate visualizations
        print("\n📊 Creating visualizations...")
        self.create_segment_profile_visualization()
        self.create_segment_scatter_analysis()
        self.create_segment_heatmap()
        self.create_customer_type_distribution()
        
        # Generate insights
        self.generate_segment_insights()
        
        print("\n✅ Analysis completed successfully!")
        print("📁 Visualizations saved:")
        print("   - customer_segments_profile.png")
        print("   - customer_segments_scatter.png")
        print("   - customer_segments_heatmap.png")
        print("   - customer_type_distribution.png")

# Main execution
def main():
    """Main execution function"""
    analyzer = AdvancedCustomerSegmentation()
    analyzer.run_complete_analysis()
    return analyzer

if __name__ == "__main__":
    segmentation_analyzer = main()
