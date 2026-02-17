import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re
from textblob import TextBlob
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

class SentimentPriceElasticityIntegration:
    def __init__(self, cache_dir='./data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data attributes
        self.financial_data = None
        self.attendance_data = None
        self.booking_data = None
        self.utilisation_data = None
        self.cancellation_data = None
        self.nps_data = None
        self.qualitative_feedback = None
        self.integrated_sentiment_data = None
        self.elasticity_sentiment_matrix = None
        
        # Cache file paths
        self.cache_files = {
            'financial_data': self.cache_dir / 'financial_data.pkl',
            'attendance_data': self.cache_dir / 'attendance_data.pkl',
            'booking_data': self.cache_dir / 'booking_data.pkl',
            'utilisation_data': self.cache_dir / 'utilisation_data.pkl',
            'cancellation_data': self.cache_dir / 'cancellation_data.pkl',
            'nps_data': self.cache_dir / 'nps_data.pkl',
            'qualitative_feedback': self.cache_dir / 'qualitative_feedback.pkl',
            'preprocessed_complete': self.cache_dir / 'preprocessed_complete.pkl'
        }

    def save_data_to_cache(self, data, cache_key):
        """Save data to cache file"""
        try:
            with open(self.cache_files[cache_key], 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved {cache_key} to cache")
        except Exception as e:
            print(f"✗ Error saving {cache_key}: {e}")

    def load_data_from_cache(self, cache_key):
        """Load data from cache file"""
        try:
            if self.cache_files[cache_key].exists():
                with open(self.cache_files[cache_key], 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ Loaded {cache_key} from cache")
                return data
            return None
        except Exception as e:
            print(f"✗ Error loading {cache_key}: {e}")
            return None

    def save_all_preprocessed_data(self):
        """Save all preprocessed data as a complete package"""
        try:
            all_data = {
                'financial_data': self.financial_data,
                'attendance_data': self.attendance_data,
                'booking_data': self.booking_data,
                'utilisation_data': self.utilisation_data,
                'cancellation_data': self.cancellation_data,
                'nps_data': self.nps_data,
                'qualitative_feedback': self.qualitative_feedback,
                'preprocessing_timestamp': datetime.now()
            }
            
            with open(self.cache_files['preprocessed_complete'], 'wb') as f:
                pickle.dump(all_data, f)
            print("✓ All preprocessed data saved successfully!")
            
        except Exception as e:
            print(f"✗ Error saving complete preprocessed data: {e}")

    def load_all_preprocessed_data(self):
        """Load all preprocessed data if available"""
        try:
            if self.cache_files['preprocessed_complete'].exists():
                with open(self.cache_files['preprocessed_complete'], 'rb') as f:
                    all_data = pickle.load(f)
                
                # Load all data into class attributes
                self.financial_data = all_data['financial_data']
                self.attendance_data = all_data['attendance_data']
                self.booking_data = all_data['booking_data']
                self.utilisation_data = all_data['utilisation_data']
                self.cancellation_data = all_data['cancellation_data']
                self.nps_data = all_data['nps_data']
                self.qualitative_feedback = all_data['qualitative_feedback']
                
                timestamp = all_data.get('preprocessing_timestamp', 'Unknown')
                print(f"✓ All preprocessed data loaded successfully!")
                print(f"  Data preprocessed on: {timestamp}")
                return True
                
        except Exception as e:
            print(f"✗ Error loading preprocessed data: {e}")
            
        return False

    def check_cache_validity(self, force_reload=False):
        """Check if we should use cached data or reload from source"""
        if force_reload:
            print("🔄 Force reload requested - will preprocess from source files")
            return False
            
        if self.cache_files['preprocessed_complete'].exists():
            cache_time = datetime.fromtimestamp(self.cache_files['preprocessed_complete'].stat().st_mtime)
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            print(f"📊 Found cached data from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Cache age: {age_hours:.1f} hours")
            
            # You can set your own cache validity period here
            if age_hours < 24:  # Cache valid for 24 hours
                return True
            else:
                print("⚠️  Cache is older than 24 hours - will refresh")
                return False
        
        print("📁 No cached data found - will preprocess from source files")
        return False

    def load_and_preprocess_data(self, force_reload=False):
        """Load and preprocess all datasets with caching"""
        
        # Check if we can use cached data
        if not force_reload and self.check_cache_validity():
            if self.load_all_preprocessed_data():
                return
        
        print("\n🔄 Starting data preprocessing from source files...")
        
        try:
            # Load financial data
            print("📈 Loading financial data...")
            financial_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx',
                                            sheet_name=['S&F', 'Tiverton'])
            self.financial_data = pd.concat([financial_sheets['S&F'], financial_sheets['Tiverton']],
                                          ignore_index=True)
            
            # Load attendance data
            print("👥 Loading attendance data...")
            attendance_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx',
                                             sheet_name=['Gym', 'Gym 2','Reception Barrier 1','Reception Barrier 2',
                                                        'Recpetion Barrier 3', 'Reception Barrier 4'])
            self.attendance_data = pd.concat([attendance_sheets['Gym'],
                                            attendance_sheets['Gym 2'],
                                            attendance_sheets['Reception Barrier 1'],
                                            attendance_sheets['Reception Barrier 2'],
                                            attendance_sheets['Recpetion Barrier 3'],
                                            attendance_sheets['Reception Barrier 4']],
                                           ignore_index=True)

            # Load booking utilisation data
            print("📅 Loading booking data...")
            self.booking_data = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Booking_and_Utilisation_25_G24.xlsx')

            # Load utilisation data
            print("📊 Loading utilisation data...")
            util_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Utilisation_Data_25G24.xlsx',
                                       sheet_name=['2023 July to Dec', '2024 Jan - June','2024 July - Dec', '2025 Jan - May'])
            self.utilisation_data = pd.concat([util_sheets['2023 July to Dec'],
                                             util_sheets['2024 Jan - June'],
                                             util_sheets['2024 July - Dec'],
                                             util_sheets['2025 Jan - May']],
                                            ignore_index=True)

            # Load cancellation and NPS data
            print("❌ Loading cancellation data...")
            self.cancellation_data = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Cancellation_Updated_25_G24.xlsx')
            
            print("⭐ Loading NPS data...")
            self.nps_data = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')
            
            print("💬 Loading qualitative feedback...")
            self.qualitative_feedback = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')

            # Preprocess dates
            print("📅 Preprocessing dates...")
            self._preprocess_dates()
            
            # Save all preprocessed data
            print("💾 Saving preprocessed data to cache...")
            self.save_all_preprocessed_data()
            
            print("✅ Data loaded and preprocessed successfully!")
            
        except Exception as e:
            print(f"❌ Error during data loading: {e}")
            raise

    def _preprocess_dates(self):
        """Convert date columns to datetime"""
        try:
            # Financial data
            if 'Sales Detail Participation Date' in self.financial_data.columns:
                self.financial_data['Sales Detail Participation Date'] = pd.to_datetime(
                    self.financial_data['Sales Detail Participation Date'], errors='coerce')

            # Attendance data
            if 'Attendance Detail Date' in self.attendance_data.columns:
                self.attendance_data['Attendance Detail Date'] = pd.to_datetime(
                    self.attendance_data['Attendance Detail Date'], errors='coerce')

            # Booking data
            if 'BookedDate' in self.booking_data.columns:
                self.booking_data['BookedDate'] = pd.to_datetime(
                    self.booking_data['BookedDate'], errors='coerce')
            if 'StartDate' in self.booking_data.columns:
                self.booking_data['StartDate'] = pd.to_datetime(
                    self.booking_data['StartDate'], errors='coerce')

            # Utilisation data
            if 'Resource Utilisation Date' in self.utilisation_data.columns:
                self.utilisation_data['Resource Utilisation Date'] = pd.to_datetime(
                    self.utilisation_data['Resource Utilisation Date'], errors='coerce')

            # Cancellation data
            if 'Cancelled on' in self.cancellation_data.columns:
                self.cancellation_data['Cancelled on'] = pd.to_datetime(
                    self.cancellation_data['Cancelled on'], errors='coerce')

            # NPS data
            if 'Response Date' in self.nps_data.columns:
                self.nps_data['Response Date'] = pd.to_datetime(
                    self.nps_data['Response Date'], errors='coerce')
                
        except Exception as e:
            print(f"⚠️  Warning during date preprocessing: {e}")

    def clear_cache(self):
        """Clear all cached data"""
        try:
            for cache_file in self.cache_files.values():
                if cache_file.exists():
                    cache_file.unlink()
            print("🗑️  All cache files cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")

    def get_cache_info(self):
        """Get information about cached files"""
        print("\n📁 Cache Information:")
        print("-" * 50)
        
        total_size = 0
        for name, cache_file in self.cache_files.items():
            if cache_file.exists():
                size = cache_file.stat().st_size
                total_size += size
                modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
                print(f"✓ {name}: {size/1024/1024:.1f} MB (Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print(f"✗ {name}: Not cached")
        
        print(f"\nTotal cache size: {total_size/1024/1024:.1f} MB")

    def calculate_sentiment_scores(self):
        """Calculate enhanced sentiment scores from NPS and qualitative feedback"""
        # Process NPS data
        nps_sentiment = self.nps_data.copy()
        
        # Categorize NPS scores
        nps_sentiment['Sentiment_Category'] = nps_sentiment['Score'].apply(
            lambda x: 'Promoter' if x >= 9 else ('Passive' if x >= 7 else 'Detractor')
        )
        
        nps_sentiment['Sentiment_Score'] = nps_sentiment['Score'].apply(
            lambda x: (x - 6) / 4  # Normalize to -1 to 1 scale
        )

        # Process qualitative feedback
        qualitative_sentiment = self.qualitative_feedback.copy()
        
        # Extract sentiment from feedback text
        def analyze_text_sentiment(text):
            if pd.isna(text):
                return 0
            blob = TextBlob(str(text))
            return blob.sentiment.polarity

        qualitative_sentiment['Text_Sentiment'] = qualitative_sentiment['Feedback Received'].apply(
            analyze_text_sentiment
        )

        # Categorize feedback by service type
        def categorize_feedback(text):
            if pd.isna(text):
                return 'General'
            text = str(text).lower()
            if any(word in text for word in ['pool', 'swim', 'shower', 'changing']):
                return 'Pool_Services'
            elif any(word in text for word in ['gym', 'equipment', 'weights', 'cardio']):
                return 'Gym_Services'
            elif any(word in text for word in ['class', 'instructor', 'yoga', 'spin']):
                return 'Classes'
            elif any(word in text for word in ['staff', 'reception', 'service']):
                return 'Customer_Service'
            elif any(word in text for word in ['price', 'cost', 'fee', 'membership']):
                return 'Pricing'
            else:
                return 'General'

        qualitative_sentiment['Service_Category'] = qualitative_sentiment['Feedback Received'].apply(
            categorize_feedback
        )

        # Severity classification
        def classify_severity(text):
            if pd.isna(text):
                return 'Low'
            text = str(text).lower()
            if any(word in text for word in ['terrible', 'awful', 'disgusting', 'appalling', 'cancel']):
                return 'High'
            elif any(word in text for word in ['poor', 'bad', 'disappointing', 'frustrated']):
                return 'Medium'
            else:
                return 'Low'

        qualitative_sentiment['Severity'] = qualitative_sentiment['Feedback Received'].apply(
            classify_severity
        )

        return nps_sentiment, qualitative_sentiment

    def integrate_sentiment_with_usage_data(self):
        """Integrate sentiment data with usage and financial data"""
        nps_sentiment, qualitative_sentiment = self.calculate_sentiment_scores()

        # Calculate member value and usage frequency
        member_value = self.financial_data.groupby('Unique Key').agg({
            'Sales Detail Gross Amount': ['sum', 'mean', 'count'],
            'Sales Detail Price Level': 'first'
        }).reset_index()
        member_value.columns = ['Unique_Key', 'Total_Revenue', 'Avg_Revenue', 'Transaction_Count', 'Price_Level']

        # Calculate usage frequency
        usage_frequency = self.attendance_data.groupby('Unique Key').agg({
            'Attendance Detail Date': 'count',
            'Attendance Detail Date Time': ['min', 'max']
        }).reset_index()
        usage_frequency.columns = ['Unique_Key', 'Total_Visits', 'First_Visit', 'Last_Visit']

        # Calculate usage intensity (visits per month)
        usage_frequency['Usage_Duration_Days'] = (
            usage_frequency['Last_Visit'] - usage_frequency['First_Visit']
        ).dt.days
        usage_frequency['Usage_Intensity'] = np.where(
            usage_frequency['Usage_Duration_Days'] > 0,
            usage_frequency['Total_Visits'] / (usage_frequency['Usage_Duration_Days'] / 30),
            usage_frequency['Total_Visits']
        )

        # Merge sentiment with member data
        weighted_nps = nps_sentiment.merge(member_value, left_on='Unique ID', right_on='Unique_Key', how='left')
        weighted_nps = weighted_nps.merge(usage_frequency, on='Unique_Key', how='left')

        # Calculate weighted sentiment scores
        weighted_nps['Revenue_Weight'] = weighted_nps['Total_Revenue'] / weighted_nps['Total_Revenue'].max()
        weighted_nps['Usage_Weight'] = weighted_nps['Usage_Intensity'] / weighted_nps['Usage_Intensity'].max()
        weighted_nps['Combined_Weight'] = (weighted_nps['Revenue_Weight'].fillna(0) +
                                          weighted_nps['Usage_Weight'].fillna(0)) / 2
        weighted_nps['Weighted_Sentiment'] = (weighted_nps['Sentiment_Score'] *
                                             (1 + weighted_nps['Combined_Weight']))

        self.integrated_sentiment_data = weighted_nps

        return weighted_nps, qualitative_sentiment

    def calculate_price_elasticity_by_segment(self):
        """Calculate price elasticity for different member segments"""
        # Get member data with pricing and usage
        valid_data = self.financial_data.merge(
            self.attendance_data.groupby('Unique Key').size().reset_index(),
            left_on='Unique Key', right_on='Unique Key', how='inner'
        )
        
        valid_data.columns = list(valid_data.columns[:-1]) + ['Total_Visits']

        # Filter valid data
        valid_data = valid_data[
            (valid_data['Sales Detail Gross Amount'].notna()) &
            (valid_data['Sales Detail Gross Amount'] > 0) &
            (valid_data['Total_Visits'] > 0)
        ]

        # Calculate elasticity by membership tier
        elasticity_by_tier = {}
        for tier in valid_data['Sales Detail Price Level'].dropna().unique():
            tier_data = valid_data[valid_data['Sales Detail Price Level'] == tier]
            if len(tier_data) < 10:  # Minimum sample size
                continue

            # Simple elasticity calculation using price-quantity relationship
            price_demand_corr = tier_data[['Sales Detail Gross Amount', 'Total_Visits']].corr().iloc[0, 1]

            # Calculate percentage changes
            tier_data = tier_data.sort_values('Sales Detail Participation Date')
            tier_data['Price_Change'] = tier_data['Sales Detail Gross Amount'].pct_change()
            tier_data['Demand_Change'] = tier_data['Total_Visits'].pct_change()

            # Calculate elasticity where price changes exist
            valid_changes = tier_data[
                (tier_data['Price_Change'].notna()) &
                (tier_data['Demand_Change'].notna()) &
                (tier_data['Price_Change'] != 0)
            ]

            if len(valid_changes) > 0:
                elasticity = (valid_changes['Demand_Change'] / valid_changes['Price_Change']).mean()
            else:
                elasticity = price_demand_corr  # Use correlation as proxy

            elasticity_by_tier[tier] = {
                'elasticity': elasticity,
                'correlation': price_demand_corr,
                'sample_size': len(tier_data),
                'avg_price': tier_data['Sales Detail Gross Amount'].mean(),
                'avg_visits': tier_data['Total_Visits'].mean()
            }

        return elasticity_by_tier

    def create_sentiment_elasticity_matrix(self):
        """Create 2x2 matrix of sentiment vs elasticity"""
        weighted_sentiment, _ = self.integrate_sentiment_with_usage_data()
        elasticity_data = self.calculate_price_elasticity_by_segment()

        # Create matrix data
        matrix_data = []
        for tier, elasticity_info in elasticity_data.items():
            # Get sentiment for this tier
            tier_sentiment = weighted_sentiment[
                weighted_sentiment['Price_Level'] == tier
            ]['Weighted_Sentiment'].mean()

            if pd.isna(tier_sentiment):
                continue

            # Categorize sentiment and elasticity
            sentiment_cat = 'Positive' if tier_sentiment > 0 else 'Negative'
            elasticity_cat = 'Inelastic' if elasticity_info['elasticity'] > -0.5 else 'Elastic'

            matrix_data.append({
                'Tier': tier,
                'Sentiment_Category': sentiment_cat,
                'Elasticity_Category': elasticity_cat,
                'Sentiment_Score': tier_sentiment,
                'Elasticity_Value': elasticity_info['elasticity'],
                'Sample_Size': elasticity_info['sample_size'],
                'Avg_Price': elasticity_info['avg_price'],
                'Avg_Visits': elasticity_info['avg_visits']
            })

        matrix_df = pd.DataFrame(matrix_data)

        # Create strategy recommendations
        def get_strategy(row):
            if row['Sentiment_Category'] == 'Positive' and row['Elasticity_Category'] == 'Inelastic':
                return 'Premium Pricing Opportunities'
            elif row['Sentiment_Category'] == 'Positive' and row['Elasticity_Category'] == 'Elastic':
                return 'Volume-Based Pricing'
            elif row['Sentiment_Category'] == 'Negative' and row['Elasticity_Category'] == 'Inelastic':
                return 'Service Improvement Priority'
            else:
                return 'Churn Risk - Careful Pricing'

        matrix_df['Strategy_Recommendation'] = matrix_df.apply(get_strategy, axis=1)

        self.elasticity_sentiment_matrix = matrix_df

        return matrix_df

    def analyze_peak_hours_pricing(self):
        """Analyze optimal pricing windows using utilisation data"""
        # Process utilisation data for peak analysis
        utilisation_analysis = self.utilisation_data.copy()

        # Calculate utilisation by hour and day
        hourly_utilisation = utilisation_analysis.groupby(
            ['Resource Utilisation Resource', 'Resource Utilisation Hour', 'Resource Utilisation Weekday']
        ).agg({
            'Resource Utilisation Hours Used': 'sum',
            'Resource Utilisation Hours Available': 'sum'
        }).reset_index()

        hourly_utilisation['Utilisation_Rate'] = (
            hourly_utilisation['Resource Utilisation Hours Used'] /
            hourly_utilisation['Resource Utilisation Hours Available']
        )

        # Identify peak hours (top 25% utilisation)
        peak_threshold = hourly_utilisation['Utilisation_Rate'].quantile(0.75)
        hourly_utilisation['Is_Peak'] = hourly_utilisation['Utilisation_Rate'] >= peak_threshold

        # Calculate potential premium pricing
        hourly_utilisation['Pricing_Multiplier'] = np.where(
            hourly_utilisation['Is_Peak'],
            1 + (hourly_utilisation['Utilisation_Rate'] - peak_threshold) * 2,
            1.0
        )

        return hourly_utilisation

    def analyze_capacity_management(self):
        """Analyze capacity management and demand shifting opportunities"""
        # Combine booking and attendance data
        booking_analysis = self.booking_data.copy()

        # Extract hour from booking time
        booking_analysis['Booking_Hour'] = pd.to_datetime(
            booking_analysis['StartDateTime']
        ).dt.hour

        booking_analysis['Booking_Weekday'] = pd.to_datetime(
            booking_analysis['StartDate']
        ).dt.day_name()

        # Calculate no-show rates
        booking_analysis['No_Show'] = booking_analysis['Attended'] == 'No'

        # Analyze by time slot
        capacity_analysis = booking_analysis.groupby(
            ['Activity', 'Booking_Hour', 'Booking_Weekday']
        ).agg({
            'BookingID': 'count',
            'No_Show': 'sum'
        }).reset_index()

        capacity_analysis['No_Show_Rate'] = (
            capacity_analysis['No_Show'] / capacity_analysis['BookingID']
        )

        capacity_analysis['Actual_Demand'] = (
            capacity_analysis['BookingID'] * (1 - capacity_analysis['No_Show_Rate'])
        )

        return capacity_analysis

    def identify_at_risk_members(self):
        """Identify at-risk members using cancellation patterns and declining usage"""
        
        # Analyze cancellation patterns
        if self.cancellation_data is not None and not self.cancellation_data.empty:
            cancellation_patterns = self.cancellation_data.groupby('Reasoning').size().reset_index()
            cancellation_patterns.columns = ['Cancellation_Reason', 'Count']
            cancellation_patterns['Percentage'] = (
                cancellation_patterns['Count'] / cancellation_patterns['Count'].sum() * 100
            )
        else:
            # Create empty DataFrame with proper structure if no cancellation data
            cancellation_patterns = pd.DataFrame({
                'Cancellation_Reason': [],
                'Count': [],
                'Percentage': []
            })

        # Analyze declining usage patterns
        recent_attendance = self.attendance_data[
            self.attendance_data['Attendance Detail Date'] >= 
            (datetime.now() - timedelta(days=90))
        ]
        
        print(f"Recent attendance records: {len(recent_attendance)}")
        
        # Calculate usage trends
        member_trends = []
        unique_members = recent_attendance['Unique Key'].unique()
        print(f"Unique members in recent attendance: {len(unique_members)}")
        
        for member in unique_members:
            member_data = recent_attendance[recent_attendance['Unique Key'] == member]
            member_data = member_data.sort_values('Attendance Detail Date')
            
            # Calculate monthly visits with more flexible grouping
            member_data['Month'] = member_data['Attendance Detail Date'].dt.to_period('M')
            monthly_visits = member_data.groupby('Month').size()
            
            print(f"Member {member}: {len(monthly_visits)} months of data")
            
            if len(monthly_visits) >= 2:  # Need at least 2 months for trend
                try:
                    # Calculate trend (slope of visits over time)
                    x_values = range(len(monthly_visits))
                    y_values = monthly_visits.values
                    
                    # Use numpy polyfit to calculate trend
                    if len(x_values) > 1:  # Additional safety check
                        trend = np.polyfit(x_values, y_values, 1)[0]
                    else:
                        trend = 0
                        
                    member_trends.append({
                        'Unique_Key': member,
                        'Usage_Trend': trend,
                        'Recent_Visits': monthly_visits.iloc[-1] if len(monthly_visits) > 0 else 0,
                        'Avg_Monthly_Visits': monthly_visits.mean(),
                        'Months_Data': len(monthly_visits)
                    })
                except Exception as e:
                    print(f"Error calculating trend for member {member}: {e}")
                    # Add member with neutral trend if calculation fails
                    member_trends.append({
                        'Unique_Key': member,
                        'Usage_Trend': 0,
                        'Recent_Visits': len(member_data),
                        'Avg_Monthly_Visits': len(member_data) / max(1, len(monthly_visits)),
                        'Months_Data': len(monthly_visits)
                    })
        
        print(f"Total member trends calculated: {len(member_trends)}")
        
        # Create DataFrame from trends
        if member_trends:
            trends_df = pd.DataFrame(member_trends)
            print(f"Trends DataFrame shape: {trends_df.shape}")
            print(f"Usage_Trend column stats:\n{trends_df['Usage_Trend'].describe()}")
            
            # Identify at-risk members (declining usage)
            if len(trends_df) > 0 and 'Usage_Trend' in trends_df.columns:
                at_risk_threshold = trends_df['Usage_Trend'].quantile(0.25)
                trends_df['At_Risk'] = trends_df['Usage_Trend'] <= at_risk_threshold
                print(f"At-risk threshold: {at_risk_threshold}")
                print(f"At-risk members: {trends_df['At_Risk'].sum()}")
            else:
                # Handle case where no trends calculated
                trends_df['At_Risk'] = False
                print("No usage trends calculated - marking all members as not at risk")
        else:
            # Create empty DataFrame with proper structure
            trends_df = pd.DataFrame({
                'Unique_Key': [],
                'Usage_Trend': [],
                'Recent_Visits': [],
                'Avg_Monthly_Visits': [],
                'Months_Data': [],
                'At_Risk': []
            })
            print("No member trends data available - created empty DataFrame")

        return trends_df, cancellation_patterns

    # VISUALIZATION FUNCTIONS
    def create_sentiment_elasticity_matrix_chart(self, sentiment_elasticity_matrix):
        """Create 2x2 Sentiment vs Elasticity Matrix with Strategic Insights"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create scatter plot with enhanced styling
        sentiment_colors = {'Positive': '#2E8B57', 'Negative': '#DC143C'}  # Sea Green and Crimson
        elasticity_shapes = {'Elastic': 'o', 'Inelastic': 's'}
        
        for i, row in sentiment_elasticity_matrix.iterrows():
            ax.scatter(row['Elasticity_Value'], row['Sentiment_Score'],
                      s=row['Sample_Size']*3,  # Size by sample size
                      c=sentiment_colors[row['Sentiment_Category']],
                      marker=elasticity_shapes[row['Elasticity_Category']],
                      alpha=0.8,
                      edgecolors='black',
                      linewidth=2,
                      label=f"{row['Tier']} ({row['Strategy_Recommendation']})")
            
            # Add tier labels with strategy
            ax.annotate(f"{row['Tier']}\n{row['Strategy_Recommendation'][:15]}...",
                       (row['Elasticity_Value'], row['Sentiment_Score']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add quadrant lines and labels
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=-0.5, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # Enhanced quadrant labels with strategic insights
        quadrant_insights = {
            'top_right': 'PREMIUM PRICING\nHigh satisfaction + Price insensitive\n→ Increase prices strategically',
            'top_left': 'VOLUME STRATEGY\nHigh satisfaction + Price sensitive\n→ Focus on retention & volume',
            'bottom_right': 'SERVICE PRIORITY\nLow satisfaction + Price insensitive\n→ Improve service quality',
            'bottom_left': 'CHURN RISK\nLow satisfaction + Price sensitive\n→ Immediate intervention needed'
        }
        
        # Position quadrant labels
        ax.text(0.3, 0.8, quadrant_insights['top_right'], transform=ax.transAxes, 
                fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#90EE90", alpha=0.8))
        
        ax.text(0.15, 0.8, quadrant_insights['top_left'], transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#87CEEB", alpha=0.8))
        
        ax.text(0.3, 0.15, quadrant_insights['bottom_right'], transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFB347", alpha=0.8))
        
        ax.text(0.15, 0.15, quadrant_insights['bottom_left'], transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F08080", alpha=0.8))
        
        # Styling
        ax.set_xlabel('Price Elasticity (More Negative = More Elastic)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weighted Sentiment Score', fontsize=14, fontweight='bold')
        ax.set_title('Strategic Sentiment-Price Elasticity Matrix\nMembership Tier Positioning & Pricing Strategy', 
                     fontsize=18, fontweight='bold', pad=20)
        
        # Add legend for shapes and colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57', markersize=12, 
                      label='Positive Sentiment', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', markersize=12, 
                      label='Negative Sentiment', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
                      label='Elastic (Price Sensitive)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, 
                      label='Inelastic (Price Insensitive)', markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print strategic insights
        print("\n" + "="*80)
        print("STRATEGIC INSIGHTS FROM SENTIMENT-ELASTICITY MATRIX")
        print("="*80)
        
        for _, row in sentiment_elasticity_matrix.iterrows():
            print(f"\n{row['Tier']} Membership:")
            print(f"  Strategy: {row['Strategy_Recommendation']}")
            print(f"  Sentiment Score: {row['Sentiment_Score']:.2f}")
            print(f"  Price Elasticity: {row['Elasticity_Value']:.2f}")
            print(f"  Average Price: £{row['Avg_Price']:.2f}")
            print(f"  Average Visits: {row['Avg_Visits']:.1f}")
            print(f"  Sample Size: {row['Sample_Size']} members")

    def create_peak_pricing_heatmap(self, peak_hours_analysis):
        """Create facility utilisation and pricing heatmaps"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
        
        # Utilisation rate heatmap
        try:
            pivot_utilisation = peak_hours_analysis.pivot_table(
                values='Utilisation_Rate',
                index='Resource Utilisation Hour',
                columns='Resource Utilisation Weekday',
                aggfunc='mean'
            )
            
            # Reorder columns for logical day sequence
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_utilisation = pivot_utilisation.reindex(columns=[d for d in day_order if d in pivot_utilisation.columns])
            
            sns.heatmap(pivot_utilisation, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       ax=ax1, cbar_kws={'label': 'Utilisation Rate (%)'})
            ax1.set_title('Facility Utilisation Rate by Hour and Day\n(Higher values = Peak times)', 
                         fontweight='bold', fontsize=14)
            ax1.set_ylabel('Hour of Day', fontweight='bold')
            ax1.set_xlabel('')
        except Exception as e:
            print(f"Could not create utilisation heatmap: {e}")
            ax1.text(0.5, 0.5, 'Utilisation data not available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16)
            ax1.set_title('Facility Utilisation Rate (Data Not Available)', fontweight='bold')
        
        # Pricing multiplier heatmap
        try:
            pivot_pricing = peak_hours_analysis.pivot_table(
                values='Pricing_Multiplier',
                index='Resource Utilisation Hour',
                columns='Resource Utilisation Weekday',
                aggfunc='mean'
            )
            
            # Reorder columns for logical day sequence
            pivot_pricing = pivot_pricing.reindex(columns=[d for d in day_order if d in pivot_pricing.columns])
            
            sns.heatmap(pivot_pricing, annot=True, fmt='.2f', cmap='RdYlGn',
                       ax=ax2, cbar_kws={'label': 'Pricing Multiplier'})
            ax2.set_title('Recommended Dynamic Pricing Multipliers\n(Values > 1.0 = Premium pricing opportunities)', 
                         fontweight='bold', fontsize=14)
            ax2.set_ylabel('Hour of Day', fontweight='bold')
            ax2.set_xlabel('Day of Week', fontweight='bold')
        except Exception as e:
            print(f"Could not create pricing heatmap: {e}")
            ax2.text(0.5, 0.5, 'Pricing data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16)
            ax2.set_title('Dynamic Pricing Multipliers (Data Not Available)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print pricing insights
        if not peak_hours_analysis.empty:
            print("\n" + "="*60)
            print("DYNAMIC PRICING INSIGHTS")
            print("="*60)
            
            # Find peak hours
            peak_times = peak_hours_analysis[peak_hours_analysis['Is_Peak'] == True]
            if not peak_times.empty:
                avg_peak_utilisation = peak_times['Utilisation_Rate'].mean()
                avg_pricing_multiplier = peak_times['Pricing_Multiplier'].mean()
                
                print(f"Peak Hours Identified: {len(peak_times)} time slots")
                print(f"Average Peak Utilisation: {avg_peak_utilisation:.1%}")
                print(f"Recommended Peak Pricing Multiplier: {avg_pricing_multiplier:.2f}x")
                print(f"Potential Revenue Increase: {(avg_pricing_multiplier-1)*100:.1f}% during peak hours")

    def create_capacity_utilization_chart(self, capacity_analysis):
        """Create comprehensive capacity utilization analysis"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Bookings by hour
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            hourly_bookings = capacity_analysis.groupby('Booking_Hour')['BookingID'].sum().sort_index()
            bars1 = ax1.bar(hourly_bookings.index, hourly_bookings.values,
                           color=plt.cm.viridis(np.linspace(0, 1, len(hourly_bookings))), alpha=0.8)
            ax1.set_xlabel('Hour of Day', fontweight='bold')
            ax1.set_ylabel('Total Bookings', fontweight='bold')
            ax1.set_title('Booking Distribution by Hour\nIdentifying Peak Demand Times', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, hourly_bookings.values):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hourly_bookings)*0.01,
                            f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        except Exception as e:
            print(f"Error creating hourly bookings chart: {e}")
            ax1.text(0.5, 0.5, 'Booking data not available', ha='center', va='center', transform=ax1.transAxes)
        
        # 2. No-show rates by day
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            daily_noshows = capacity_analysis.groupby('Booking_Weekday')['No_Show_Rate'].mean()
            colors = ['#FF6B6B' if rate > 0.2 else '#4ECDC4' if rate < 0.1 else '#45B7D1' for rate in daily_noshows.values]
            bars2 = ax2.bar(daily_noshows.index, daily_noshows.values * 100, color=colors, alpha=0.8)
            ax2.set_ylabel('No-Show Rate (%)', fontweight='bold')
            ax2.set_title('No-Show Rate by Day\nRevenue Impact Analysis', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels and color coding explanation
            for bar, value in zip(bars2, daily_noshows.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
                
        except Exception as e:
            print(f"Error creating no-show chart: {e}")
            ax2.text(0.5, 0.5, 'No-show data not available', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Activity popularity
        ax3 = fig.add_subplot(gs[1, 0])
        try:
            activity_bookings = capacity_analysis.groupby('Activity')['BookingID'].sum().sort_values(ascending=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(activity_bookings)))
            bars3 = ax3.barh(range(len(activity_bookings)), activity_bookings.values, color=colors, alpha=0.8)
            ax3.set_yticks(range(len(activity_bookings)))
            ax3.set_yticklabels(activity_bookings.index, fontsize=10)
            ax3.set_xlabel('Total Bookings', fontweight='bold')
            ax3.set_title('Activity Popularity Ranking\nResource Allocation Insights', fontweight='bold')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars3, activity_bookings.values)):
                ax3.text(value + max(activity_bookings)*0.01, i,
                        f'{int(value)}', va='center', fontweight='bold')
                        
        except Exception as e:
            print(f"Error creating activity popularity chart: {e}")
            ax3.text(0.5, 0.5, 'Activity data not available', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Actual vs booked demand
        ax4 = fig.add_subplot(gs[1, 1])
        try:
            demand_comparison = capacity_analysis.groupby('Activity').agg({
                'BookingID': 'sum',
                'Actual_Demand': 'sum'
            }).reset_index()
            
            x = np.arange(len(demand_comparison))
            width = 0.35
            bars4a = ax4.bar(x - width/2, demand_comparison['BookingID'], width,
                            label='Booked Demand', color='#FF9999', alpha=0.8)
            bars4b = ax4.bar(x + width/2, demand_comparison['Actual_Demand'], width,
                            label='Actual Demand', color='#66B2FF', alpha=0.8)
            
            ax4.set_xlabel('Activity', fontweight='bold')
            ax4.set_ylabel('Demand', fontweight='bold')
            ax4.set_title('Booked vs Actual Demand by Activity\nCapacity Planning Insights', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(demand_comparison['Activity'], rotation=45, ha='right')
            ax4.legend(fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating demand comparison chart: {e}")
            ax4.text(0.5, 0.5, 'Demand comparison data not available', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Capacity efficiency metrics (bottom span)
        ax5 = fig.add_subplot(gs[2, :])
        try:
            # Calculate efficiency metrics
            efficiency_data = capacity_analysis.groupby('Activity').agg({
                'BookingID': 'sum',
                'No_Show_Rate': 'mean',
                'Actual_Demand': 'sum'
            }).reset_index()
            efficiency_data['Efficiency_Score'] = (1 - efficiency_data['No_Show_Rate']) * 100
            efficiency_data = efficiency_data.sort_values('Efficiency_Score', ascending=True)
            
            # Create horizontal bar chart with efficiency scores
            colors = ['#FF4444' if score < 70 else '#FFB347' if score < 85 else '#90EE90' for score in efficiency_data['Efficiency_Score']]
            bars5 = ax5.barh(range(len(efficiency_data)), efficiency_data['Efficiency_Score'], color=colors, alpha=0.8)
            
            ax5.set_yticks(range(len(efficiency_data)))
            ax5.set_yticklabels(efficiency_data['Activity'])
            ax5.set_xlabel('Efficiency Score (% of bookings that show up)', fontweight='bold')
            ax5.set_title('Activity Efficiency Ranking\nCapacity Optimization Opportunities', fontweight='bold', pad=20)
            
            # Add efficiency score labels and interpretation
            for i, (bar, score, bookings) in enumerate(zip(bars5, efficiency_data['Efficiency_Score'], efficiency_data['BookingID'])):
                ax5.text(score + 1, i, f'{score:.1f}% ({int(bookings)} bookings)', 
                        va='center', fontweight='bold')
            
            # Add efficiency zones
            ax5.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='Poor Efficiency (<70%)')
            ax5.axvline(x=85, color='orange', linestyle='--', alpha=0.7, label='Good Efficiency (70-85%)')
            ax5.axvline(x=85, color='green', linestyle='--', alpha=0.7)
            ax5.text(95, len(efficiency_data)*0.9, 'Excellent\n(>85%)', ha='center', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax5.text(77.5, len(efficiency_data)*0.9, 'Good\n(70-85%)', ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            ax5.text(50, len(efficiency_data)*0.9, 'Poor\n(<70%)', ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            
        except Exception as e:
            print(f"Error creating efficiency chart: {e}")
            ax5.text(0.5, 0.5, 'Efficiency data not available', ha='center', va='center', transform=ax5.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Print capacity insights
        print("\n" + "="*70)
        print("CAPACITY MANAGEMENT INSIGHTS")
        print("="*70)
        
        try:
            if not capacity_analysis.empty:
                total_bookings = capacity_analysis['BookingID'].sum()
                avg_no_show_rate = capacity_analysis['No_Show_Rate'].mean()
                total_actual_demand = capacity_analysis['Actual_Demand'].sum()
                
                print(f"Total Bookings: {total_bookings:,}")
                print(f"Average No-Show Rate: {avg_no_show_rate:.1%}")
                print(f"Actual Demand: {total_actual_demand:,.1f}")
                print(f"Lost Revenue from No-Shows: {(total_bookings - total_actual_demand):,.1f} booking slots")
                print(f"Capacity Efficiency: {(total_actual_demand/total_bookings):.1%}")
        except Exception as e:
            print(f"Could not calculate capacity metrics: {e}")

    def create_at_risk_analysis_chart(self, at_risk_analysis, cancellation_patterns):
        """Create comprehensive at-risk member analysis"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Usage trend distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if not at_risk_analysis.empty and 'Usage_Trend' in at_risk_analysis.columns:
            ax1.hist(at_risk_analysis['Usage_Trend'], bins=30, alpha=0.7,
                    color='skyblue', edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
            
            if at_risk_analysis['Usage_Trend'].quantile(0.25) is not None:
                threshold = at_risk_analysis['Usage_Trend'].quantile(0.25)
                ax1.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label='At-Risk Threshold')
                
                # Add annotations
                ax1.text(threshold, ax1.get_ylim()[1]*0.8, f'At-Risk\nThreshold\n({threshold:.2f})', 
                        ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            
            ax1.set_xlabel('Usage Trend (visits/month change)', fontweight='bold')
            ax1.set_ylabel('Number of Members', fontweight='bold')
            ax1.set_title('Member Usage Trend Distribution\nEarly Warning System', fontweight='bold')
            ax1.legend(fontweight='bold')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Usage trend data not available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Usage Trend Distribution (Data Not Available)', fontweight='bold')
        
        # 2. At-risk vs normal members pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        if not at_risk_analysis.empty and 'At_Risk' in at_risk_analysis.columns:
            risk_summary = at_risk_analysis['At_Risk'].value_counts()
            risk_labels = ['Normal Members', 'At-Risk Members']
            colors = ['#90EE90', '#FF6B6B']
            sizes = [risk_summary.get(False, 0), risk_summary.get(True, 0)]
            
            # Only create pie chart if we have data
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax2.pie(sizes, labels=risk_labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90, explode=(0, 0.1))
                
                # Enhance text formatting
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                ax2.set_title('At-Risk Member Distribution\nChurn Prevention Target', fontweight='bold')
                
                # Add total count
                total_members = sum(sizes)
                ax2.text(0, -1.3, f'Total Members Analyzed: {total_members:,}', ha='center', 
                        fontweight='bold', fontsize=12, transform=ax2.transData)
            else:
                ax2.text(0.5, 0.5, 'No at-risk data available', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'At-risk classification not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('At-Risk Distribution (Data Not Available)', fontweight='bold')
        
        # 3. Usage comparison
        ax3 = fig.add_subplot(gs[1, 0])
        if not at_risk_analysis.empty and 'At_Risk' in at_risk_analysis.columns:
            try:
                usage_comparison = at_risk_analysis.groupby('At_Risk').agg({
                    'Recent_Visits': 'mean',
                    'Avg_Monthly_Visits': 'mean'
                }).reset_index()
                
                x = np.arange(len(usage_comparison))
                width = 0.35
                bars3a = ax3.bar(x - width/2, usage_comparison['Recent_Visits'], width,
                                label='Recent Month Visits', color='#87CEEB', alpha=0.8)
                bars3b = ax3.bar(x + width/2, usage_comparison['Avg_Monthly_Visits'], width,
                                label='Historical Average', color='#4169E1', alpha=0.8)
                
                # Add value labels
                for bars in [bars3a, bars3b]:
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
                
                ax3.set_xlabel('Member Risk Status', fontweight='bold')
                ax3.set_ylabel('Average Visits', fontweight='bold')
                ax3.set_title('Usage Patterns: At-Risk vs Normal Members\nBehavioral Differences', fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(['Normal', 'At-Risk'])
                ax3.legend(fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
            except Exception as e:
                ax3.text(0.5, 0.5, f'Usage comparison error: {str(e)}', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'Usage comparison data not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Usage Comparison (Data Not Available)', fontweight='bold')
        
        # 4. Cancellation reasons analysis
        ax4 = fig.add_subplot(gs[1, 1])
        if not cancellation_patterns.empty:
            top_reasons = cancellation_patterns.head(8).sort_values('Count', ascending=True)
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_reasons)))
            bars4 = ax4.barh(range(len(top_reasons)), top_reasons['Count'], color=colors, alpha=0.8)
            
            ax4.set_yticks(range(len(top_reasons)))
            ax4.set_yticklabels([reason[:20] + '...' if len(reason) > 20 else reason 
                               for reason in top_reasons['Cancellation_Reason']], fontsize=10)
            ax4.set_xlabel('Number of Cancellations', fontweight='bold')
            ax4.set_title('Top Cancellation Reasons\nChurn Pattern Analysis', fontweight='bold')
            
            # Add value labels and percentages
            total_cancellations = cancellation_patterns['Count'].sum()
            for i, (bar, count) in enumerate(zip(bars4, top_reasons['Count'])):
                percentage = (count / total_cancellations) * 100
                ax4.text(count + max(top_reasons['Count'])*0.01, i, 
                        f'{count} ({percentage:.1f}%)', va='center', fontweight='bold')
            
        else:
            ax4.text(0.5, 0.5, 'Cancellation data not available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Cancellation Reasons (Data Not Available)', fontweight='bold')
        
        # 5. Intervention priority matrix (bottom span)
        ax5 = fig.add_subplot(gs[2, :])
        if not at_risk_analysis.empty and 'At_Risk' in at_risk_analysis.columns:
            try:
                at_risk_members = at_risk_analysis[at_risk_analysis['At_Risk'] == True].copy()
                
                if not at_risk_members.empty:
                    # Create intervention priority score
                    at_risk_members['Intervention_Score'] = (
                        abs(at_risk_members['Usage_Trend']) * 2 +  # Higher decline = higher priority
                        (at_risk_members['Avg_Monthly_Visits'] / at_risk_members['Avg_Monthly_Visits'].max()) * 3  # Higher historical usage = higher value
                    )
                    
                    # Create scatter plot
                    scatter = ax5.scatter(at_risk_members['Usage_Trend'], 
                                        at_risk_members['Avg_Monthly_Visits'],
                                        s=at_risk_members['Intervention_Score'] * 50,
                                        c=at_risk_members['Intervention_Score'],
                                        cmap='Reds', alpha=0.6, edgecolors='black')
                    
                    ax5.set_xlabel('Usage Trend (decline rate)', fontweight='bold')
                    ax5.set_ylabel('Historical Average Monthly Visits', fontweight='bold')
                    ax5.set_title('At-Risk Member Intervention Priority Matrix\n(Bubble size & color = Priority score)', 
                                 fontweight='bold', pad=20)
                    ax5.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax5)
                    cbar.set_label('Intervention Priority Score', fontweight='bold')
                    
                    # Add quadrant labels
                    ax5.text(0.02, 0.95, 'HIGH PRIORITY\nHigh value + Steep decline', 
                            transform=ax5.transAxes, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                    ax5.text(0.02, 0.05, 'MEDIUM PRIORITY\nLow value + Steep decline', 
                            transform=ax5.transAxes, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
                    
                    # Print top priority members
                    top_priority = at_risk_members.nlargest(5, 'Intervention_Score')
                    print(f"\n🚨 TOP 5 INTERVENTION PRIORITIES:")
                    for _, member in top_priority.iterrows():
                        print(f"   Member {member['Unique_Key']}: Score {member['Intervention_Score']:.1f}, "
                              f"Trend {member['Usage_Trend']:.2f}, Avg Visits {member['Avg_Monthly_Visits']:.1f}")
                        
                else:
                    ax5.text(0.5, 0.5, 'No at-risk members identified', ha='center', va='center', 
                            transform=ax5.transAxes, fontsize=14)
                    
            except Exception as e:
                ax5.text(0.5, 0.5, f'Intervention analysis error: {str(e)}', ha='center', va='center', 
                        transform=ax5.transAxes, fontsize=12)
        else:
            ax5.text(0.5, 0.5, 'Intervention priority data not available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Intervention Priority Matrix (Data Not Available)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed insights
        print("\n" + "="*70)
        print("AT-RISK MEMBER ANALYSIS INSIGHTS")
        print("="*70)
        
        if not at_risk_analysis.empty:
            total_members = len(at_risk_analysis)
            at_risk_count = at_risk_analysis['At_Risk'].sum() if 'At_Risk' in at_risk_analysis.columns else 0
            at_risk_percentage = (at_risk_count / total_members) * 100 if total_members > 0 else 0
            
            print(f"Total Members Analyzed: {total_members:,}")
            print(f"At-Risk Members: {at_risk_count:,} ({at_risk_percentage:.1f}%)")
            
            if 'Usage_Trend' in at_risk_analysis.columns:
                avg_trend = at_risk_analysis['Usage_Trend'].mean()
                declining_members = (at_risk_analysis['Usage_Trend'] < 0).sum()
                print(f"Members with Declining Usage: {declining_members:,}")
                print(f"Average Usage Trend: {avg_trend:.2f} visits/month change")

    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all key insights"""
        print("\n" + "="*100)
        print("COMPREHENSIVE STRATEGIC DASHBOARD")
        print("="*100)
        
        try:
            # Run all analyses
            weighted_sentiment, qualitative_sentiment = self.integrate_sentiment_with_usage_data()
            sentiment_elasticity_matrix = self.create_sentiment_elasticity_matrix()
            peak_hours_analysis = self.analyze_peak_hours_pricing()
            capacity_analysis = self.analyze_capacity_management()
            at_risk_analysis, cancellation_patterns = self.identify_at_risk_members()
            
            # Create all visualizations separately
            print("\n📊 Creating Sentiment-Elasticity Strategic Matrix...")
            self.create_sentiment_elasticity_matrix_chart(sentiment_elasticity_matrix)
            
            print("\n📈 Creating Peak Hours & Dynamic Pricing Analysis...")
            self.create_peak_pricing_heatmap(peak_hours_analysis)
            
            print("\n🏢 Creating Capacity Management Dashboard...")
            self.create_capacity_utilization_chart(capacity_analysis)
            
            print("\n⚠️  Creating At-Risk Member Analysis...")
            self.create_at_risk_analysis_chart(at_risk_analysis, cancellation_patterns)
            
            # Summary insights
            self._print_executive_summary(sentiment_elasticity_matrix, peak_hours_analysis, 
                                        capacity_analysis, at_risk_analysis, cancellation_patterns)
            
        except Exception as e:
            print(f"❌ Error creating comprehensive dashboard: {e}")
            import traceback
            traceback.print_exc()

    def _print_executive_summary(self, sentiment_elasticity_matrix, peak_hours_analysis, 
                                capacity_analysis, at_risk_analysis, cancellation_patterns):
        """Print executive summary with key strategic insights"""
        print("\n" + "="*100)
        print("🎯 EXECUTIVE SUMMARY - KEY STRATEGIC INSIGHTS")
        print("="*100)
        
        # 1. Sentiment-Elasticity Insights
        print("\n1. PRICING STRATEGY INSIGHTS:")
        print("-" * 40)
        if not sentiment_elasticity_matrix.empty:
            premium_opportunities = sentiment_elasticity_matrix[
                sentiment_elasticity_matrix['Strategy_Recommendation'] == 'Premium Pricing Opportunities'
            ]
            if not premium_opportunities.empty:
                print(f"   ✅ PREMIUM PRICING OPPORTUNITIES: {len(premium_opportunities)} membership tiers")
                for _, tier in premium_opportunities.iterrows():
                    print(f"      → {tier['Tier']}: High satisfaction + Price insensitive")
                    print(f"        Potential price increase: 10-20% (Current avg: £{tier['Avg_Price']:.0f})")
            
            at_risk_segments = sentiment_elasticity_matrix[
                sentiment_elasticity_matrix['Strategy_Recommendation'] == 'Churn Risk - Careful Pricing'
            ]
            if not at_risk_segments.empty:
                print(f"   ⚠️  CHURN RISK SEGMENTS: {len(at_risk_segments)} membership tiers need attention")
        
        # 2. Capacity & Revenue Optimization
        print("\n2. CAPACITY & REVENUE OPTIMIZATION:")
        print("-" * 45)
        if not capacity_analysis.empty:
            total_bookings = capacity_analysis['BookingID'].sum()
            avg_no_show = capacity_analysis['No_Show_Rate'].mean()
            lost_revenue_slots = total_bookings * avg_no_show
            
            print(f"   📊 Total Bookings: {total_bookings:,}")
            print(f"   ❌ Average No-Show Rate: {avg_no_show:.1%}")
            print(f"   💰 Lost Revenue Opportunity: {lost_revenue_slots:,.0f} booking slots")
            print(f"   🎯 Recommendation: Implement overbooking strategy (105-110% of capacity)")
        
        # 3. Member Retention Insights
        print("\n3. MEMBER RETENTION INSIGHTS:")
        print("-" * 35)
        if not at_risk_analysis.empty:
            total_analyzed = len(at_risk_analysis)
            at_risk_count = at_risk_analysis['At_Risk'].sum() if 'At_Risk' in at_risk_analysis.columns else 0
            
            print(f"   👥 Members Analyzed: {total_analyzed:,}")
            print(f"   🚨 At-Risk Members: {at_risk_count:,} ({(at_risk_count/total_analyzed)*100:.1f}%)")
            print(f"   📈 Immediate Action Required: Top {min(10, at_risk_count)} members need intervention")
        
        if not cancellation_patterns.empty:
            top_reason = cancellation_patterns.iloc[0]
            print(f"   📊 Top Cancellation Reason: {top_reason['Cancellation_Reason']} ({top_reason['Percentage']:.1f}%)")
        
        # 4. Dynamic Pricing Opportunities
        print("\n4. DYNAMIC PRICING OPPORTUNITIES:")
        print("-" * 40)
        if not peak_hours_analysis.empty:
            peak_times = peak_hours_analysis[peak_hours_analysis['Is_Peak'] == True]
            if not peak_times.empty:
                avg_multiplier = peak_times['Pricing_Multiplier'].mean()
                potential_increase = (avg_multiplier - 1) * 100
                print(f"   ⏰ Peak Hours Identified: {len(peak_times)} time slots")
                print(f"   💵 Recommended Price Premium: {potential_increase:.1f}% during peak times")
                print(f"   📈 Estimated Revenue Increase: {potential_increase/2:.1f}% overall")
        
        # 5. Strategic Recommendations
        print("\n5. 🎯 TOP STRATEGIC RECOMMENDATIONS:")
        print("-" * 45)
        print("   1. Implement tiered dynamic pricing during peak hours")
        print("   2. Focus retention efforts on identified at-risk members")
        print("   3. Optimize capacity with strategic overbooking")
        print("   4. Develop premium service packages for price-insensitive segments")
        print("   5. Address top cancellation reasons through targeted improvements")
        
        print("\n" + "="*100)
        print("Dashboard Complete - All visualizations generated separately above")
        print("="*100)


# Enhanced usage example with comprehensive visualizations
def main():
    print("🚀 Starting Advanced Sentiment-Price Elasticity Analysis with Professional Visualizations")
    print("=" * 100)
    
    # Initialize analyzer
    analyzer = SentimentPriceElasticityIntegration()
    
    # Show cache information
    analyzer.get_cache_info()
    
    # Load data (will use cache if available and valid)
    analyzer.load_and_preprocess_data(force_reload=False)  # Set to True to force reload
    
    print("\n🔬 Running comprehensive analysis with professional visualizations...")
    
    # Create comprehensive dashboard with separate visualizations
    analyzer.create_comprehensive_dashboard()

if __name__ == "__main__":
    main()

# Additional utility functions for cache management
def clear_all_cache():
    """Utility function to clear all cached data"""
    analyzer = SentimentPriceElasticityIntegration()
    analyzer.clear_cache()

def force_reload_data():
    """Utility function to force reload all data from source"""
    analyzer = SentimentPriceElasticityIntegration()
    analyzer.load_and_preprocess_data(force_reload=True)
    return analyzer

def quick_analysis():
    """Quick analysis using cached data if available"""
    analyzer = SentimentPriceElasticityIntegration()
    analyzer.load_and_preprocess_data()
    
    # Run quick analysis with visualizations
    analyzer.create_comprehensive_dashboard()
    
    return analyzer
