import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

# Set professional plotting style with improved parameters
plt.style.use('default')
sns.set_palette("tab10")
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#666666',
    'axes.linewidth': 1.2
})

def drop_duplicate_columns(df):
    """Drop duplicate columns, keeping first occurrence. Print duplicate names found."""
    v = pd.Series(df.columns)
    dups = v[v.duplicated()].unique()
    if len(dups):
        print(f"❗ Duplicate columns detected: {list(dups)}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print(f"✅ Duplicates dropped. Columns now: {list(df.columns)}")
    return df

def drop_duplicate_index(df):
    """Drop duplicate index, keeping first occurrence."""
    if df.index.duplicated().any():
        print("Dropping duplicate DataFrame indices")
        df = df[~df.index.duplicated(keep='first')]
    return df.reset_index(drop=True)

def canonicalize_financial_columns(df):
    """Canonicalize column names to standard format for consistent processing"""
    if df is None or df.empty:
        return df

    column_mapping = {
        'sales_detail_participation_date': 'date',
        'sales detail participation date': 'date',
        'sales detail participation date time': 'date',
        'participation_date': 'date', 'participation date': 'date',
        'transaction_date': 'date', 'transaction date': 'date',
        'sale_date': 'date', 'sale date': 'date',
        'attendance_detail_date': 'date', 'attendance detail date': 'date',
        'sales_detail_gross_amount': 'price',
        'sales detail gross amount': 'price',
        'sales_detail_net_amount': 'price',
        'sales detail net amount': 'price',
        'gross_amount': 'price',
        'gross amount': 'price',
        'net_amount': 'price',
        'net amount': 'price',
        'amount': 'price', 'cost': 'price', 'value': 'price',
        'sales_detail_quantity': 'quantity',
        'sales detail quantity': 'quantity', 'qty': 'quantity',
        'count': 'quantity', 'units': 'quantity', 'number': 'quantity',
        'product_hierarchy_product': 'product',
        'product hierarchy product': 'product',
        'sales_detail_price_level': 'customer_type',
        'sales detail price level': 'customer_type',
        'contacts_detail_price_level': 'customer_type',
        'contacts detail price level': 'customer_type',
        'price_level': 'customer_type', 'price level': 'customer_type',
        'sales_detail_status': 'status', 'sales detail status': 'status',
        'unique_key': 'unique_id', 'unique key': 'unique_id',
        'contacts_detail_unique_key': 'unique_id',
        'contacts detail unique key': 'unique_id'
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

    # Drop any new duplicates after renaming
    df = drop_duplicate_columns(df)
    return df

class EnhancedSentimentPriceElasticityIntegration:
    def __init__(self, cache_dir='./data_cache', pkl_data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pkl_data_dir = pkl_data_dir
        self.financial_data = None
        self.attendance_data = None
        self.booking_data = None
        self.utilisation_data = None
        self.cancellation_data = None
        self.nps_data = None
        self.qualitative_feedback = None
        self.user_data = None
        self.integrated_sentiment_data = None
        self.elasticity_sentiment_matrix = None
        
        self.excluded_profiles = [
            'Junior 8 To 13 Years', 'Junior 14 &amp; Over Non Memb', 'Junior Non Member',
            'Child Staff Lrn2 Member', 'Suspended Alumni', 'Non-Member Over 65'
        ]
        
        self.primary_profiles = [
            'Student Member', 'Staff Member', 'Alumni Member',
            'Community Member', 'Associate Member',
            'Community Over 65 Member', 'Staff Non Member'
        ]

    def load_pkl_datasets(self):
        datasets = {}
        pkl_files = {
            'financial': 'financial.pkl',
            'attendance': 'attendance.pkl',
            'bookings_squash': 'bookings_squash.pkl',
            'bookings_classes': 'bookings_classes.pkl',
            'bookings_alt_sessions': 'bookings_alt_sessions.pkl',
            'bookings_other': 'bookings_other_activities.pkl',
            'nps': 'nps.pkl',
            'cancellation': 'cancellations.pkl'
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

        try:
            qualitative_df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')
            datasets['qualitative'] = qualitative_df
            print(f"✅ Loaded qualitative feedback: {len(qualitative_df)} records")
        except Exception as e:
            print(f"❌ Error loading qualitative feedback: {str(e)}")

        return datasets

    def load_excel_fallback(self, dataset_name):
        try:
            if dataset_name == 'financial':
                financial_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name=['S&amp;F', 'Tiverton'])
                df = pd.concat([financial_sheets['S&amp;F'], financial_sheets['Tiverton']], ignore_index=True)
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'attendance':
                attendance_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx',
                    sheet_name=['Gym', 'Gym 2','Reception Barrier 1','Reception Barrier 2','Recpetion Barrier 3', 'Reception Barrier 4'])
                df = pd.concat([attendance_sheets['Gym'], attendance_sheets['Gym 2'],
                               attendance_sheets['Reception Barrier 1'], attendance_sheets['Reception Barrier 2'],
                               attendance_sheets['Recpetion Barrier 3'], attendance_sheets['Reception Barrier 4']],
                              ignore_index=True)
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'user':
                df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/User_25_G24.xlsx', sheet_name='Sheet1')
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'booking':
                df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Booking_and_Utilisation_25_G24.xlsx')
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'utilisation':
                util_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Utilisation_Data_25G24.xlsx',
                    sheet_name=['2023 July to Dec', '2024 Jan - June','2024 July - Dec', '2025 Jan - May'])
                df = pd.concat([util_sheets['2023 July to Dec'], util_sheets['2024 Jan - June'],
                               util_sheets['2024 July - Dec'], util_sheets['2025 Jan - May']], ignore_index=True)
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'cancellation':
                df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Cancellation_Updated_25_G24.xlsx')
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'nps':
                df = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')
                df = drop_duplicate_columns(df)
                return df
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} from Excel: {e}")
            return None

    def load_and_preprocess_data(self, force_reload=False):
        print("\n🔄 Starting enhanced data preprocessing for primary member profiles...")
        
        try:
            pkl_datasets = self.load_pkl_datasets()
            
            print("📈 Loading financial data...")
            if 'financial' in pkl_datasets and pkl_datasets['financial'] is not None:
                self.financial_data = pkl_datasets['financial']
                if isinstance(self.financial_data, dict) and 'S&amp;F' in self.financial_data:
                    self.financial_data = pd.concat([self.financial_data['S&amp;F'], self.financial_data['Tiverton']], ignore_index=True)
            else:
                self.financial_data = self.load_excel_fallback('financial')
            
            self.financial_data = drop_duplicate_columns(self.financial_data)
            self.financial_data = drop_duplicate_index(self.financial_data)
            self.financial_data = canonicalize_financial_columns(self.financial_data)
            self.financial_data = drop_duplicate_columns(self.financial_data)
            
            print("👥 Loading attendance data...")
            if 'attendance' in pkl_datasets and pkl_datasets['attendance'] is not None:
                self.attendance_data = pkl_datasets['attendance']
                if isinstance(self.attendance_data, dict):
                    all_attendance = []
                    for sheet_name, sheet_data in self.attendance_data.items():
                        if sheet_data is not None:
                            all_attendance.append(sheet_data)
                    self.attendance_data = pd.concat(all_attendance, ignore_index=True) if all_attendance else pd.DataFrame()
            else:
                self.attendance_data = self.load_excel_fallback('attendance')
            
            self.attendance_data = drop_duplicate_columns(self.attendance_data)
            self.attendance_data = drop_duplicate_index(self.attendance_data)
            self.attendance_data = canonicalize_financial_columns(self.attendance_data)
            self.attendance_data = drop_duplicate_columns(self.attendance_data)
            
            print("⭐ Loading NPS data...")
            if 'nps' in pkl_datasets and pkl_datasets['nps'] is not None:
                self.nps_data = pkl_datasets['nps']
            else:
                self.nps_data = self.load_excel_fallback('nps')

            if 'qualitative' in pkl_datasets:
                self.qualitative_feedback = pkl_datasets['qualitative']
            
            print("👤 Loading user profile data...")
            self.user_data = self.load_excel_fallback('user')
            if self.user_data is not None:
                self.user_data = drop_duplicate_columns(self.user_data)
                self.user_data = drop_duplicate_index(self.user_data)
            
            if 'cancellation' in pkl_datasets:
                self.cancellation_data = pkl_datasets['cancellation']
            else:
                self.cancellation_data = self.load_excel_fallback('cancellation')
            
            if self.cancellation_data is not None:
                self.cancellation_data = drop_duplicate_columns(self.cancellation_data)
                self.cancellation_data = drop_duplicate_index(self.cancellation_data)
            
            self._filter_primary_profiles()
            print("📅 Preprocessing dates...")
            self._preprocess_dates()
            print("✅ Enhanced data loaded and preprocessed successfully!")
            
        except Exception as e:
            print(f"❌ Error during data loading: {e}")
            raise

    def _filter_primary_profiles(self):
        print("🎯 Filtering data for primary member profiles...")
        
        if self.financial_data is not None:
            self.financial_data = drop_duplicate_columns(self.financial_data)
            initial_financial_count = len(self.financial_data)
            
            if 'customer_type' in self.financial_data.columns:
                self.financial_data = self.financial_data[
                    self.financial_data['customer_type'].isin(self.primary_profiles)
                ]
                self.financial_data = drop_duplicate_columns(self.financial_data)
                
                if 'price' in self.financial_data.columns:
                    self.financial_data = self.financial_data[
                        (self.financial_data['price'] > 0) & (self.financial_data['customer_type'].notna())
                    ]
                self.financial_data = drop_duplicate_columns(self.financial_data)
                print(f"  Financial data: {initial_financial_count} → {len(self.financial_data)} records")
                print(f"  Primary profiles kept: {sorted(self.financial_data['customer_type'].unique().tolist())}")
        
        if self.user_data is not None:
            self.user_data = drop_duplicate_columns(self.user_data)
            initial_user_count = len(self.user_data)
            
            price_level_col = None
            for col in self.user_data.columns:
                if 'price level' in col.lower():
                    price_level_col = col
                    break
            
            if price_level_col:
                self.user_data = self.user_data[
                    self.user_data[price_level_col].isin(self.primary_profiles)
                ]
                self.user_data = drop_duplicate_columns(self.user_data)
                print(f"  User data: {initial_user_count} → {len(self.user_data)} records")
        
        if self.financial_data is not None and self.attendance_data is not None:
            self.attendance_data = drop_duplicate_columns(self.attendance_data)
            if 'unique_id' in self.financial_data.columns:
                primary_member_ids = self.financial_data['unique_id'].unique()
                initial_attendance_count = len(self.attendance_data)
                
                if 'unique_id' in self.attendance_data.columns:
                    self.attendance_data = self.attendance_data[
                        self.attendance_data['unique_id'].isin(primary_member_ids)
                    ]
                self.attendance_data = drop_duplicate_columns(self.attendance_data)
                print(f"  Attendance data: {initial_attendance_count} → {len(self.attendance_data)} records")

    def _preprocess_dates(self):
        try:
            if self.financial_data is not None and 'date' in self.financial_data.columns:
                self.financial_data['date'] = pd.to_datetime(self.financial_data['date'], errors='coerce')
            
            if self.attendance_data is not None and 'date' in self.attendance_data.columns:
                self.attendance_data['date'] = pd.to_datetime(self.attendance_data['date'], errors='coerce')
            
            if self.nps_data is not None:
                for col in self.nps_data.columns:
                    if 'date' in col.lower():
                        self.nps_data[col] = pd.to_datetime(self.nps_data[col], errors='coerce')
        except Exception as e:
            print(f"⚠️ Warning during date preprocessing: {e}")

    def calculate_price_elasticity_by_segment(self):
        """Calculate price elasticity for primary member segments using data-driven approach"""
        print("📊 Calculating price elasticity for primary member segments...")
        
        if self.financial_data is None or self.attendance_data is None:
            print("❌ Missing required data for elasticity calculation")
            return {}

        required_financial_cols = ['unique_id', 'customer_type', 'price', 'date']
        required_attendance_cols = ['unique_id']
        
        missing_financial = [col for col in required_financial_cols if col not in self.financial_data.columns]
        missing_attendance = [col for col in required_attendance_cols if col not in self.attendance_data.columns]

        if missing_financial:
            print(f"❌ Missing financial columns: {missing_financial}")
            print(f"Available financial columns: {list(self.financial_data.columns)}")
            return {}

        if missing_attendance:
            print(f"❌ Missing attendance columns: {missing_attendance}")
            print(f"Available attendance columns: {list(self.attendance_data.columns)}")
            return {}

        print("✅ All required columns found!")

        unique_key_col = 'unique_id'
        price_level_col = 'customer_type'
        price_col = 'price'
        date_col = 'date'
        attendance_key_col = 'unique_id'

        if 'date' in self.attendance_data.columns:
            key_for_count = 'date'
        else:
            key_for_count = 'unique_id'

        usage_frequency = self.attendance_data.groupby(attendance_key_col).agg({key_for_count: 'count'}).reset_index()
        usage_frequency.columns = ['Unique_Key', 'Total_Visits']

        valid_data = self.financial_data.merge(
            usage_frequency, left_on=unique_key_col, right_on='Unique_Key', how='inner'
        )

        valid_data = valid_data[
            (valid_data[price_col] > 0) &
            (valid_data['Total_Visits'] > 0)
        ]

        print(f"  Valid records for analysis: {len(valid_data)}")

        valid_data = valid_data[valid_data[price_level_col].isin(self.primary_profiles)]
        print(f"  Records after primary profile filter: {len(valid_data)}")
        print(f"  Customer types in analysis: {sorted(valid_data[price_level_col].unique().tolist())}")

        elasticity_by_tier = {}

        for tier in valid_data[price_level_col].dropna().unique():
            tier_data = valid_data[valid_data[price_level_col] == tier].copy()
            
            if len(tier_data) < 10:
                print(f"  Skipping {tier}: insufficient sample size ({len(tier_data)} < 10)")
                continue

            if date_col and date_col in tier_data.columns:
                tier_data = tier_data.sort_values(date_col)

            tier_data['Price_Change'] = tier_data[price_col].pct_change()
            tier_data['Demand_Change'] = tier_data['Total_Visits'].pct_change()

            valid_changes = tier_data[
                (tier_data['Price_Change'].notna()) &
                (tier_data['Demand_Change'].notna()) &
                (abs(tier_data['Price_Change']) > 0.01)
            ]

            if len(valid_changes) > 0:
                elasticity = (valid_changes['Demand_Change'] / valid_changes['Price_Change']).mean()
            else:
                price_demand_corr = tier_data[[price_col, 'Total_Visits']].corr().iloc[0, 1]
                elasticity = price_demand_corr if not pd.isna(price_demand_corr) else 0

            elasticity_by_tier[tier] = {
                'elasticity': elasticity,
                'sample_size': len(tier_data),
                'avg_price': tier_data[price_col].mean(),
                'avg_visits': tier_data['Total_Visits'].mean(),
                'price_std': tier_data[price_col].std(),
                'visits_std': tier_data['Total_Visits'].std()
            }

            print(f"  {tier}: Elasticity = {elasticity:.3f}, Sample = {len(tier_data)}")

        return elasticity_by_tier

    def calculate_sentiment_scores(self):
        """Calculate sentiment scores from NPS and qualitative feedback"""
        print("😊 Calculating sentiment scores...")
        
        nps_sentiment = None
        qualitative_sentiment = None

        if self.nps_data is not None:
            nps_sentiment = self.nps_data.copy()
            
            score_col = None
            for col in nps_sentiment.columns:
                if 'score' in col.lower():
                    score_col = col
                    break

            if score_col:
                nps_sentiment['Sentiment_Score'] = nps_sentiment[score_col].apply(
                    lambda x: (x - 6) / 4
                )
                nps_sentiment['Sentiment_Score'] = nps_sentiment['Sentiment_Score'].clip(-1, 1)

                nps_sentiment['Sentiment_Category'] = nps_sentiment[score_col].apply(
                    lambda x: 'Promoter' if x >= 9 else ('Passive' if x >= 7 else 'Detractor')
                )

        if self.qualitative_feedback is not None and not self.qualitative_feedback.empty:
            qualitative_sentiment = self.qualitative_feedback.copy()
            
            feedback_col = None
            for col in qualitative_sentiment.columns:
                if 'feedback' in col.lower():
                    feedback_col = col
                    break

            if feedback_col:
                def analyze_text_sentiment(text):
                    if pd.isna(text):
                        return 0
                    blob = TextBlob(str(text))
                    return blob.sentiment.polarity

                qualitative_sentiment['Text_Sentiment'] = qualitative_sentiment[feedback_col].apply(
                    analyze_text_sentiment
                )
        else:
            qualitative_sentiment = pd.DataFrame()

        return nps_sentiment, qualitative_sentiment

    def integrate_sentiment_with_elasticity(self):
        """Integrate sentiment analysis with price elasticity for strategic insights"""
        print("🔗 Integrating sentiment with price elasticity...")

        nps_sentiment, qualitative_sentiment = self.calculate_sentiment_scores()
        elasticity_data = self.calculate_price_elasticity_by_segment()

        if not elasticity_data:
            print("❌ No elasticity data available")
            return pd.DataFrame()

        matrix_data = []

        if self.financial_data is not None and 'unique_id' in self.financial_data.columns:
            member_value = self.financial_data.groupby('unique_id').agg({
                'price': ['sum', 'mean'],
                'customer_type': 'first'
            }).reset_index()
            member_value.columns = ['Unique_Key', 'Total_Revenue', 'Avg_Revenue', 'Price_Level']

            member_value = member_value[member_value['Price_Level'].isin(self.primary_profiles)]

            sentiment_by_tier = pd.DataFrame()
            if nps_sentiment is not None:
                unique_id_col = None
                for col in nps_sentiment.columns:
                    if 'unique' in col.lower() or 'id' in col.lower():
                        unique_id_col = col
                        break

                if unique_id_col:
                    sentiment_by_member = nps_sentiment.merge(
                        member_value, left_on=unique_id_col, right_on='Unique_Key', how='left'
                    )

                    sentiment_by_tier = sentiment_by_member.groupby('Price_Level')['Sentiment_Score'].agg(['mean', 'count']).reset_index()
                    sentiment_by_tier.columns = ['Price_Level', 'Avg_Sentiment', 'Sentiment_Sample_Size']

            for tier, elasticity_info in elasticity_data.items():
                if tier not in self.primary_profiles:
                    continue

                tier_sentiment = sentiment_by_tier[sentiment_by_tier['Price_Level'] == tier] if not sentiment_by_tier.empty else pd.DataFrame()

                if not tier_sentiment.empty:
                    avg_sentiment = tier_sentiment['Avg_Sentiment'].iloc[0]
                    sentiment_sample = tier_sentiment['Sentiment_Sample_Size'].iloc
                else:
                    avg_sentiment = 0
                    sentiment_sample = 0

                matrix_data.append({
                    'Tier': tier,
                    'Sentiment_Score': avg_sentiment,
                    'Elasticity_Value': elasticity_info['elasticity'],
                    'Sample_Size': elasticity_info['sample_size'],
                    'Sentiment_Sample_Size': sentiment_sample,
                    'Avg_Price': elasticity_info['avg_price'],
                    'Avg_Visits': elasticity_info['avg_visits'],
                    'Price_Std': elasticity_info['price_std'],
                    'Visits_Std': elasticity_info['visits_std']
                })

        matrix_df = pd.DataFrame(matrix_data)

        if not matrix_df.empty:
            sentiment_median = matrix_df['Sentiment_Score'].median()
            elasticity_median = abs(matrix_df['Elasticity_Value']).median()

            matrix_df['Sentiment_Category'] = matrix_df['Sentiment_Score'].apply(
                lambda x: 'Positive' if x >= sentiment_median else 'Negative'
            )

            matrix_df['Elasticity_Category'] = matrix_df['Elasticity_Value'].apply(
                lambda x: 'Inelastic' if abs(x) <= elasticity_median else 'Elastic'
            )

            def get_data_driven_strategy(row):
                if row['Sentiment_Category'] == 'Positive' and row['Elasticity_Category'] == 'Inelastic':
                    price_increase = min(20, max(5, abs(row['Elasticity_Value']) * 100))
                    return f'Premium Pricing (+{price_increase:.1f}%)'
                elif row['Sentiment_Category'] == 'Positive' and row['Elasticity_Category'] == 'Elastic':
                    return 'Volume Strategy (Competitive Pricing)'
                elif row['Sentiment_Category'] == 'Negative' and row['Elasticity_Category'] == 'Inelastic':
                    return 'Service Improvement Priority'
                else:
                    return 'Retention Focus (Careful Pricing)'

            matrix_df['Strategy_Recommendation'] = matrix_df.apply(get_data_driven_strategy, axis=1)

            matrix_df['Sample_Size'] = pd.to_numeric(matrix_df['Sample_Size'], errors='coerce').fillna(0)
            matrix_df['Sentiment_Sample_Size'] = pd.to_numeric(matrix_df['Sentiment_Sample_Size'], errors='coerce').fillna(0)
            matrix_df['Analysis_Confidence'] = matrix_df.apply(
                lambda row: min(1.0, (row['Sample_Size'] + row['Sentiment_Sample_Size']) / 50), axis=1
            )

            self.elasticity_sentiment_matrix = matrix_df
            print(f"✅ Created matrix with {len(matrix_df)} primary member profiles")
            print(f"Primary profiles in analysis: {sorted(matrix_df['Tier'].unique().tolist())}")

            return matrix_df

        return pd.DataFrame()

    def get_optimal_label_positions(self, x_vals, y_vals, labels, width_ratio=0.15, height_ratio=0.1):
        """Calculate optimal label positions to avoid overlaps using a grid-based approach"""
        if not x_vals or not y_vals or not labels:
            return []
        
        # Normalize coordinates to [0,1] range
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        positions = []
        
        # Define candidate positions around each point (in relative coordinates)
        offsets = [
            (width_ratio, height_ratio),      # top-right
            (-width_ratio, height_ratio),     # top-left  
            (width_ratio, -height_ratio),     # bottom-right
            (-width_ratio, -height_ratio),    # bottom-left
            (width_ratio, 0),                 # right
            (-width_ratio, 0),                # left
            (0, height_ratio),                # top
            (0, -height_ratio),               # bottom
        ]
        
        for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
            best_pos = None
            min_overlap = float('inf')
            
            # Try each offset position
            for dx, dy in offsets:
                # Convert to actual coordinates
                label_x = x + dx * x_range
                label_y = y + dy * y_range
                
                # Check overlap with existing positions
                overlap_count = 0
                for prev_x, prev_y, _ in positions:
                    dist = ((label_x - prev_x) / x_range) ** 2 + ((label_y - prev_y) / y_range) ** 2
                    if dist < (width_ratio * 2) ** 2:  # Overlap threshold
                        overlap_count += 1
                
                if overlap_count < min_overlap:
                    min_overlap = overlap_count
                    best_pos = (label_x, label_y)
            
            if best_pos is None:
                # Fallback to top-right if no good position found
                best_pos = (x + width_ratio * x_range, y + height_ratio * y_range)
            
            positions.append((*best_pos, label))
        
        return positions

    def create_sentiment_elasticity_scatter(self):
        """Create professional sentiment-elasticity scatter plot with improved labeling"""
        if self.elasticity_sentiment_matrix is None or self.elasticity_sentiment_matrix.empty:
            print("❌ No data available for sentiment-elasticity scatter plot")
            return

        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create scatter plot
        colors = ['#2E8B57' if cat == 'Positive' else '#DC143C' 
                 for cat in self.elasticity_sentiment_matrix['Sentiment_Category']]
        markers = ['s' if cat == 'Inelastic' else 'o' 
                  for cat in self.elasticity_sentiment_matrix['Elasticity_Category']]
        
        # Plot points with varied sizes based on sample size
        max_sample = self.elasticity_sentiment_matrix['Sample_Size'].max()
        sizes = [max(200, (size/max_sample) * 800) for size in self.elasticity_sentiment_matrix['Sample_Size']]
        
        for i, (_, row) in enumerate(self.elasticity_sentiment_matrix.iterrows()):
            ax.scatter(row['Elasticity_Value'], row['Sentiment_Score'], 
                      s=sizes[i], c=[colors[i]], marker=markers[i],
                      alpha=0.8, edgecolors='white', linewidth=2.5)

        # Add reference lines
        sentiment_median = self.elasticity_sentiment_matrix['Sentiment_Score'].median()
        elasticity_median = -abs(self.elasticity_sentiment_matrix['Elasticity_Value']).median()
        
        ax.axhline(y=sentiment_median, color='#666666', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=elasticity_median, color='#666666', linestyle='--', alpha=0.7, linewidth=2)

        # Get optimal label positions
        x_vals = self.elasticity_sentiment_matrix['Elasticity_Value'].tolist()
        y_vals = self.elasticity_sentiment_matrix['Sentiment_Score'].tolist()
        labels = [tier.replace(' Member', '').replace(' Non', ' N').replace('Community Over 65', 'Comm 65+') 
                 for tier in self.elasticity_sentiment_matrix['Tier']]
        
        label_positions = self.get_optimal_label_positions(x_vals, y_vals, labels)

        # Add labels with connecting lines
        for i, (label_x, label_y, label) in enumerate(label_positions):
            orig_x, orig_y = x_vals[i], y_vals[i]
            
            # Add annotation with connection line
            ax.annotate(label, xy=(orig_x, orig_y), xytext=(label_x, label_y),
                       fontsize=11, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                               edgecolor='black', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                     color='black', alpha=0.7, lw=1.5))

        # Customize plot
        ax.set_xlabel('Price Elasticity', fontweight='bold', fontsize=16)
        ax.set_ylabel('Sentiment Score', fontweight='bold', fontsize=16)
        ax.set_title('Sentiment-Elasticity Matrix\nUoB Sports & Fitness Centre - Primary Profiles Only',
                    fontweight='bold', fontsize=18, pad=25)

        # Create legend
        legend_elements = [
            Patch(facecolor='#2E8B57', label='Positive Sentiment', alpha=0.8),
            Patch(facecolor='#DC143C', label='Negative Sentiment', alpha=0.8),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=14, label='Inelastic', markeredgecolor='white', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=14, label='Elastic', markeredgecolor='white', markeredgewidth=2)
        ]
        
        legend = ax.legend(handles=legend_elements, loc='upper left', 
                          bbox_to_anchor=(1.02, 1), frameon=True,
                          fontsize=14, title='Legend', title_fontsize=16)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        # Set axis limits with padding
        x_margin = (max(x_vals) - min(x_vals)) * 0.15
        y_margin = (max(y_vals) - min(y_vals)) * 0.15
        ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

        plt.tight_layout()
        plt.savefig('sentiment_elasticity_scatter.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

    def create_price_usage_analysis(self):
        """Create professional price vs usage analysis chart"""
        if self.elasticity_sentiment_matrix is None or self.elasticity_sentiment_matrix.empty:
            print("❌ No data available for price usage analysis")
            return

        fig, ax = plt.subplots(figsize=(16, 12))

        # Create scatter plot with confidence coloring
        scatter = ax.scatter(self.elasticity_sentiment_matrix['Avg_Price'],
                           self.elasticity_sentiment_matrix['Avg_Visits'],
                           s=self.elasticity_sentiment_matrix['Sample_Size']/100,
                           c=self.elasticity_sentiment_matrix['Analysis_Confidence'],
                           cmap='viridis', alpha=0.8, edgecolors='white', linewidth=2.5)

        # Get optimal label positions  
        x_vals = self.elasticity_sentiment_matrix['Avg_Price'].tolist()
        y_vals = self.elasticity_sentiment_matrix['Avg_Visits'].tolist()
        labels = [tier.replace(' Member', '').replace(' Non', ' N').replace('Community Over 65', 'Comm 65+') 
                 for tier in self.elasticity_sentiment_matrix['Tier']]
        
        label_positions = self.get_optimal_label_positions(x_vals, y_vals, labels, 
                                                         width_ratio=0.12, height_ratio=0.08)

        # Add labels with connecting lines
        for i, (label_x, label_y, label) in enumerate(label_positions):
            orig_x, orig_y = x_vals[i], y_vals[i]
            
            ax.annotate(label, xy=(orig_x, orig_y), xytext=(label_x, label_y),
                       fontsize=11, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                               edgecolor='black', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                     color='black', alpha=0.7, lw=1.5))

        # Customize plot
        ax.set_xlabel('Average Price (£)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Average Visits per Member', fontweight='bold', fontsize=16)
        ax.set_title('Price vs Usage by Member Tier\nUoB Sports & Fitness Centre - Primary Profiles Only',
                    fontweight='bold', fontsize=18, pad=25)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Analysis Confidence', fontweight='bold', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Set axis limits with padding
        x_margin = (max(x_vals) - min(x_vals)) * 0.1
        y_margin = (max(y_vals) - min(y_vals)) * 0.1
        ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

        plt.tight_layout()
        plt.savefig('price_usage_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

    def create_confidence_assessment(self):
        """Create professional confidence assessment charts"""
        if self.elasticity_sentiment_matrix is None or self.elasticity_sentiment_matrix.empty:
            print("❌ No data available for confidence assessment")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Shorten tier names for better display
        short_names = [tier.replace(' Member', '').replace(' Non', ' N').replace('Community Over 65', 'Comm 65+')
                    for tier in self.elasticity_sentiment_matrix['Tier']]

        # Bar chart for sample sizes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.elasticity_sentiment_matrix)))
        bars = ax1.bar(range(len(self.elasticity_sentiment_matrix)),
                    self.elasticity_sentiment_matrix['Sample_Size'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_xticks(range(len(self.elasticity_sentiment_matrix)))
        ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=12)
        ax1.set_ylabel('Sample Size', fontweight='bold', fontsize=14)
        ax1.set_title('Sample Sizes by Member Tier', fontweight='bold', fontsize=16, pad=20)

        # Add value labels on bars with better positioning
        for i, (bar, val) in enumerate(zip(bars, self.elasticity_sentiment_matrix['Sample_Size'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(val*0.01, 500),
                    f'{int(val):,}', ha='center', va='bottom', fontweight='bold',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Scatter plot for confidence assessment with improved positioning
        np.random.seed(42)
        jitter = np.random.uniform(-0.001, 0.001, len(self.elasticity_sentiment_matrix))

        scatter = ax2.scatter(self.elasticity_sentiment_matrix['Sample_Size'],
                            self.elasticity_sentiment_matrix['Analysis_Confidence'] + jitter,
                            c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)

        # Use the improved label positioning function for scatter plot
        x_vals = self.elasticity_sentiment_matrix['Sample_Size'].tolist()
        y_vals = (self.elasticity_sentiment_matrix['Analysis_Confidence'] + jitter).tolist()
        
        # Calculate optimal label positions using the same function as other charts
        label_positions = self.get_optimal_label_positions(x_vals, y_vals, short_names, 
                                                        width_ratio=0.08, height_ratio=0.002)

        # Add labels with connecting lines
        for i, (label_x, label_y, label) in enumerate(label_positions):
            orig_x, orig_y = x_vals[i], y_vals[i]
            
            ax2.annotate(label, xy=(orig_x, orig_y), xytext=(label_x, label_y),
                        fontsize=11, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='black', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                    color='black', alpha=0.7, lw=1.5))

        ax2.set_xlabel('Sample Size', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Analysis Confidence', fontweight='bold', fontsize=14)
        ax2.set_title('Analysis Confidence Assessment', fontweight='bold', fontsize=16, pad=20)

        # Set proper axis limits with adequate margins
        x_margin = (max(x_vals) - min(x_vals)) * 0.15
        y_margin = (max(y_vals) - min(y_vals)) * 0.3  # Larger margin for y-axis due to small range
        ax2.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax2.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

        plt.suptitle('Data Quality Assessment\nUoB Sports & Fitness Centre - Primary Profiles Only',
                    fontweight='bold', fontsize=18, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('confidence_assessment.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
        plt.show()
        plt.close()


    def create_strategy_distribution_chart(self):
        """Create professional strategy recommendations distribution chart"""
        if self.elasticity_sentiment_matrix is None or self.elasticity_sentiment_matrix.empty:
            print("❌ No data available for strategy distribution chart")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        strategy_counts = self.elasticity_sentiment_matrix['Strategy_Recommendation'].value_counts()
        
        # Use professional color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_counts)))
        
        # Create pie chart with better spacing
        wedges, texts, autotexts = ax.pie(strategy_counts.values,
                                         labels=None,  # Remove labels from pie to avoid overlap
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90,
                                         explode=[0.08] * len(strategy_counts),
                                         textprops={'fontsize': 12, 'fontweight': 'bold'})

        # Enhance autopct text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Create legend with strategy names and counts
        legend_labels = [f'{strategy} ({count})' for strategy, count in strategy_counts.items()]
        ax.legend(wedges, legend_labels, title="Strategy Recommendations", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=12, title_fontsize=14)

        ax.set_title('Strategic Recommendations Distribution\nUoB Sports & Fitness Centre - Primary Profiles Only',
                    fontweight='bold', fontsize=16, pad=25)

        plt.tight_layout()
        plt.savefig('strategy_distribution.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

    def create_strategic_dashboard(self):
        """Create comprehensive strategic dashboard with improved professional visualizations"""
        print("\n📊 Creating Strategic Dashboard...")

        matrix_df = self.integrate_sentiment_with_elasticity()

        if matrix_df.empty:
            print("❌ No data available for analysis")
            return

        print("🎨 Generating professional visualization charts...")

        # Create individual charts with improved styling
        self.create_sentiment_elasticity_scatter()
        self.create_strategy_distribution_chart()
        self.create_price_usage_analysis()
        self.create_confidence_assessment()

        # Print strategic insights
        self._print_strategic_insights(matrix_df)

    def _print_strategic_insights(self, matrix_df):
        """Print data-driven strategic insights"""
        print("\n" + "="*80)
        print("🎯 DATA-DRIVEN STRATEGIC INSIGHTS - UoB SPORTS & FITNESS CENTRE")
        print("🎯 PRIMARY MEMBER PROFILES ANALYSIS")
        print("="*80)

        for _, row in matrix_df.iterrows():
            print(f"\n📊 {row['Tier']} Membership:")
            print(f"  Strategy: {row['Strategy_Recommendation']}")
            print(f"  Sentiment Score: {row['Sentiment_Score']:.3f}")
            print(f"  Price Elasticity: {row['Elasticity_Value']:.3f}")
            print(f"  Average Price: £{row['Avg_Price']:.2f} ± £{row['Price_Std']:.2f}")
            print(f"  Average Visits: {row['Avg_Visits']:.1f} ± {row['Visits_Std']:.1f}")
            print(f"  Sample Size: {row['Sample_Size']} members")
            print(f"  Analysis Confidence: {row['Analysis_Confidence']:.1%}")

        # Overall strategic recommendations
        print(f"\n🎯 TOP STRATEGIC RECOMMENDATIONS:")
        print("-" * 60)

        premium_tiers = matrix_df[matrix_df['Strategy_Recommendation'].str.contains('Premium', na=False)]
        if not premium_tiers.empty:
            print(f"1. PREMIUM PRICING OPPORTUNITIES:")
            for _, tier in premium_tiers.iterrows():
                try:
                    potential_increase = float(tier['Strategy_Recommendation'].split('(+')[1].split('%'))
                    revenue_impact = tier['Avg_Price'] * potential_increase / 100 * tier['Sample_Size']
                    print(f"  → {tier['Tier']}: {potential_increase}% increase (£{revenue_impact:,.0f} potential revenue)")
                except:
                    print(f"  → {tier['Tier']}: Premium pricing opportunity identified")

        retention_tiers = matrix_df[matrix_df['Strategy_Recommendation'].str.contains('Retention', na=False)]
        if not retention_tiers.empty:
            print(f"2. RETENTION FOCUS REQUIRED:")
            for _, tier in retention_tiers.iterrows():
                print(f"  → {tier['Tier']}: {tier['Sample_Size']} members at risk")

        service_tiers = matrix_df[matrix_df['Strategy_Recommendation'].str.contains('Service', na=False)]
        if not service_tiers.empty:
            print(f"3. SERVICE IMPROVEMENT PRIORITY:")
            for _, tier in service_tiers.iterrows():
                print(f"  → {tier['Tier']}: Address satisfaction issues ({tier['Sentiment_Score']:.2f} sentiment)")

        print(f"\n📈 OVERALL IMPACT ESTIMATION:")
        total_members = matrix_df['Sample_Size'].sum()
        weighted_elasticity = (matrix_df['Elasticity_Value'] * matrix_df['Sample_Size']).sum() / total_members
        weighted_sentiment = (matrix_df['Sentiment_Score'] * matrix_df['Sample_Size']).sum() / total_members

        print(f"  Total Primary Members Analyzed: {total_members:,}")
        print(f"  Weighted Average Elasticity: {weighted_elasticity:.3f}")
        print(f"  Weighted Average Sentiment: {weighted_sentiment:.3f}")
        print(f"  Primary Profiles Included: {sorted(matrix_df['Tier'].unique().tolist())}")

        print("\n" + "="*80)

# Main execution function
def main():
    """Main execution function"""
    print("🚀 Enhanced Sentiment-Price Elasticity Analysis")
    print("🎯 Focus: UoB Sports & Fitness Centre - PRIMARY MEMBER PROFILES ONLY")
    print("📊 Strategy: Data-Driven Pricing Recommendations")
    print("=" * 80)

    analyzer = EnhancedSentimentPriceElasticityIntegration()

    try:
        analyzer.load_and_preprocess_data()
        analyzer.create_strategic_dashboard()

        print("✅ Analysis completed successfully!")
        print("📁 Professional charts saved as PNG files:")
        print("  - sentiment_elasticity_scatter.png")
        print("  - strategy_distribution.png")
        print("  - price_usage_analysis.png")
        print("  - confidence_assessment.png")
        print(f"📂 Charts saved in: {os.getcwd()}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Additional utility functions
def quick_analysis():
    """Quick analysis function"""
    analyzer = EnhancedSentimentPriceElasticityIntegration()
    analyzer.load_and_preprocess_data()
    return analyzer.integrate_sentiment_with_elasticity()

def get_pricing_recommendations():
    """Get specific pricing recommendations"""
    analyzer = EnhancedSentimentPriceElasticityIntegration()
    analyzer.load_and_preprocess_data()
    matrix_df = analyzer.integrate_sentiment_with_elasticity()
    
    recommendations = {}
    for _, row in matrix_df.iterrows():
        recommendations[row['Tier']] = {
            'strategy': row['Strategy_Recommendation'],
            'confidence': row['Analysis_Confidence'],
            'current_price': row['Avg_Price'],
            'sample_size': row['Sample_Size']
        }
    
    return recommendations
