# Enhanced Price Elasticity Analysis for University of Birmingham Sports & Fitness Centre
# Redesigned with focus on core adult segments and improved statistical rigor
# FIXED VERSION - Resolves year_month KeyError issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import mstats
import warnings
import os
import pickle
from datetime import datetime, timedelta
import calendar
from scipy import stats
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Set styling for better visuals
plt.style.use('default')  # Changed from seaborn for better control
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# =============================================================================
# PART 1: CONFIGURATION PARAMETERS
# =============================================================================

# Core analysis parameters
adults_only = True
include_over65 = False
min_obs = 12
min_qty_per_period = 5
winsor_limits = (0.01, 0.99)
robust_se = "HC3"
use_usage_controls = True

print("✅ Enhanced Price Elasticity Analysis - University of Birmingham Sports & Fitness Centre")
print("🎯 Focus: Core adult segments with improved statistical rigor\n")

# =============================================================================
# PART 2: CANONICAL MAPPINGS AND HELPER FUNCTIONS
# =============================================================================

def get_canonical_mappings():
    """Define canonical product and customer type mappings"""
    product_mapping = {
        'gym': ['gym', 'chrissie', 'fitness', 'weight'],
        'swim': ['swim', 'pool', 'aqua'],
        'classes': ['class', 'group', 'cardio', 'toning', 'body pump', 'pilates', 'yoga'],
        'courts': ['squash', 'badminton', 'court', 'racket'],
        'inclusive': ['inclusive', 'all', 'unlimited', 'combined']
    }
    
    customer_mapping = {
        'Student Member': ['student member', 'uobstm', 'student'],
        'Writing Up Student': ['writing up student'],
        'Associate Member': ['associate member'],
        'Staff Member': ['staff member', 'staffm'],
        'Staff Non Member': ['staff non member'],
        'Alumni Member': ['alumni member', 'alum'],
        'Community Member': ['community member', 'comm'],
        'Community Over 65 Member': ['community over 65', 'rcomm', 'over 65'],
        'Non-Member': ['non-member', 'non member', 'stand', 'alts']
    }
    
    # Define exclusion patterns for filtering
    exclusion_patterns = [
        'junior', 'child', 'kid', 'youth', 'under', 'lrn2', 'learn',
        'external booking', 'ubs', 'internal', 'partner', 'test', 'comp'
    ]
    
    return product_mapping, customer_mapping, exclusion_patterns

def normalize_labels(label, mapping):
    """Normalize labels using canonical mappings"""
    if pd.isna(label):
        return None
    label_lower = str(label).lower()
    for canonical, variants in mapping.items():
        for variant in variants:
            if variant in label_lower:
                return canonical
    return 'other'

def filter_adult_segments(df, adults_only=True, include_over65=False):
    """Filter dataset for adult segments only"""
    adult_customers = [
        'Student Member', 'Writing Up Student', 'Associate Member', 'Staff Member', 
        'Staff Non Member', 'Alumni Member', 'Community Member', 'Non-Member'
    ]
    
    if include_over65:
        adult_customers.append('Community Over 65 Member')
    
    # Age-based filtering (if age column exists)
    if 'Contacts Detail Age' in df.columns:
        df_filtered = df[
            (df['Contacts Detail Age'] >= 16) | 
            (df['customer_type_canonical'].isin(adult_customers))
        ]
    else:
        df_filtered = df.copy()
    
    if adults_only:
        df_filtered = df_filtered[df_filtered['customer_type_canonical'].isin(adult_customers)]
    
    return df_filtered

def winsorize_series(series, limits=(0.01, 0.99)):
    """Winsorize series to handle outliers"""
    return mstats.winsorize(series, limits=limits)

def check_segment_viability(subset, min_obs=12, min_qty=5):
    """Check if segment has sufficient data for reliable analysis"""
    unique_periods = subset['period'].nunique()
    if unique_periods < min_obs:
        return False, f"Insufficient periods: {unique_periods} < {min_obs}"
    
    if subset['quantity'].var() == 0 or subset['avg_price'].var() == 0:
        return False, "No variation in quantity or price"
    
    if (subset['quantity'] < min_qty).sum() > len(subset) * 0.5:
        return False, f"Too many periods below quantity threshold {min_qty}"
    
    return True, "Viable"

def drop_high_vif_features(df, features, thresh=10.0):
    """Drop features with high VIF to avoid multicollinearity"""
    features = features.copy()
    while True:
        try:
            vif_df = pd.DataFrame()
            vif_df["Feature"] = features
            vif_df["VIF"] = [variance_inflation_factor(df[features].values, i) 
                            for i in range(len(features))]
            max_vif = vif_df["VIF"].max()
            if max_vif > thresh:
                drop_feature = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
                features.remove(drop_feature)
                print(f"Dropped {drop_feature} (VIF={max_vif:.2f})")
            else:
                break
        except:
            break
    return features

# =============================================================================
# PART 3: DATA LOADING AND PREPARATION - FIXED VERSION
# =============================================================================

def load_pkl_datasets(data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
    """Load existing pkl files and the qualitative feedback data"""
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

    # Load pkl files
    for name, filename in pkl_files.items():
        try:
            if os.path.exists(f'{data_dir}/{filename}'):
                with open(f'{data_dir}/{filename}', 'rb') as f:
                    datasets[name] = pickle.load(f)
                print(f"✅ Loaded {name} data from {filename}")
            else:
                print(f"⚠️ {filename} not found, trying alternative loading...")
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")

    # Load qualitative feedback from Excel (not pkl)
    try:
        qualitative_df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')
        datasets['qualitative'] = qualitative_df
        print(f"✅ Loaded qualitative feedback: {len(qualitative_df)} records")
    except Exception as e:
        print(f"❌ Error loading qualitative feedback: {str(e)}")

    return datasets

def prepare_financial_data_enhanced_fixed(financial_data):
    """Enhanced financial data preparation with robust year_month handling"""
    
    # Handle different possible data structures
    if isinstance(financial_data, dict):
        if 'S&F' in financial_data:
            df = financial_data['S&F'].copy()
        elif 'Sheet1' in financial_data:
            df = financial_data['Sheet1'].copy()
        else:
            df = list(financial_data.values())[0].copy()
    else:
        df = financial_data.copy()

    print(f"📊 Initial data shape: {df.shape}")
    print(f"📊 Available columns: {list(df.columns)}")

    # Enhanced column mapping with multiple fallbacks
    column_mappings = {
        'date': [
            'Sales Detail Participation Date',
            'Sales Detail Participation Date Time', 
            'sales_detail_participation_date',
            'date', 'Date', 'Transaction Date'
        ],
        'price': [
            'Sales Detail Gross Amount',
            'Sales Detail Net Amount',
            'sales_detail_gross_amount',
            'price', 'Price', 'Amount'
        ],
        'quantity': [
            'Sales Detail Quantity',
            'sales_detail_quantity',
            'quantity', 'Qty', 'Quantity'
        ],
        'product': [
            'Product Hierarchy Product',
            'product_hierarchy_product',
            'product', 'Product'
        ],
        'customer_type': [
            'Sales Detail Price Level',
            'sales_detail_price_level',
            'customer_type', 'Customer Type'
        ],
        'status': [
            'Sales Detail Status',
            'sales_detail_status',
            'status', 'Status'
        ],
        'unique_id': [
            'Unique Key',
            'unique_key',
            'unique_id', 'ID'
        ]
    }

    # Apply column mappings with fallbacks
    for target_col, possible_cols in column_mappings.items():
        mapped = False
        for col in possible_cols:
            if col in df.columns:
                df[target_col] = df[col]
                print(f"✅ Mapped '{col}' -> '{target_col}'")
                mapped = True
                break
        if not mapped:
            print(f"⚠️ Could not find column for '{target_col}'")

    # Essential column validation
    essential_columns = ['date', 'price', 'quantity']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing essential columns: {missing_columns}")
        return None

    # Data type conversion and validation
    print("🔄 Converting data types...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

    # Remove invalid data
    initial_rows = len(df)
    df = df.dropna(subset=['date', 'price', 'quantity'])
    df = df[df['quantity'] > 0]
    df = df[df['price'] > 0]
    
    print(f"📊 Filtered from {initial_rows} to {len(df)} rows")

    # Filter for paid transactions
    if 'status' in df.columns:
        paid_statuses = ['Paid', 'paid', 'PAID', 'Completed', 'Complete']
        df = df[df['status'].isin(paid_statuses)]
        print(f"💰 After status filter: {len(df)} rows")

    # **CRITICAL FIX**: Create robust time-based features
    print("📅 Creating time-based features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # **KEY FIX**: Use string format instead of Period objects
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    df['period'] = df['year_month']  # Create consistent alias
    
    # Verify the column was created successfully
    print(f"✅ Created year_month column with {df['year_month'].nunique()} unique periods")
    print(f"📅 Sample year_month values: {df['year_month'].head().tolist()}")
    
    # Academic calendar features
    df['term_time'] = df['month'].apply(
        lambda x: 1 if x in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6] else 0
    )

    # Peak vs Off-peak detection
    if 'product' in df.columns:
        df['is_offpeak'] = df['product'].str.contains(
            'Op|off.peak|Off.Peak', case=False, na=False
        ).astype(int)

    # Get mappings and apply canonical mappings
    try:
        product_mapping, customer_mapping, exclusion_patterns = get_canonical_mappings()
        
        if 'product' in df.columns:
            df['product_canonical'] = df['product'].apply(lambda x: normalize_labels(x, product_mapping))
        if 'customer_type' in df.columns:
            df['customer_type_canonical'] = df['customer_type'].apply(lambda x: normalize_labels(x, customer_mapping))
            
        # Filter out junior/child segments and other exclusions
        exclusion_mask = pd.Series(False, index=df.index)
        
        for pattern in exclusion_patterns:
            if 'product' in df.columns:
                exclusion_mask |= df['product'].str.contains(pattern, case=False, na=False)
            if 'customer_type' in df.columns:
                exclusion_mask |= df['customer_type'].str.contains(pattern, case=False, na=False)
        
        df_excluded = df[exclusion_mask].copy()
        df = df[~exclusion_mask].copy()
        
        print(f"📊 After exclusion filtering: {len(df)} rows ({len(df_excluded)} excluded)")
        
        # Apply adult segment filtering
        df = filter_adult_segments(df, adults_only=adults_only, include_over65=include_over65)
        
    except Exception as e:
        print(f"⚠️ Warning during filtering: {str(e)}")
        df_excluded = pd.DataFrame()

    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📊 Final prepared data shape: {df.shape}")
    
    return df

def create_monthly_aggregation_fixed(financial_df, attendance_data=None):
    """Enhanced monthly aggregation with robust year_month handling"""
    
    if financial_df is None:
        print("❌ No financial data available for aggregation")
        return None

    print("🔄 Creating monthly aggregation...")
    
    # Verify year_month column exists
    if 'year_month' not in financial_df.columns:
        print("❌ year_month column missing, recreating...")
        financial_df['year_month'] = financial_df['date'].dt.to_period('M').astype(str)
        print(f"✅ Recreated year_month column")

    # Use consistent column names for aggregation
    groupby_cols = ['year_month']
    
    # Add product and customer type if they exist
    if 'product_canonical' in financial_df.columns:
        groupby_cols.append('product_canonical')
    elif 'product' in financial_df.columns:
        groupby_cols.append('product')
        
    if 'customer_type_canonical' in financial_df.columns:
        groupby_cols.append('customer_type_canonical')
    elif 'customer_type' in financial_df.columns:
        groupby_cols.append('customer_type')
    
    # Remove any None values in groupby columns
    for col in groupby_cols:
        if col in financial_df.columns:
            financial_df[col] = financial_df[col].fillna('Unknown')

    print(f"📊 Grouping by: {groupby_cols}")
    print(f"📊 Available columns for aggregation: {list(financial_df.columns)}")

    try:
        # Monthly aggregation
        agg_dict = {
            'quantity': 'sum',
            'price': 'mean',
            'month': 'first',
            'year': 'first'
        }
        
        # Add optional columns if they exist
        optional_cols = ['quarter', 'term_time', 'is_offpeak', 'unique_id']
        for col in optional_cols:
            if col in financial_df.columns:
                agg_dict[col] = 'first' if col != 'unique_id' else 'nunique'

        monthly_agg = financial_df.groupby(groupby_cols).agg(agg_dict).reset_index()

        # Rename columns for clarity
        new_columns = ['period']
        if len(groupby_cols) > 1:
            new_columns.extend(groupby_cols[1:])  # Add product, customer_type if present
        
        new_columns.extend(['quantity', 'avg_price', 'month', 'year'])
        
        # Add optional renamed columns
        for col in optional_cols:
            if col in financial_df.columns:
                if col == 'unique_id':
                    new_columns.append('unique_customers')
                else:
                    new_columns.append(col)
        
        monthly_agg.columns = new_columns[:len(monthly_agg.columns)]
        
        print(f"✅ Successfully created monthly aggregation: {len(monthly_agg)} rows")
        
    except Exception as e:
        print(f"❌ Aggregation failed: {str(e)}")
        print("🔄 Trying simplified aggregation...")
        
        # Fallback aggregation
        monthly_agg = financial_df.groupby('year_month').agg({
            'quantity': 'sum',
            'price': 'mean',
            'month': 'first',
            'year': 'first'
        }).reset_index()
        
        monthly_agg.columns = ['period', 'quantity', 'avg_price', 'month', 'year']
        monthly_agg['product_category'] = 'All'
        monthly_agg['customer_type'] = 'All'
        monthly_agg['term_time'] = monthly_agg['month'].apply(
            lambda x: 1 if x in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6] else 0
        )
        monthly_agg['is_offpeak'] = 0
        monthly_agg['unique_customers'] = 1

    # Filter out low-volume periods
    monthly_agg = monthly_agg[monthly_agg['quantity'] >= min_qty_per_period].copy()

    # Create log variables with safety checks
    monthly_agg['log_quantity'] = np.log(monthly_agg['quantity'].clip(lower=0.1))
    monthly_agg['log_price'] = np.log(monthly_agg['avg_price'].clip(lower=0.01))

    # Remove invalid log values
    monthly_agg = monthly_agg.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['log_quantity', 'log_price']
    )

    print(f"✅ Final monthly aggregation: {len(monthly_agg)} observations")
    print(f"📊 Period range: {monthly_agg['period'].min()} to {monthly_agg['period'].max()}")
    
    return monthly_agg

# =============================================================================
# PART 4: ENHANCED ELASTICITY ANALYZER - FIXED VERSION
# =============================================================================

class EnhancedPriceElasticityAnalyzer:
    """Enhanced analyzer with robust column handling"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.elasticity_results = []
        
        print(f"🔄 Initializing analyzer with {len(data)} observations")
        print(f"📊 Available columns: {list(data.columns)}")
        
        # Verify essential columns exist
        essential_cols = ['period', 'log_quantity', 'log_price']
        missing_cols = [col for col in essential_cols if col not in data.columns]
        
        if missing_cols:
            print(f"❌ Missing essential columns: {missing_cols}")
        else:
            print("✅ All essential columns present")

    def create_seasonality_features(self, df):
        """Create seasonality features with error handling"""
        try:
            # Month dummies (excluding December)
            for month in range(1, 12):
                df[f'month_{month}'] = (df['month'] == month).astype(int)
            
            # Quarter dummies (excluding Q4) 
            if 'quarter' in df.columns:
                for quarter in range(1, 4):
                    df[f'quarter_{quarter}'] = (df['quarter'] == quarter).astype(int)
                
            return df
            
        except Exception as e:
            print(f"⚠️ Error creating seasonality features: {str(e)}")
            return df

    def run_elasticity_regression(self, subset_data, segment_name):
        """Enhanced regression with better error handling"""
        
        if len(subset_data) < 10:
            print(f"⚠️ Insufficient data for {segment_name}: {len(subset_data)} observations")
            return None

        print(f"🔄 Running regression for {segment_name}...")
        print(f"   - Data shape: {subset_data.shape}")
        print(f"   - Available columns: {list(subset_data.columns)}")

        try:
            # Prepare data
            model_data = subset_data.copy()
            
            # Check for required columns
            required_cols = ['log_quantity', 'log_price']
            missing_cols = [col for col in required_cols if col not in model_data.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns for {segment_name}: {missing_cols}")
                return None
            
            # Check for variation in key variables
            if model_data['log_price'].std() < 0.01:
                print(f"⚠️ Insufficient price variation for {segment_name}")
                return None
                
            if model_data['log_quantity'].std() < 0.01:
                print(f"⚠️ Insufficient quantity variation for {segment_name}")
                return None

            # Add seasonality features
            model_data = self.create_seasonality_features(model_data)

            # Define control variables
            potential_controls = ['term_time', 'is_offpeak']
            month_vars = [f'month_{i}' for i in range(1, 12)]
            potential_controls.extend(month_vars)

            # Select only variables that exist and have variation
            control_vars = []
            for var in potential_controls:
                if var in model_data.columns and model_data[var].std() > 0:
                    control_vars.append(var)

            # Create formula
            if control_vars:
                formula = f"log_quantity ~ log_price + {' + '.join(control_vars)}"
            else:
                formula = "log_quantity ~ log_price"

            print(f"   - Formula: {formula}")

            # Fit model
            model = smf.ols(formula, data=model_data).fit()

            # Extract results
            elasticity = model.params['log_price']
            elasticity_se = model.bse['log_price']
            p_value = model.pvalues['log_price']

            result = {
                'segment': segment_name,
                'elasticity': elasticity,
                'std_error': elasticity_se,
                'p_value': p_value,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'n_observations': len(model_data),
                'confidence_interval': [
                    elasticity - 1.96 * elasticity_se,
                    elasticity + 1.96 * elasticity_se
                ],
                'model': model,
                'formula': formula
            }

            print(f"✅ {segment_name}: Elasticity = {elasticity:.3f}, R² = {model.rsquared:.3f}")
            return result

        except Exception as e:
            print(f"❌ Model failed for {segment_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_by_customer_type_enhanced(self):
        """Enhanced customer type analysis that handles low variation issues"""
        print("\n🔄 Analyzing by customer type...")
        
        # Check if customer_type column exists
        customer_col = None
        for col in ['customer_type', 'customer_type_canonical']:
            if col in self.data.columns:
                customer_col = col
                break
                
        if customer_col is None:
            print("❌ No customer type column found")
            return
            
        customer_types = self.data[customer_col].unique()
        print(f"📊 Found {len(customer_types)} customer types: {list(customer_types)}")
        
        # Strategy 1: Individual customer types with relaxed criteria
        for customer_type in customer_types:
            if pd.notna(customer_type) and customer_type not in ['Unknown', 'other']:
                customer_data = self.data[self.data[customer_col] == customer_type]
                
                if len(customer_data) >= 6:  # Reduced threshold
                    result = self.run_elasticity_regression(
                        customer_data, f"Customer: {customer_type}"
                    )
                    
                    if result:
                        self.elasticity_results.append(result)
                        self.models[f"customer_{customer_type.replace(' ', '_')}"] = result['model']
                else:
                    print(f"⚠️ Insufficient data for {customer_type}: {len(customer_data)} observations")

        # Strategy 2: Grouped analysis if individual analysis yields few results
        if len([r for r in self.elasticity_results if 'Customer:' in r['segment']]) < 3:
            print("\n🔄 Attempting grouped customer analysis...")
            
            # Create broader customer groups
            student_types = ['Student Member', 'Writing Up Student', 'student', 'uobstm']
            staff_types = ['Staff Member', 'Staff Non Member', 'staff', 'staffm']
            community_types = ['Community Member', 'Community Over 65 Member', 'comm', 'rcomm']
            
            groups = {
                'Students': student_types,
                'Staff': staff_types,
                'Community': community_types
            }
            
            for group_name, group_types in groups.items():
                group_mask = self.data[customer_col].str.lower().str.contains(
                    '|'.join(group_types), case=False, na=False
                )
                group_data = self.data[group_mask]
                
                if len(group_data) >= 10:
                    result = self.run_elasticity_regression(
                        group_data, f"Customer Group: {group_name}"
                    )
                    
                    if result:
                        self.elasticity_results.append(result)
                        self.models[f"customer_group_{group_name.lower()}"] = result['model']

    def analyze_by_product_category(self):
        """Analyze elasticity by product category"""
        print("\n🔄 Analyzing by product category...")
        
        # Check if product column exists
        product_col = None
        for col in ['product_category', 'product_canonical', 'product']:
            if col in self.data.columns:
                product_col = col
                break
                
        if product_col is None:
            print("❌ No product column found")
            return
            
        categories = self.data[product_col].unique()
        print(f"📊 Found {len(categories)} product categories: {list(categories)}")
        
        for category in categories:
            if pd.notna(category) and category not in ['Unknown', 'other']:
                category_data = self.data[self.data[product_col] == category]
                
                if len(category_data) >= 6:
                    result = self.run_elasticity_regression(
                        category_data, f"Product: {category.title()}"
                    )
                    
                    if result:
                        self.elasticity_results.append(result)
                        self.models[f"product_{category}"] = result['model']
                else:
                    print(f"⚠️ Insufficient data for {category}: {len(category_data)} observations")

    def run_overall_analysis(self):
        """Enhanced overall analysis"""
        print("\n🔄 Running overall analysis...")
        print(f"📊 Total data shape: {self.data.shape}")
        
        # Check data availability
        if len(self.data) < 10:
            print("❌ Insufficient data for overall analysis")
            return
            
        result = self.run_elasticity_regression(self.data, "Overall Market")
        
        if result:
            self.elasticity_results.append(result)
            self.models["overall"] = result['model']

    def get_results_summary(self):
        """Get summary of results"""
        if not self.elasticity_results:
            return pd.DataFrame()

        summary_data = []
        for result in self.elasticity_results:
            summary_data.append({
                'Segment': result['segment'],
                'Price_Elasticity': result['elasticity'],
                'Std_Error': result['std_error'],
                'P_Value': result['p_value'],
                'R_Squared': result['r_squared'],
                'Adj_R_Squared': result['adj_r_squared'],
                'N_Observations': result['n_observations'],
                'CI_Lower': result['confidence_interval'][0],
                'CI_Upper': result['confidence_interval'][1],
                'Significant': result['p_value'] < 0.05,
                'Elasticity_Type': 'Elastic' if abs(result['elasticity']) > 1 else 'Inelastic'
            })

        return pd.DataFrame(summary_data)

# =============================================================================
# PART 5: ENHANCED VISUALIZATION FUNCTIONS
# =============================================================================

def create_elasticity_overview_chart(results_df, save_path='elasticity_overview.png'):
    """Create main elasticity overview chart"""
    if results_df.empty:
        print("⚠️ No results to visualize")
        return

    plt.figure(figsize=(14, 10))
    
    # Sort by absolute elasticity
    results_sorted = results_df.sort_values('Price_Elasticity', key=abs, ascending=True)
    
    # Create color scheme based on elasticity and significance
    colors = []
    for _, row in results_sorted.iterrows():
        if not row['Significant']:
            colors.append('#95a5a6')  # Gray for non-significant
        elif abs(row['Price_Elasticity']) > 1:
            colors.append('#e74c3c')  # Red for elastic
        else:
            colors.append('#3498db')  # Blue for inelastic
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(results_sorted)), results_sorted['Price_Elasticity'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize chart
    plt.yticks(range(len(results_sorted)), 
               [label.replace('Product: ', '').replace('Customer: ', '') 
                for label in results_sorted['Segment']], fontsize=11)
    plt.xlabel('Price Elasticity Coefficient', fontweight='bold', fontsize=13)
    plt.ylabel('Segments', fontweight='bold', fontsize=13)
    plt.title('Price Elasticity Analysis Results\nUniversity of Birmingham Sports & Fitness Centre', 
              fontweight='bold', fontsize=16, pad=20)
    
    # Add reference lines
    plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Unit Elastic Boundary')
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Perfectly Inelastic')
    
    # Add value labels with significance indicators
    for i, (bar, row) in enumerate(zip(bars, results_sorted.itertuples())):
        width = bar.get_width()
        x_pos = width + (0.1 if width >= 0 else -0.1)
        sig_indicator = "**" if row.P_Value < 0.01 else "*" if row.P_Value < 0.05 else ""
        plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}{sig_indicator}', ha='left' if width >= 0 else 'right',
                va='center', fontweight='bold', fontsize=10)
    
    # Enhanced legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Inelastic & Significant'),
        Patch(facecolor='#e74c3c', label='Elastic & Significant'),
        Patch(facecolor='#95a5a6', label='Not Significant'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Unit Elastic Boundaries'),
        plt.Line2D([0], [0], color='black', linestyle='-', label='Zero Elasticity')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add significance note
    plt.figtext(0.02, 0.02, "** p<0.01, * p<0.05", fontsize=9, style='italic')
    
    plt.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✅ Elasticity overview chart saved as {save_path}")

def create_statistical_significance_chart(results_df, save_path='statistical_significance.png'):
    """Create statistical significance analysis chart"""
    if results_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with different colors for significance levels
    colors = []
    sizes = []
    for _, row in results_df.iterrows():
        if row['P_Value'] < 0.01:
            colors.append('#27ae60')  # Dark green for highly significant
            sizes.append(120)
        elif row['P_Value'] < 0.05:
            colors.append('#f39c12')  # Orange for significant
            sizes.append(100)
        else:
            colors.append('#e74c3c')  # Red for not significant
            sizes.append(80)
    
    scatter = ax.scatter(results_df['Price_Elasticity'], results_df['P_Value'],
                        c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add significance threshold lines
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='5% Significance Threshold')
    ax.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.8, linewidth=2,
               label='1% Significance Threshold')
    
    # Customize chart
    ax.set_xlabel('Price Elasticity Coefficient', fontweight='bold', fontsize=13)
    ax.set_ylabel('P-Value', fontweight='bold', fontsize=13)
    ax.set_title('Statistical Significance of Elasticity Estimates\nUniversity of Birmingham Sports & Fitness Centre',
                fontweight='bold', fontsize=16, pad=20)
    
    # Add annotations for each point
    for i, row in results_df.iterrows():
        ax.annotate(row['Segment'].replace('Product: ', 'P:').replace('Customer: ', 'C:'),
                   (row['Price_Elasticity'], row['P_Value']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Highly Significant (p < 0.01)'),
        Patch(facecolor='#f39c12', label='Significant (p < 0.05)'),
        Patch(facecolor='#e74c3c', label='Not Significant (p ≥ 0.05)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization of p-values
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"✅ Statistical significance chart saved as {save_path}")

def create_strategic_recommendations_table(results_df):
    """Generate strategic recommendations table"""
    if results_df.empty:
        return pd.DataFrame()
    
    print("\n" + "="*100)
    print("🎯 COMPREHENSIVE PRICE ELASTICITY ANALYSIS SUMMARY")
    print("="*100)
    
    # Create detailed recommendations
    recommendations = []
    for _, row in results_df.iterrows():
        segment = row['Segment']
        elasticity = row['Price_Elasticity']
        significant = row['Significant']
        r_squared = row['R_Squared']
        n_obs = row['N_Observations']
        
        # Determine confidence level
        if significant and r_squared > 0.5 and n_obs >= 15:
            confidence = "High"
        elif significant and (r_squared > 0.3 or n_obs >= 10):
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Strategic recommendation
        if confidence == "Low":
            recommendation = "⚠️ INSUFFICIENT EVIDENCE - Collect more data or conduct targeted analysis"
            price_action = "No change recommended"
            risk_level = "High"
        elif abs(elasticity) < 0.5:
            recommendation = "💰 STRONG PRICING POWER - Revenue maximization opportunity"
            price_action = "Consider 5-10% price increase"
            risk_level = "Low"
        elif abs(elasticity) < 1:
            recommendation = "💡 MODERATE PRICING POWER - Balanced growth strategy"
            price_action = "Consider 2-5% price increase"
            risk_level = "Medium"
        else:
            recommendation = "⚠️ PRICE SENSITIVE - Focus on value enhancement"
            price_action = "Avoid price increases, improve service quality"
            risk_level = "Medium"
        
        recommendations.append({
            'Segment': segment,
            'Price_Elasticity': f"{elasticity:.3f}",
            'Statistical_Significance': f"p = {row['P_Value']:.3f}",
            'Model_Quality': f"R² = {r_squared:.3f}",
            'Sample_Size': int(n_obs),
            'Confidence_Level': confidence,
            'Strategic_Recommendation': recommendation,
            'Recommended_Price_Action': price_action,
            'Risk_Level': risk_level
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    # Display formatted summary
    print("\n📊 SEGMENT-BY-SEGMENT ANALYSIS:")
    print("-" * 100)
    
    for i, row in recommendations_df.iterrows():
        print(f"\n🎯 {row['Segment']}")
        print(f"   • Elasticity: {row['Price_Elasticity']}")
        print(f"   • Significance: {row['Statistical_Significance']}")
        print(f"   • Model Quality: {row['Model_Quality']} | Sample: {row['Sample_Size']} observations")
        print(f"   • Confidence: {row['Confidence_Level']}")
        print(f"   • Recommendation: {row['Strategic_Recommendation']}")
        print(f"   • Action: {row['Recommended_Price_Action']}")
        print(f"   • Risk Level: {row['Risk_Level']}")
    
    print("\n" + "="*100)
    
    return recommendations_df

# =============================================================================
# PART 6: MAIN EXECUTION - FIXED VERSION
# =============================================================================

def main_analysis():
    """Main analysis function with comprehensive error handling"""
    
    print("=" * 80)
    print("🚀 ENHANCED PRICE ELASTICITY ANALYSIS - FIXED VERSION")
    print("=" * 80)
    
    try:
        # Load datasets
        print("\n🔄 Loading datasets...")
        datasets = load_pkl_datasets()
        
        if 'financial' not in datasets:
            print("❌ Financial data not found")
            return
        
        # Prepare financial data with fixed function
        print("\n🔄 Preparing financial data...")
        financial_df = prepare_financial_data_enhanced_fixed(datasets['financial'])
        
        if financial_df is None:
            print("❌ Failed to prepare financial data")
            return
        
        # Create monthly aggregation with fixed function  
        print("\n🔄 Creating monthly aggregation...")
        monthly_data = create_monthly_aggregation_fixed(financial_df)
        
        if monthly_data is None:
            print("❌ Failed to create monthly aggregation")
            return
        
        # Run analysis with enhanced analyzer
        print("\n🔄 Running elasticity analysis...")
        analyzer = EnhancedPriceElasticityAnalyzer(monthly_data)
        
        # Run analyses
        analyzer.run_overall_analysis()
        analyzer.analyze_by_product_category()
        analyzer.analyze_by_customer_type_enhanced()
        
        # Get results
        results_df = analyzer.get_results_summary()
        
        if not results_df.empty:
            print(f"\n✅ Analysis complete! Generated {len(results_df)} elasticity estimates")
            print("\n📊 RESULTS SUMMARY:")
            print(results_df[['Segment', 'Price_Elasticity', 'P_Value', 'R_Squared', 'Significant']].to_string(index=False))
            
            # Generate visualizations
            print("\n🎨 Generating visualizations...")
            create_elasticity_overview_chart(results_df)
            create_statistical_significance_chart(results_df)
            
            # Generate recommendations
            print("\n📝 Generating strategic recommendations...")
            recommendations_df = create_strategic_recommendations_table(results_df)
            
            # Export results
            with pd.ExcelWriter('fixed_price_elasticity_results.xlsx', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Elasticity_Results', index=False)
                recommendations_df.to_excel(writer, sheet_name='Strategic_Recommendations', index=False)
                monthly_data.to_excel(writer, sheet_name='Monthly_Data', index=False)
            
            print("\n💾 Results exported to 'fixed_price_elasticity_results.xlsx'")
            
        else:
            print("❌ No valid results generated")
            
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Execute the fixed analysis
if __name__ == "__main__":
    main_analysis()
