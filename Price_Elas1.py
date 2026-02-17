# Enhanced Price Elasticity Analysis for University of Birmingham Sports & Fitness Centre

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
import warnings
import os
import pickle
from datetime import datetime, timedelta
import calendar
from scipy import stats
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



print("✅ Libraries imported successfully!")

#Data Loading
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

def prepare_financial_data_for_elasticity(financial_data):
    """Prepare financial data with proper column mapping for elasticity analysis"""
    
    # Handle different possible data structures
    if isinstance(financial_data, dict):
        # If it's a dictionary with sheets, get the S&F sheet
        if 'S&F' in financial_data:
            df = financial_data['S&F'].copy()
        elif 'Sheet1' in financial_data:
            df = financial_data['Sheet1'].copy()
        else:
            # Take the first sheet
            df = list(financial_data.values())[0].copy()
    else:
        df = financial_data.copy()
    
    print(f"📊 Financial data columns: {list(df.columns)}")
    print(f"📋 Data shape: {df.shape}")
    
    # Define column mapping - check which columns actually exist
    column_mapping = {
        'sales_detail_participation_date': 'date',
        'sales_detail_gross_amount': 'price',
        'sales_detail_quantity': 'quantity', 
        'product_hierarchy_product': 'product',
        'sales_detail_price_level': 'customer_type',
        'sales_detail_status': 'status',
        'unique_key': 'unique_id'
    }
    
    # Apply column mapping only for columns that exist
    mapped_columns = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
            mapped_columns[old_col] = new_col
            print(f"✅ Mapped '{old_col}' -> '{new_col}'")
        else:
            print(f"⚠️ Column '{old_col}' not found in data")
    
    # Check if essential columns were mapped successfully
    essential_columns = ['date', 'price', 'quantity']
    missing_columns = []
    
    for col in essential_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        print(f"❌ Missing essential columns: {missing_columns}")
        print("Available columns after mapping:")
        print(df.columns.tolist())
        
        # Try alternative column names
        alternative_mappings = {
            'date': ['Date', 'Transaction Date', 'Sale Date', 'Participation Date', 'Sales Detail Participation Date Time'],
            'price': ['Price', 'Amount', 'Cost', 'Value', 'Gross Amount', 'Sales Detail Net Amount'],
            'quantity': ['Qty', 'Count', 'Units', 'Number', 'Sales Detail Quantity']
        }
        
        for missing_col in missing_columns:
            found = False
            if missing_col in alternative_mappings:
                for alt_name in alternative_mappings[missing_col]:
                    if alt_name in df.columns:
                        df[missing_col] = df[alt_name]
                        print(f"✅ Found alternative mapping: '{alt_name}' -> '{missing_col}'")
                        found = True
                        break
            
            if not found:
                print(f"❌ Could not find suitable column for '{missing_col}'")
                return None
    
    # Now safely process the date column
    if 'date' in df.columns:
        print(f"🔄 Processing date column...")
        print(f"Original date sample: {df['date'].head()}")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️ Found {invalid_dates} invalid dates that were set to NaT")
    
    # Process price column safely
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        invalid_prices = df['price'].isna().sum()
        if invalid_prices > 0:
            print(f"⚠️ Found {invalid_prices} invalid prices that were set to NaN")
    
    # Process quantity column safely
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        invalid_quantities = df['quantity'].isna().sum()
        if invalid_quantities > 0:
            print(f"⚠️ Found {invalid_quantities} invalid quantities that were set to NaN")
    
    # Filter for valid data
    print(f"🔄 Filtering data...")
    initial_rows = len(df)
    
    # Remove rows with missing essential data
    if all(col in df.columns for col in ['date', 'price', 'quantity']):
        df = df.dropna(subset=['date', 'price', 'quantity'])
        df = df[df['quantity'] > 0]  # Positive quantities only
        df = df[df['price'] > 0]    # Positive prices only
        
        print(f"📊 Filtered from {initial_rows} to {len(df)} rows")
    else:
        print("❌ Cannot filter data - essential columns missing")
        return None
    
    # Filter for paid transactions if status column exists
    if 'status' in df.columns:
        paid_statuses = ['Paid', 'paid', 'PAID', 'Completed', 'Complete']
        df = df[df['status'].isin(paid_statuses)]
        print(f"💰 Kept only paid transactions: {len(df)} rows")
    
    # Create time-based features only if date column exists and is valid
    if 'date' in df.columns and not df['date'].isna().all():
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Create academic calendar features
        df['term_time'] = df['month'].apply(
            lambda x: 1 if x in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6] else 0
        )
        
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("❌ Cannot create time-based features - invalid date column")
        return None
    
    # Peak vs Off-peak indicator (only if product column exists)
    if 'product' in df.columns:
        df['is_offpeak'] = df['product'].str.contains('Op|off.peak|Off.Peak', 
                                                      case=False, na=False).astype(int)
        print(f"🏷️ Products: {df['product'].nunique()} unique products")
    
    if 'customer_type' in df.columns:
        print(f"👥 Customer types: {df['customer_type'].nunique()} unique types")
    
    print(f"✅ Successfully prepared financial data: {len(df)} valid transactions")
    
    return df


def load_attendance_usage_data(attendance_data):
    """Load and prepare attendance data for usage correlation"""
    
    usage_data = {}
    
    if isinstance(attendance_data, dict):
        for sheet_name, sheet_data in attendance_data.items():
            if sheet_data is not None and len(sheet_data) > 0:
                df = sheet_data.copy()
                
                # Map attendance columns
                attendance_mapping = {
                    'Attendance Detail Date': 'date',
                    'Unique Key': 'unique_id',
                    'Contacts Detail Price Level': 'customer_type'
                }
                
                for old_col, new_col in attendance_mapping.items():
                    if old_col in df.columns:
                        df[new_col] = df[old_col]
                
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df['year_month'] = df['date'].dt.to_period('M')
                
                usage_data[sheet_name.lower()] = df
                print(f"✅ Loaded {sheet_name} attendance: {len(df)} records")
    
    return usage_data

# Load all datasets
print("🔄 Loading datasets...")
datasets = load_pkl_datasets()

# Prepare main financial data
if 'financial' in datasets:
    financial_df = prepare_financial_data_for_elasticity(datasets['financial'])
else:
    print("❌ Financial data not found. Cannot proceed with elasticity analysis.")
    financial_df = None

# Load attendance data
if 'attendance' in datasets:
    attendance_data = load_attendance_usage_data(datasets['attendance'])
else:
    print("⚠️ Attendance data not found.")
    attendance_data = {}

# Load NPS data
if 'nps' in datasets:
    nps_df = datasets['nps'].copy()
    if 'Response Date' in nps_df.columns:
        nps_df['Response Date'] = pd.to_datetime(nps_df['Response Date'], format='%d/%m/%Y', errors='coerce')
    print(f"✅ NPS data loaded: {len(nps_df)} records")
else:
    print("⚠️ NPS data not found.")
    nps_df = None

#Aggregation Functions
def create_monthly_aggregation(financial_df, attendance_data=None):
    """Create monthly aggregated data for elasticity analysis"""
    
    if financial_df is None:
        print("❌ No financial data available for aggregation")
        return None
    
    # Monthly aggregation by product and customer type
    monthly_agg = financial_df.groupby([
        'year_month', 'product', 'customer_type'
    ]).agg({
        'quantity': 'sum',
        'price': 'mean',  # Average price in the period
        'month': 'first',
        'quarter': 'first', 
        'year': 'first',
        'term_time': 'first',
        'is_offpeak': 'first',
        'unique_id': 'nunique'  # Count of unique customers
    }).reset_index()
    
    # Rename columns for clarity
    monthly_agg.columns = ['period', 'product', 'customer_type', 'quantity', 
                          'avg_price', 'month', 'quarter', 'year', 'term_time', 
                          'is_offpeak', 'unique_customers']
    
    # Add usage data if available
    if attendance_data:
        usage_monthly = {}
        for facility, usage_df in attendance_data.items():
            if len(usage_df) > 0:
                facility_usage = usage_df.groupby(['year_month', 'customer_type']).agg({
                    'unique_id': 'nunique'
                }).reset_index()
                facility_usage.columns = ['period', 'customer_type', f'{facility}_users']
                usage_monthly[facility] = facility_usage
        
        # Merge usage data
        for facility, usage_df in usage_monthly.items():
            monthly_agg = monthly_agg.merge(
                usage_df, 
                on=['period', 'customer_type'], 
                how='left'
            )
            monthly_agg[f'{facility}_users'] = monthly_agg[f'{facility}_users'].fillna(0)
    
    # Filter out periods with very low quantities
    monthly_agg = monthly_agg[monthly_agg['quantity'] >= 5].copy()
    
    # Calculate price per unit (handle cases where quantity might be memberships)
    monthly_agg['price_per_unit'] = monthly_agg['avg_price'] / monthly_agg['quantity']
    monthly_agg['price_per_unit'] = monthly_agg['price_per_unit'].fillna(monthly_agg['avg_price'])
    
    # Create log variables for elasticity analysis
    monthly_agg['log_quantity'] = np.log(monthly_agg['quantity'])
    monthly_agg['log_price'] = np.log(monthly_agg['avg_price'].abs())  # Use absolute value
    
    # Remove invalid log values
    monthly_agg = monthly_agg.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['log_quantity', 'log_price']
    )
    
    print(f"✅ Created monthly aggregation: {len(monthly_agg)} observations")
    print(f"📊 Product-Customer combinations: {monthly_agg.groupby(['product', 'customer_type']).size().shape[0]}")
    
    return monthly_agg

# Create monthly aggregated data
if financial_df is not None:
    monthly_data = create_monthly_aggregation(financial_df, attendance_data)
else:
    monthly_data = None

#Elasticity Analysis
class EnhancedPriceElasticityAnalyzer:
    """Enhanced Price Elasticity Analysis with proper column handling"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.elasticity_results = []
        
        # Product categories for analysis
        self.product_categories = {
            'gym': ['gym', 'chrissie'],
            'swim': ['swim', 'pool'],
            'inclusive': ['inclusive', 'all'],
            'classes': ['class', 'group'],
            'courts': ['squash', 'badminton', 'court']
        }
        
    def categorize_product(self, product_name):
        """Categorize products into main groups"""
        if pd.isna(product_name):
            return 'other'
        
        product_lower = str(product_name).lower()
        
        for category, keywords in self.product_categories.items():
            if any(keyword in product_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def create_seasonality_features(self, df):
        """Create seasonality dummy variables"""
        
        # Create month dummies (excluding December to avoid multicollinearity)
        for month in range(1, 12):
            df[f'month_{month}'] = (df['month'] == month).astype(int)
        
        # Create quarter dummies (excluding Q4)
        for quarter in range(1, 4):
            df[f'quarter_{quarter}'] = (df['quarter'] == quarter).astype(int)
        
        return df
    
    def run_elasticity_regression(self, subset_data, segment_name):
        """Run log-log regression for price elasticity"""
        
        if len(subset_data) < 10:
            print(f"⚠️ Insufficient data for {segment_name}: {len(subset_data)} observations")
            return None
        
        try:
            # Prepare data
            model_data = subset_data.copy()
            model_data = self.create_seasonality_features(model_data)
            
            # Define model variables
            month_vars = [f'month_{i}' for i in range(1, 12)]
            control_vars = ['term_time', 'is_offpeak'] + month_vars
            
            # Remove variables with no variation
            control_vars = [var for var in control_vars 
                          if var in model_data.columns and model_data[var].std() > 0]
            
            # Create formula
            if control_vars:
                formula = f"log_quantity ~ log_price + {' + '.join(control_vars)}"
            else:
                formula = "log_quantity ~ log_price"
            
            # Fit model
            model = smf.ols(formula, data=model_data).fit()
            
            # Extract results
            elasticity = model.params['log_price']
            elasticity_se = model.bse['log_price']
            p_value = model.pvalues['log_price']
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            
            result = {
                'segment': segment_name,
                'elasticity': elasticity,
                'std_error': elasticity_se,
                'p_value': p_value,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'n_observations': len(model_data),
                'confidence_interval': [
                    elasticity - 1.96 * elasticity_se,
                    elasticity + 1.96 * elasticity_se
                ],
                'model': model,
                'formula': formula
            }
            
            print(f"✅ {segment_name}: Elasticity = {elasticity:.3f}, R² = {r_squared:.3f}")
            return result
            
        except Exception as e:
            print(f"❌ Model failed for {segment_name}: {str(e)}")
            return None
    
    def analyze_by_product_category(self):
        """Analyze elasticity by product category"""
        
        print("\n🔄 Analyzing by product category...")
        
        # Add product category
        self.data['product_category'] = self.data['product'].apply(self.categorize_product)
        
        for category in self.data['product_category'].unique():
            if category != 'other':
                category_data = self.data[self.data['product_category'] == category]
                
                result = self.run_elasticity_regression(
                    category_data, f"Product Category: {category.title()}"
                )
                if result:
                    self.elasticity_results.append(result)
                    self.models[f"category_{category}"] = result['model']
    
    def analyze_by_customer_type(self):
        """Analyze elasticity by customer type"""
        
        print("\n🔄 Analyzing by customer type...")
        
        for customer_type in self.data['customer_type'].unique():
            if pd.notna(customer_type):
                customer_data = self.data[self.data['customer_type'] == customer_type]
                
                result = self.run_elasticity_regression(
                    customer_data, f"Customer: {customer_type}"
                )
                if result:
                    self.elasticity_results.append(result)
                    self.models[f"customer_{customer_type}"] = result['model']
    
    def analyze_peak_vs_offpeak(self):
        """Analyze elasticity for peak vs off-peak pricing"""
        
        print("\n🔄 Analyzing peak vs off-peak...")
        
        for is_offpeak in [0, 1]:
            peak_label = "Off-Peak" if is_offpeak else "Peak"
            peak_data = self.data[self.data['is_offpeak'] == is_offpeak]
            
            result = self.run_elasticity_regression(
                peak_data, f"Pricing: {peak_label}"
            )
            if result:
                self.elasticity_results.append(result)
                self.models[f"peak_{peak_label.lower()}"] = result['model']
    
    def run_overall_analysis(self):
        """Run overall elasticity analysis"""
        
        print("\n🔄 Running overall analysis...")
        
        result = self.run_elasticity_regression(self.data, "Overall Market")
        if result:
            self.elasticity_results.append(result)
            self.models["overall"] = result['model']
    
    def get_results_summary(self):
        """Get summary of all results"""
        
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

# Run enhanced elasticity analysis
if monthly_data is not None:
    print("\n" + "="*60)
    print("🚀 RUNNING ENHANCED PRICE ELASTICITY ANALYSIS")
    print("="*60)
    
    analyzer = EnhancedPriceElasticityAnalyzer(monthly_data)
    
    # Run all analyses
    analyzer.run_overall_analysis() 
    analyzer.analyze_by_product_category()
    analyzer.analyze_by_customer_type()
    analyzer.analyze_peak_vs_offpeak()
    
    # Get results
    results_df = analyzer.get_results_summary()
    
    if not results_df.empty:
        print(f"\n✅ Analysis complete! Generated {len(results_df)} elasticity estimates")
        print("\n📊 SUMMARY OF RESULTS:")
        print(results_df[['Segment', 'Price_Elasticity', 'R_Squared', 'Significant', 'Elasticity_Type']].to_string())
    else:
        print("❌ No valid results generated")
else:
    print("❌ Cannot run analysis - no valid monthly data available")

# Visualization of Results
def create_elasticity_overview_chart(results_df):
    """Create a clean overview of price elasticity results"""
    
    if results_df.empty:
        print("⚠️ No results to visualize")
        return
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort results by absolute elasticity for better visualization
    results_sorted = results_df.sort_values('Price_Elasticity', key=abs, ascending=True)
    
    # Create color scheme based on elasticity values
    colors = []
    for val in results_sorted['Price_Elasticity']:
        if abs(val) > 1:
            colors.append('#e74c3c')  # Red for elastic
        else:
            colors.append('#3498db')  # Blue for inelastic
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(results_sorted)), 
                   results_sorted['Price_Elasticity'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels([label.replace('Product: ', '').replace('Customer: ', '') 
                       for label in results_sorted['Segment']], fontsize=10)
    ax.set_xlabel('Price Elasticity Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('Price Elasticity Analysis Results\nSports & Fitness Centre', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add reference lines
    ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, results_sorted['Price_Elasticity'])):
        width = bar.get_width()
        x_pos = width + (0.05 if width >= 0 else -0.05)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if width >= 0 else 'right', 
                va='center', fontweight='bold', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Elastic (|elasticity| > 1)'),
                      Patch(facecolor='#3498db', label='Inelastic (|elasticity| ≤ 1)')]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
             fancybox=True, shadow=True)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('price_elasticity_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Execute the visualization
if 'results_df' in locals() and not results_df.empty:
    create_elasticity_overview_chart(results_df)

#Statistical Significance and Recommendations
def create_statistical_significance_chart(results_df):
    """Create a chart showing statistical significance of results"""
    
    if results_df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Elasticity vs P-value scatter plot
    significant = results_df[results_df['P_Value'] < 0.05]
    not_significant = results_df[results_df['P_Value'] >= 0.05]
    
    ax1.scatter(significant['Price_Elasticity'], significant['P_Value'], 
               c='green', s=120, alpha=0.7, label='Significant (p < 0.05)', 
               edgecolors='darkgreen', linewidth=1)
    ax1.scatter(not_significant['Price_Elasticity'], not_significant['P_Value'], 
               c='red', s=120, alpha=0.7, label='Not Significant (p ≥ 0.05)',
               edgecolors='darkred', linewidth=1)
    
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Price Elasticity Coefficient', fontweight='bold')
    ax1.set_ylabel('P-Value', fontweight='bold')
    ax1.set_title('Statistical Significance of Elasticity Estimates', 
                 fontweight='bold', fontsize=14)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(results_df['P_Value'].max() * 1.1, 0.1))
    
    # Add annotations for significant threshold
    ax1.text(results_df['Price_Elasticity'].mean(), 0.05, 
            'Significance Threshold (p = 0.05)', 
            verticalalignment='bottom', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Chart 2: R-squared values
    segments_short = [s.replace('Product: ', '').replace('Customer: ', '') 
                     for s in results_df['Segment']]
    
    colors_r2 = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' 
                for r2 in results_df['R_Squared']]
    
    bars = ax2.bar(range(len(results_df)), results_df['R_Squared'], 
                   color=colors_r2, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(segments_short, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('R-Squared', fontweight='bold')
    ax2.set_title('Model Quality (R-Squared Values)', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, results_df['R_Squared']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('Statistical Analysis of Price Elasticity Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('elasticity_statistical_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Execute the visualization
if 'results_df' in locals() and not results_df.empty:
    create_statistical_significance_chart(results_df)

#Customer Segmentation

def create_customer_segment_analysis(results_df):
    """Create analysis for customer segment elasticity"""
    
    if results_df.empty:
        return
    
    # Filter for customer results
    customer_results = results_df[results_df['Segment'].str.contains('Customer:', na=False)]
    
    if customer_results.empty:
        print("⚠️ No customer segment results found")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Clean customer names
    customer_results = customer_results.copy()
    customer_results['Customer_Type'] = customer_results['Segment'].str.replace('Customer: ', '')
    customer_results = customer_results.sort_values('Price_Elasticity', ascending=True)
    
    # Chart 1: Horizontal bar chart for elasticity
    colors = ['#e74c3c' if abs(x) > 1 else '#3498db' for x in customer_results['Price_Elasticity']]
    bars1 = ax1.barh(range(len(customer_results)), customer_results['Price_Elasticity'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_yticks(range(len(customer_results)))
    ax1.set_yticklabels(customer_results['Customer_Type'], fontsize=11)
    ax1.set_xlabel('Price Elasticity Coefficient', fontweight='bold', fontsize=12)
    ax1.set_title('Price Elasticity by Customer Segment', fontweight='bold', fontsize=14)
    
    # Add reference lines
    ax1.axvline(x=-1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars1, customer_results['Price_Elasticity']):
        width = bar.get_width()
        x_pos = width + (0.05 if width >= 0 else -0.05)
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if width >= 0 else 'right', 
                va='center', fontweight='bold', fontsize=10)
    
    ax1.grid(axis='x', alpha=0.3)
    
    # Chart 2: Sample sizes and confidence
    bars2 = ax2.bar(range(len(customer_results)), customer_results['N_Observations'],
                    color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xticks(range(len(customer_results)))
    ax2.set_xticklabels(customer_results['Customer_Type'], rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Observations', fontweight='bold', fontsize=12)
    ax2.set_title('Sample Sizes by Customer Segment', fontweight='bold', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars2, customer_results['N_Observations']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Customer Segment Price Elasticity Analysis\nSports & Fitness Centre', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('customer_segment_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Execute the visualization
if 'results_df' in locals() and not results_df.empty:
    create_customer_segment_analysis(results_df)


#Comprehensive Dashboard
def create_elasticity_dashboard(results_df):
    """Create a comprehensive dashboard using Plotly"""
    
    if results_df.empty:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Elasticity Distribution', 'Statistical Significance', 
                       'Model Quality Assessment', 'Elasticity vs Sample Size'],
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Elasticity distribution histogram
    fig.add_trace(
        go.Histogram(
            x=results_df['Price_Elasticity'],
            nbinsx=15,
            name='Elasticity Distribution',
            marker_color='lightblue',
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Add vertical lines for reference
    fig.add_vline(x=-1, line_dash="dash", line_color="red", 
                  annotation_text="Unit Elastic", row=1, col=1)
    fig.add_vline(x=1, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=0, line_dash="solid", line_color="black", row=1, col=1)
    
    # 2. Statistical significance scatter
    colors = ['green' if p < 0.05 else 'red' for p in results_df['P_Value']]
    fig.add_trace(
        go.Scatter(
            x=results_df['Price_Elasticity'],
            y=results_df['P_Value'],
            mode='markers',
            marker=dict(
                color=colors,
                size=12,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=results_df['Segment'],
            name='Significance Test'
        ),
        row=1, col=2
    )
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                  annotation_text="Significance Threshold", row=1, col=2)
    
    # 3. Model quality (R-squared)
    fig.add_trace(
        go.Bar(
            x=results_df['Segment'],
            y=results_df['R_Squared'],
            name='R-Squared',
            marker_color='lightgreen',
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # 4. Elasticity vs Sample Size
    fig.add_trace(
        go.Scatter(
            x=results_df['N_Observations'],
            y=results_df['Price_Elasticity'],
            mode='markers',
            marker=dict(
                color=results_df['R_Squared'],
                colorscale='Viridis',
                size=12,
                opacity=0.8,
                colorbar=dict(title="R-Squared"),
                line=dict(width=1, color='black')
            ),
            text=results_df['Segment'],
            name='Sample Size Analysis'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Price Elasticity Analysis Dashboard - Sports & Fitness Centre",
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Price Elasticity", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Price Elasticity", row=1, col=2)
    fig.update_yaxes(title_text="P-Value", row=1, col=2)
    
    fig.update_xaxes(title_text="Segment", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="R-Squared", row=2, col=1)
    
    fig.update_xaxes(title_text="Number of Observations", row=2, col=2)
    fig.update_yaxes(title_text="Price Elasticity", row=2, col=2)
    
    # Save and show
    fig.write_html('elasticity_dashboard.html')
    fig.show()

# Execute the dashboard
if 'results_df' in locals() and not results_df.empty:
    create_elasticity_dashboard(results_df)


def create_strategic_recommendations_table(results_df):
    """Create strategic recommendations based on elasticity results"""
    
    if results_df.empty:
        return
    
    print("\n" + "="*80)
    print("🎯 STRATEGIC PRICING RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    for _, row in results_df.iterrows():
        segment = row['Segment']
        elasticity = row['Price_Elasticity']
        significant = row['Significant']
        
        if not significant:
            recommendation = "⚠️ Results not statistically significant - proceed with caution"
            strategy = "Conduct more detailed analysis"
        elif abs(elasticity) < 0.5:
            recommendation = "💰 STRONG PRICING POWER - Consider 5-15% price increase"
            strategy = "Revenue maximization opportunity"
        elif abs(elasticity) < 1:
            recommendation = "💡 MODERATE PRICING POWER - Consider 2-8% price increase"
            strategy = "Balanced approach to revenue growth"
        else:
            recommendation = "⚠️ PRICE SENSITIVE - Focus on value enhancement"
            strategy = "Avoid price increases, improve service quality"
        
        recommendations.append({
            'Segment': segment,
            'Elasticity': f"{elasticity:.3f}",
            'Recommendation': recommendation,
            'Strategy': strategy
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    # Display as formatted table
    print(recommendations_df.to_string(index=False, max_colwidth=50))
    
    return recommendations_df

#Strategic Insights Chart
def create_strategic_insights_chart(results_df):
    """Create a chart summarizing strategic insights"""
    
    if results_df.empty:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Elasticity categories pie chart
    elastic_count = len(results_df[abs(results_df['Price_Elasticity']) > 1])
    inelastic_count = len(results_df[abs(results_df['Price_Elasticity']) <= 1])
    
    sizes = [elastic_count, inelastic_count]
    labels = ['Elastic\n(Price Sensitive)', 'Inelastic\n(Price Insensitive)']
    colors = ['#e74c3c', '#3498db']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
                                      autopct='%1.1f%%', startangle=90, shadow=True)
    ax1.set_title('Price Sensitivity Distribution', fontweight='bold', fontsize=14)
    
    # 2. Statistical significance
    sig_count = len(results_df[results_df['P_Value'] < 0.05])
    not_sig_count = len(results_df) - sig_count
    
    bars = ax2.bar(['Significant\n(p < 0.05)', 'Not Significant\n(p ≥ 0.05)'], 
                   [sig_count, not_sig_count],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
    
    ax2.set_title('Statistical Significance of Results', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Number of Segments', fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 3. Model quality distribution
    high_quality = len(results_df[results_df['R_Squared'] > 0.7])
    medium_quality = len(results_df[(results_df['R_Squared'] > 0.5) & (results_df['R_Squared'] <= 0.7)])
    low_quality = len(results_df[results_df['R_Squared'] <= 0.5])
    
    bars3 = ax3.bar(['High\n(R² > 0.7)', 'Medium\n(0.5 < R² ≤ 0.7)', 'Low\n(R² ≤ 0.5)'],
                    [high_quality, medium_quality, low_quality],
                    color=['darkgreen', 'orange', 'red'], alpha=0.7, edgecolor='black')
    
    ax3.set_title('Model Quality Distribution', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Number of Models', fontweight='bold')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 4. Average elasticity by category (if available)
    if 'product_results' in locals() and 'customer_results' in locals():
        categories = ['Products', 'Customers']
        avg_elasticities = [
            abs(results_df[results_df['Segment'].str.contains('Product:', na=False)]['Price_Elasticity']).mean(),
            abs(results_df[results_df['Segment'].str.contains('Customer:', na=False)]['Price_Elasticity']).mean()
        ]
        
        bars4 = ax4.bar(categories, avg_elasticities, 
                       color=['skyblue', 'lightcoral'], alpha=0.7, edgecolor='black')
        
        ax4.set_title('Average Price Sensitivity by Category', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Average |Elasticity|', fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars4, avg_elasticities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data\nfor Category Analysis', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
        ax4.set_title('Category Analysis', fontweight='bold', fontsize=14)
    
    plt.suptitle('Strategic Price Elasticity Insights\nSports & Fitness Centre', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('strategic_insights_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Execute the strategic insights visualization
if 'results_df' in locals() and not results_df.empty:
    create_strategic_insights_chart(results_df)


# Generate visualizations and recommendations
if 'results_df' in locals() and not results_df.empty:
    recommendations_df = create_strategic_recommendations_table(results_df)
    
    # Export results
    with pd.ExcelWriter('enhanced_price_elasticity_results.xlsx') as writer:
        results_df.to_excel(writer, sheet_name='Elasticity_Results', index=False)
        if 'recommendations_df' in locals():
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
        if monthly_data is not None:
            monthly_data.to_excel(writer, sheet_name='Monthly_Data', index=False)
    
    print(f"\n✅ Results exported to 'enhanced_price_elasticity_results.xlsx'")

