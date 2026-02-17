import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

# --- Configuration: Setting local file path ---
DATA_PATH = "F:/UoB Study/Capstone Project/Final Project/Datasets/"

# Verify files exist before proceeding
required_files = [
    'Att_Data_25_G24.xlsx',
    'Bookings_Data_25_G24.xlsx', 
    'Financial_Data_25_G24.xlsx',
    'Cancellation_Updated_25_G24.xlsx',
    'NPS_Updated_25_G24.csv'
]

def check_files_exist(path, files):
    """Check if all required files exist in the specified path"""
    missing_files = []
    for file in files:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    
    if missing_files:
        print("ERROR: The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\nPlease ensure all files are present in the '{path}' directory.")
        return False
    else:
        print("✓ All required files found in the specified directory.")
        return True

# Check if all files exist before proceeding
if not check_files_exist(DATA_PATH, required_files):
    exit(1)

print(f"Loading data from: {os.path.abspath(DATA_PATH)}")

# --- Phase 1: Data Loading and Consolidation ---

try:
    # Load Attendance Data (multiple sheets)
    print("Loading attendance data...")
    attendance_sheets = ['Gym', 'Gym 2', 'Pool', 'Reception Barrier 1', 'Reception Barrier 2', 
                         'Recpetion Barrier 3', 'Reception Barrier 4']
    attendance_dfs = []
    
    for sheet in attendance_sheets:
        try:
            df = pd.read_excel(os.path.join(DATA_PATH, 'Att_Data_25_G24.xlsx'), sheet_name=sheet)
            attendance_dfs.append(df)
            print(f"  ✓ Loaded sheet: {sheet} ({len(df)} rows)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load sheet '{sheet}': {e}")
    
    attendance_df = pd.concat(attendance_dfs, ignore_index=True)
    print(f"  Total attendance records: {len(attendance_df)}")

    # Load Bookings Data (multiple sheets)
    print("\nLoading bookings data...")
    bookings_sheets = ['Squash', 'Classes', 'Alternative sessions', 'Other Activities']
    bookings_dfs = []
    
    for sheet in bookings_sheets:
        try:
            df = pd.read_excel(os.path.join(DATA_PATH, 'Bookings_Data_25_G24.xlsx'), sheet_name=sheet)
            bookings_dfs.append(df)
            print(f"  ✓ Loaded sheet: {sheet} ({len(df)} rows)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load sheet '{sheet}': {e}")
    
    bookings_df = pd.concat(bookings_dfs, ignore_index=True)
    print(f"  Total booking records: {len(bookings_df)}")

    # Load Financial Data (multiple sheets)
    print("\nLoading financial data...")
    financial_sheets = ['S&F', 'Tiverton', 'Non member payments']
    financial_dfs = []
    
    for sheet in financial_sheets:
        try:
            df = pd.read_excel(os.path.join(DATA_PATH, 'Financial_Data_25_G24.xlsx'), sheet_name=sheet)
            financial_dfs.append(df)
            print(f"  ✓ Loaded sheet: {sheet} ({len(df)} rows)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load sheet '{sheet}': {e}")
    
    financial_df = pd.concat(financial_dfs, ignore_index=True)
    print(f"  Total financial records: {len(financial_df)}")

    # Load Cancellation Data
    print("\nLoading cancellation data...")
    cancellation_df = pd.read_excel(os.path.join(DATA_PATH, 'Cancellation_Updated_25_G24.xlsx'))
    print(f"  ✓ Loaded cancellation data ({len(cancellation_df)} rows)")

    # Load NPS Data
    print("\nLoading NPS data...")
    nps_df = pd.read_csv(os.path.join(DATA_PATH, 'NPS_Updated_25_G24.csv'))
    print(f"  ✓ Loaded NPS data ({len(nps_df)} rows)")

except Exception as e:
    print(f"ERROR loading files: {e}")
    print("Please check your file paths and ensure all files are accessible.")
    exit(1)

print("\n" + "="*50)
print("DATA LOADING COMPLETE")
print("="*50)

# --- Phase 2: Data Cleaning and Standardization ---

print("\nPhase 2: Data Cleaning and Standardization")
print("-" * 45)

# Standardize customer identifier column names
def standardize_id(df, possible_names, new_name='unique_id'):
    """Standardize unique identifier column names across DataFrames"""
    for name in possible_names:
        if name in df.columns:
            df.rename(columns={name: new_name}, inplace=True)
            print(f"  ✓ Renamed '{name}' to '{new_name}'")
            break
    return df

print("Standardizing unique ID columns...")
attendance_df = standardize_id(attendance_df, ['Unique Key', 'Unique2', 'Unique ID'])
bookings_df = standardize_id(bookings_df, ['Unique Key', 'Unique2', 'Unique ID'])
financial_df = standardize_id(financial_df, ['Unique Key', 'Unique2', 'Unique ID'])
cancellation_df = standardize_id(cancellation_df, ['Unique Key', 'Unique2', 'Unique ID'])

# Convert unique_id in cancellation_df to integer (remove .0 if present)
if 'unique_id' in cancellation_df.columns and cancellation_df['unique_id'].dtype == float:
    original_count = len(cancellation_df)
    cancellation_df = cancellation_df.dropna(subset=['unique_id'])
    cancellation_df['unique_id'] = cancellation_df['unique_id'].astype(int)
    print(f"  ✓ Converted cancellation unique_id to integer (removed {original_count - len(cancellation_df)} null values)")

# Remove rows with missing unique_id in cancellation_df
if 'unique_id' in cancellation_df.columns:
    original_count = len(cancellation_df)
    cancellation_df = cancellation_df[~cancellation_df['unique_id'].isnull()]
    print(f"  ✓ Removed {original_count - len(cancellation_df)} rows with missing unique_id from cancellation data")

# Handle 'Unknown' in gender columns (keep as category)
print("Cleaning gender columns...")
for df_name, df in [('attendance', attendance_df), ('financial', financial_df)]:
    if 'Contacts Detail Gender' in df.columns:
        df['Contacts Detail Gender'] = df['Contacts Detail Gender'].fillna('Unknown').str.strip()
        print(f"  ✓ Cleaned gender column in {df_name} data")

# Standardize categorical data (example: fix 'Peal' to 'Peak' in cancellation_df's 'Sub title')
print("Standardizing categorical data...")
if 'Sub title' in cancellation_df.columns:
    replacements = cancellation_df['Sub title'].replace({'Peal': 'Peak'})
    changes = (cancellation_df['Sub title'] != replacements).sum()
    cancellation_df['Sub title'] = replacements.str.strip()
    print(f"  ✓ Fixed {changes} 'Peal' -> 'Peak' corrections in cancellation data")

# Trim whitespace from all string columns in all DataFrames
def trim_strings(df, df_name):
    """Remove leading/trailing whitespace from string columns"""
    string_cols = df.select_dtypes(['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
    print(f"  ✓ Trimmed whitespace from {len(string_cols)} string columns in {df_name}")
    return df

attendance_df = trim_strings(attendance_df, 'attendance')
bookings_df = trim_strings(bookings_df, 'bookings')
financial_df = trim_strings(financial_df, 'financial')
cancellation_df = trim_strings(cancellation_df, 'cancellation')
nps_df = trim_strings(nps_df, 'NPS')

# Convert date columns to datetime
def convert_dates(df, columns, df_name):
    """Convert specified columns to datetime format"""
    converted = 0
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            converted += 1
    if converted > 0:
        print(f"  ✓ Converted {converted} date columns in {df_name}")
    return df

print("Converting date columns...")
attendance_df = convert_dates(attendance_df, ['Attendance Detail Date Time'], 'attendance')
bookings_df = convert_dates(bookings_df, ['Bookings Detail Booked Date Time'], 'bookings')
financial_df = convert_dates(financial_df, ['Sales Detail Raised Date'], 'financial')

# Convert numeric columns to appropriate types
print("Converting numeric columns...")
numeric_conversions = [
    (attendance_df, 'Contacts Detail Age', 'attendance'),
    (financial_df, 'Contacts Detail Age', 'financial'),
    (financial_df, 'Sales Detail Gross Amount', 'financial')
]

for df, col, df_name in numeric_conversions:
    if col in df.columns:
        original_nulls = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        new_nulls = df[col].isnull().sum()
        print(f"  ✓ Converted '{col}' to numeric in {df_name} ({new_nulls - original_nulls} conversion errors)")

# Rename ambiguous columns in cancellation_df
print("Renaming ambiguous columns...")
column_renames = {
    'Column4': 'customer_type',
    'Sub title': 'membership_type'
}

for old_name, new_name in column_renames.items():
    if old_name in cancellation_df.columns:
        cancellation_df.rename(columns={old_name: new_name}, inplace=True)
        print(f"  ✓ Renamed '{old_name}' to '{new_name}' in cancellation data")

# --- Phase 3: Filtering and Transaction Processing ---

print("\nPhase 3: Filtering and Transaction Processing")
print("-" * 48)

# Filter financial_df for 'Paid' transactions only
if 'Sales Detail Status' in financial_df.columns:
    original_count = len(financial_df)
    financial_df = financial_df[financial_df['Sales Detail Status'] == 'Paid']
    print(f"  ✓ Filtered to 'Paid' transactions only: {len(financial_df)} of {original_count} records retained")

# Exclude zero or negative gross amount transactions
if 'Sales Detail Gross Amount' in financial_df.columns:
    original_count = len(financial_df)
    financial_df = financial_df[financial_df['Sales Detail Gross Amount'] > 0]
    print(f"  ✓ Excluded zero/negative transactions: {len(financial_df)} of {original_count} records retained")

# --- Phase 4: Data Integration ---

print("\nPhase 4: Data Integration")
print("-" * 27)

# Create customer master table from financial_df (drop duplicate unique_id)
if 'unique_id' in financial_df.columns:
    original_count = len(financial_df)
    customer_master = financial_df.drop_duplicates(subset=['unique_id']).copy()
    print(f"  ✓ Created customer master table: {len(customer_master)} unique customers from {original_count} financial records")
else:
    print("  ⚠ Warning: No unique_id column found in financial data")
    customer_master = financial_df.copy()

# Merge DataFrames using left joins
print("Merging datasets...")
master_df = customer_master.copy()

merge_operations = [
    (attendance_df, 'attendance', '_att'),
    (bookings_df, 'bookings', '_book'),
    (cancellation_df, 'cancellation', '_cancel'),
    (nps_df, 'NPS', '_nps')
]

for df, name, suffix in merge_operations:
    if 'unique_id' in df.columns:
        before_cols = len(master_df.columns)
        master_df = master_df.merge(df, on='unique_id', how='left', suffixes=('', suffix))
        new_cols = len(master_df.columns) - before_cols
        print(f"  ✓ Merged {name} data: added {new_cols} columns")
    else:
        print(f"  ⚠ Warning: Cannot merge {name} data - no unique_id column")

# --- Phase 5: Feature Engineering ---

print("\nPhase 5: Feature Engineering")
print("-" * 30)

# Extract date-based features from attendance
if 'Attendance Detail Date Time' in master_df.columns:
    date_features = ['attendance_month', 'attendance_year', 'attendance_dayofweek', 'attendance_hour']
    master_df['attendance_month'] = master_df['Attendance Detail Date Time'].dt.month
    master_df['attendance_year'] = master_df['Attendance Detail Date Time'].dt.year
    master_df['attendance_dayofweek'] = master_df['Attendance Detail Date Time'].dt.dayofweek
    master_df['attendance_hour'] = master_df['Attendance Detail Date Time'].dt.hour
    print(f"  ✓ Created {len(date_features)} date-based features from attendance data")

# Visit Frequency: average visits per month (using attendance_df)
if 'unique_id' in attendance_df.columns and 'Attendance Detail Date Time' in attendance_df.columns:
    visit_counts = attendance_df.groupby('unique_id')['Attendance Detail Date Time'].count()
    months_active = attendance_df.groupby('unique_id')['Attendance Detail Date Time'].apply(
        lambda x: x.dt.to_period('M').nunique()
    )
    visit_freq = (visit_counts / months_active).rename('avg_visits_per_month')
    master_df = master_df.merge(visit_freq, on='unique_id', how='left')
    print(f"  ✓ Calculated visit frequency for {len(visit_freq)} customers")

# No-Show Rate: percentage of bookings not attended (using bookings_df)
if 'Bookings Detail Attended' in bookings_df.columns and 'unique_id' in bookings_df.columns:
    bookings_df['no_show'] = bookings_df['Bookings Detail Attended'].map({'Yes': 0, 'No': 1})
    no_show_rate = bookings_df.groupby('unique_id')['no_show'].mean().rename('no_show_rate')
    master_df = master_df.merge(no_show_rate, on='unique_id', how='left')
    print(f"  ✓ Calculated no-show rate for {len(no_show_rate)} customers")

# Categorize cancellation reasons (example using regex for common reasons)
def categorize_reason(text):
    """Categorize cancellation reasons based on text content"""
    if pd.isnull(text) or text == 'nan':
        return np.nan
    text = str(text).lower()
    if any(word in text for word in ['cost', 'price', 'expens', 'money', 'afford']):
        return 'Cost'
    if any(word in text for word in ['relocat', 'move', 'moving']):
        return 'Relocation'
    if any(word in text for word in ['facilit', 'equip', 'gym', 'pool']):
        return 'Facilities'
    if any(word in text for word in ['service', 'staff', 'customer']):
        return 'Service'
    if any(word in text for word in ['health', 'ill', 'injury', 'medical']):
        return 'Health'
    if any(word in text for word in ['time', 'busy', 'schedule']):
        return 'Time'
    return 'Other'

if 'Reasoning' in cancellation_df.columns and 'unique_id' in cancellation_df.columns:
    cancellation_df['cancel_reason_category'] = cancellation_df['Reasoning'].apply(categorize_reason)
    cancel_reason_cat = cancellation_df[['unique_id', 'cancel_reason_category']].drop_duplicates()
    master_df = master_df.merge(cancel_reason_cat, on='unique_id', how='left')
    
    # Show distribution of cancellation reasons
    reason_dist = cancellation_df['cancel_reason_category'].value_counts()
    print(f"  ✓ Categorized cancellation reasons for {len(cancel_reason_cat)} customers")
    print("    Reason distribution:")
    for reason, count in reason_dist.items():
        print(f"      {reason}: {count}")

# NPS Categories
if 'Score' in nps_df.columns and 'unique_id' in nps_df.columns:
    def nps_category(score):
        """Categorize NPS scores into Detractor, Passive, Promoter"""
        if pd.isnull(score):
            return np.nan
        try:
            score = float(score)
            if score <= 6:
                return 'Detractor'
            elif score <= 8:
                return 'Passive'
            else:
                return 'Promoter'
        except:
            return np.nan
    
    nps_df['nps_category'] = nps_df['Score'].apply(nps_category)
    nps_cat = nps_df[['unique_id', 'nps_category']].drop_duplicates()
    master_df = master_df.merge(nps_cat, on='unique_id', how='left')
    
    # Show NPS distribution
    nps_dist = nps_df['nps_category'].value_counts()
    print(f"  ✓ Categorized NPS scores for {len(nps_cat)} customers")
    print("    NPS distribution:")
    for category, count in nps_dist.items():
        print(f"      {category}: {count}")

# --- Final Output and Summary ---

print("\n" + "="*50)
print("DATA PROCESSING COMPLETE")
print("="*50)

# Save the cleaned and integrated master DataFrame
output_file = 'master_customer_data.csv'
master_df.to_csv(output_file, index=False)

print(f"\n✓ Master dataset saved to: {output_file}")
print(f"✓ Total customers in master dataset: {len(master_df):,}")
print(f"✓ Total columns in master dataset: {len(master_df.columns)}")

# Display summary statistics
print(f"\nDataset Summary:")
print(f"  - Memory usage: {master_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  - Missing data percentage: {(master_df.isnull().sum().sum() / (len(master_df) * len(master_df.columns)) * 100):.1f}%")

# Show column overview
print(f"\nColumn Overview:")
for i, col in enumerate(master_df.columns, 1):
    non_null = master_df[col].notna().sum()
    print(f"  {i:2d}. {col:<35} ({non_null:,} non-null values)")

print(f"\n🎉 Data integration and feature engineering complete!")
print(f"📁 Output file: {os.path.abspath(output_file)}")