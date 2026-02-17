import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Install required packages if not already installed
# !pip install wordcloud vaderSentiment plotly

warnings.filterwarnings('ignore')

# Set style for elegant visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the master dataset created from data preparation
master_df = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Code/master_customer_data.csv')

print("="*60)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# --- Additional Data Cleaning (items not covered in preparation phase) ---

print("\nAdditional Data Cleaning:")
print("-" * 30)

# Handle anomalous 2025 dates in attendance data
if 'Attendance Detail Date Time' in master_df.columns:
    # Convert to datetime if not already
    master_df['Attendance Detail Date Time'] = pd.to_datetime(master_df['Attendance Detail Date Time'], errors='coerce')
    
    # Identify and handle 2025 dates
    future_dates = master_df['Attendance Detail Date Time'] > datetime.now()
    if future_dates.any():
        print(f"  ✓ Found {future_dates.sum()} anomalous future dates in attendance data")
        # Option 1: Remove future dates
        master_df = master_df[~future_dates]
        print(f"  ✓ Removed {future_dates.sum()} records with future dates")

# Strategy for handling missing data in gender
if 'Contacts Detail Gender' in master_df.columns:
    gender_missing = master_df['Contacts Detail Gender'].isnull().sum()
    print(f"  ✓ Missing gender data: {gender_missing} records")
    
    # Fill missing gender with 'Unknown' if not already done
    master_df['Contacts Detail Gender'] = master_df['Contacts Detail Gender'].fillna('Unknown')

# --- 1. CUSTOMER PROFILES ---

print("\n1. CUSTOMER PROFILES")
print("-" * 20)

# Create figure with subplots for customer profiles
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Profile Analysis', fontsize=20, fontweight='bold', y=0.98)

# Age Distribution Histogram
if 'Contacts Detail Age' in master_df.columns:
    age_data = master_df['Contacts Detail Age'].dropna()
    axes[0,0].hist(age_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution of Members', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Age')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(age_data.mean(), color='red', linestyle='--', 
                     label=f'Mean: {age_data.mean():.1f}')
    axes[0,0].legend()

# Gender Distribution Bar Chart
if 'Contacts Detail Gender' in master_df.columns:
    gender_counts = master_df['Contacts Detail Gender'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = axes[0,1].bar(gender_counts.index, gender_counts.values, 
                        color=colors[:len(gender_counts)])
    axes[0,1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 10,
                      f'{int(height)}', ha='center', va='bottom')

# Membership Type Distribution
membership_cols = [col for col in master_df.columns if 'membership' in col.lower()]
if membership_cols:
    membership_data = master_df[membership_cols[0]].value_counts()
    axes[1,0].pie(membership_data.values, labels=membership_data.index, autopct='%1.1f%%',
                 startangle=90, colors=sns.color_palette("Set3"))
    axes[1,0].set_title('Membership Type Distribution', fontsize=14, fontweight='bold')

# NPS Category Distribution
if 'nps_category' in master_df.columns:
    nps_counts = master_df['nps_category'].value_counts()
    colors = {'Promoter': '#2ECC71', 'Passive': '#F39C12', 'Detractor': '#E74C3C'}
    bar_colors = [colors.get(cat, '#95A5A6') for cat in nps_counts.index]
    
    bars = axes[1,1].bar(nps_counts.index, nps_counts.values, color=bar_colors)
    axes[1,1].set_title('NPS Category Distribution', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Count')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 5,
                      f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# --- 2. USAGE PATTERNS ---

print("\n2. USAGE PATTERNS")
print("-" * 17)

# Create attendance heatmap by day of week and hour
if 'attendance_dayofweek' in master_df.columns and 'attendance_hour' in master_df.columns:
    # Create pivot table for heatmap
    attendance_pivot = master_df.groupby(['attendance_dayofweek', 'attendance_hour']).size().unstack(fill_value=0)
    
    # Create heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(attendance_pivot, 
                annot=True, 
                fmt='d', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Visits'},
                xticklabels=[f'{h:02d}:00' for h in range(24)],
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.title('Facility Attendance Heatmap: Day of Week vs Hour of Day', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.show()

# Peak hours analysis
if 'attendance_hour' in master_df.columns:
    hourly_visits = master_df['attendance_hour'].value_counts().sort_index()
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(hourly_visits.index, hourly_visits.values, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    plt.title('Hourly Attendance Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Visits')
    plt.xticks(range(0, 24, 2))
    
    # ✅ FIXED: Convert peak_hour to integer before indexing
    peak_hour = int(hourly_visits.idxmax())
    bars[peak_hour].set_color('red')
    plt.axvline(peak_hour, color='red', linestyle='--', alpha=0.7, 
                label=f'Peak Hour: {peak_hour}:00')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- 3. SALES TRENDS ---

print("\n3. SALES TRENDS")
print("-" * 15)

# Revenue and transaction trends over time
if 'Sales Detail Raised Date' in master_df.columns and 'Sales Detail Gross Amount' in master_df.columns:
    # Convert date column
    master_df['Sales Detail Raised Date'] = pd.to_datetime(master_df['Sales Detail Raised Date'])
    
    # Group by month for trends
    monthly_sales = master_df.groupby(master_df['Sales Detail Raised Date'].dt.to_period('M')).agg({
        'Sales Detail Gross Amount': ['sum', 'count']
    }).round(2)
    
    monthly_sales.columns = ['Total_Revenue', 'Transaction_Count']
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    # Create subplot for revenue and transaction trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Revenue trend
    ax1.plot(monthly_sales.index, monthly_sales['Total_Revenue'], 
             marker='o', linewidth=2, markersize=6, color='green')
    ax1.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Revenue (£)')
    ax1.grid(True, alpha=0.3)
    
    # Transaction count trend
    ax2.plot(monthly_sales.index, monthly_sales['Transaction_Count'], 
             marker='s', linewidth=2, markersize=6, color='blue')
    ax2.set_title('Monthly Transaction Count Trend', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_xlabel('Month')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- 4. PRICING & MEMBERSHIP ANALYSIS ---

print("\n4. PRICING & MEMBERSHIP ANALYSIS")
print("-" * 33)

# Price distribution by product hierarchy
product_cols = [col for col in master_df.columns if 'product' in col.lower() and 'hierarchy' in col.lower()]
price_level_cols = [col for col in master_df.columns if 'price' in col.lower() and 'level' in col.lower()]

if product_cols and 'Sales Detail Gross Amount' in master_df.columns:
    plt.figure(figsize=(14, 8))
    
    # ✅ FIXED: Clean and validate data before plotting
    # Remove rows with missing product categories or prices
    clean_data = master_df.dropna(subset=[product_cols[0], 'Sales Detail Gross Amount'])
    
    # Filter out zero and negative prices
    clean_data = clean_data[clean_data['Sales Detail Gross Amount'] > 0]
    
    # Remove categories with fewer than 5 data points (optional)
    category_counts = clean_data[product_cols[0]].value_counts()
    valid_categories = category_counts[category_counts >= 5].index
    clean_data = clean_data[clean_data[product_cols[0]].isin(valid_categories)]
    
    if len(clean_data) > 0 and len(clean_data[product_cols[0]].unique()) > 1:
        # Apply outlier filtering only after ensuring we have valid data
        Q1 = clean_data['Sales Detail Gross Amount'].quantile(0.25)
        Q3 = clean_data['Sales Detail Gross Amount'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = clean_data[
            (clean_data['Sales Detail Gross Amount'] >= Q1 - 1.5 * IQR) & 
            (clean_data['Sales Detail Gross Amount'] <= Q3 + 1.5 * IQR)
        ]
        
        # Final check: ensure each category still has data after filtering
        final_category_counts = filtered_data[product_cols[0]].value_counts()
        if len(final_category_counts) > 0 and final_category_counts.min() > 0:
            try:
                sns.boxplot(data=filtered_data, x=product_cols[0], y='Sales Detail Gross Amount')
                plt.title('Price Distribution by Product Category', fontsize=16, fontweight='bold')
                plt.xlabel('Product Category')
                plt.ylabel('Price (£)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                # Print summary statistics
                print(f"  ✓ Boxplot created for {len(final_category_counts)} product categories")
                print(f"  ✓ Total data points plotted: {len(filtered_data)}")
                
            except Exception as e:
                print(f"  ⚠ Error creating boxplot: {e}")
                print("  → Falling back to alternative visualization...")
                
                # Alternative: Simple bar chart of average prices
                avg_prices = filtered_data.groupby(product_cols[0])['Sales Detail Gross Amount'].mean()
                plt.figure(figsize=(12, 6))
                bars = plt.bar(avg_prices.index, avg_prices.values, color='skyblue', alpha=0.7)
                plt.title('Average Price by Product Category', fontsize=16, fontweight='bold')
                plt.xlabel('Product Category')
                plt.ylabel('Average Price (£)')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'£{height:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.show()
        else:
            print("  ⚠ Warning: Insufficient data for boxplot after filtering")
            print("  → Skipping price distribution visualization")
    else:
        print("  ⚠ Warning: No valid product/price data found for analysis")
else:
    print("  ⚠ Warning: Required columns not found for pricing analysis")


# --- 5. DEMOGRAPHICS & USAGE SEGMENTATION ---

print("\n5. DEMOGRAPHICS & USAGE SEGMENTATION")
print("-" * 37)

# Age group analysis with booking activities
if 'Contacts Detail Age' in master_df.columns:
    # Create age groups
    master_df['age_group'] = pd.cut(master_df['Contacts Detail Age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Activity group analysis by age
    activity_cols = [col for col in master_df.columns if 'activity' in col.lower() and 'group' in col.lower()]
    
    if activity_cols:
        activity_age_crosstab = pd.crosstab(master_df['age_group'], 
                                           master_df[activity_cols[0]], 
                                           normalize='index') * 100
        
        plt.figure(figsize=(12, 8))
        activity_age_crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Activity Preferences by Age Group (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Age Group')
        plt.ylabel('Percentage')
        plt.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# --- 6. SENTIMENT ANALYSIS ---

print("\n6. SENTIMENT ANALYSIS")
print("-" * 21)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze cancellation reasons for sentiment
if 'Reasoning' in master_df.columns:
    # Clean and prepare text data
    text_data = master_df['Reasoning'].dropna().astype(str)
    text_data = text_data[text_data != 'nan']
    
    if len(text_data) > 0:
        # Generate Word Cloud
        all_text = ' '.join(text_data.values)
        
        plt.figure(figsize=(12, 6))
        
        # Word Cloud
        plt.subplot(1, 2, 1)
        wordcloud = WordCloud(width=400, height=300, 
                             background_color='white',
                             colormap='viridis').generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: Cancellation Reasons', fontsize=14, fontweight='bold')
        
        # Sentiment Analysis
        sentiments = []
        for text in text_data:
            score = analyzer.polarity_scores(text)
            sentiments.append(score['compound'])
        
        # Sentiment Distribution
        plt.subplot(1, 2, 2)
        plt.hist(sentiments, bins=20, alpha=0.7, color='coral', edgecolor='black')
        plt.axvline(np.mean(sentiments), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sentiments):.3f}')
        plt.title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Score (-1 to 1)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Sentiment categories
        sentiment_categories = []
        for score in sentiments:
            if score >= 0.05:
                sentiment_categories.append('Positive')
            elif score <= -0.05:
                sentiment_categories.append('Negative')
            else:
                sentiment_categories.append('Neutral')
        
        sentiment_counts = pd.Series(sentiment_categories).value_counts()
        
        plt.figure(figsize=(8, 6))
        colors = {'Positive': '#2ECC71', 'Neutral': '#F39C12', 'Negative': '#E74C3C'}
        bar_colors = [colors.get(cat, '#95A5A6') for cat in sentiment_counts.index]
        
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
        plt.title('Sentiment Categories Distribution', fontsize=16, fontweight='bold')
        plt.ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# --- 7. SUMMARY STATISTICS ---

print("\n7. SUMMARY STATISTICS")
print("-" * 21)

# Key metrics summary
summary_stats = {}

if 'Contacts Detail Age' in master_df.columns:
    age_data = master_df['Contacts Detail Age'].dropna()
    summary_stats['Average Age'] = f"{age_data.mean():.1f} years"
    summary_stats['Age Range'] = f"{age_data.min():.0f} - {age_data.max():.0f} years"

if 'Sales Detail Gross Amount' in master_df.columns:
    revenue_data = master_df['Sales Detail Gross Amount'].dropna()
    summary_stats['Total Revenue'] = f"£{revenue_data.sum():,.2f}"
    summary_stats['Average Transaction'] = f"£{revenue_data.mean():.2f}"

if 'avg_visits_per_month' in master_df.columns:
    visit_data = master_df['avg_visits_per_month'].dropna()
    summary_stats['Avg Visits/Month'] = f"{visit_data.mean():.1f}"

if 'no_show_rate' in master_df.columns:
    no_show_data = master_df['no_show_rate'].dropna()
    summary_stats['Average No-Show Rate'] = f"{no_show_data.mean()*100:.1f}%"

# Display summary in a formatted table
print("\nKEY INSIGHTS SUMMARY:")
print("=" * 40)
for metric, value in summary_stats.items():
    print(f"{metric:<25}: {value}")

print("\n" + "="*60)
print("EDA COMPLETE - All visualizations generated successfully!")
print("="*60)

# Save processed data with additional features
master_df.to_csv('master_customer_data_with_eda_features.csv', index=False)
print(f"\n✓ Enhanced dataset saved to: master_customer_data_with_eda_features.csv")
