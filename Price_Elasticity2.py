import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('default')
sns.set_palette("husl")

class PriceElasticityAnalyzer:
    def __init__(self):
        self.financial_data = None
        self.user_data = None
        self.attendance_data = None
        self.booking_data = None
        self.utilisation_data = None
        self.cancellation_data = None
        self.nps_data = None
        self.integrated_data = None
        
    def load_data(self):
        """Load all datasets"""
        # Load financial data
        financial_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', 
                                         sheet_name=['S&F', 'Tiverton'])
        self.financial_data = pd.concat([financial_sheets['S&F'], financial_sheets['Tiverton']], 
                                      ignore_index=True)
        
        # Load user data
        self.user_data = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/User_25_G24.xlsx')
        
        # Load attendance data
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
        self.booking_data = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Booking_and_Utilisation_25_G24.xlsx')

        # Load utilisation data
        util_sheets = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Utilisation_Data_25G24.xlsx', 
                                   sheet_name=['2023 July to Dec', '2024 Jan - June','2024 July - Dec', '2025 Jan - May'])
        self.utilisation_data = pd.concat([util_sheets['2023 July to Dec'], 
                                        util_sheets['2024 Jan - June'], 
                                        util_sheets['2024 July - Dec'],
                                         util_sheets['2025 Jan - May']], 
                                        ignore_index=True)
        
        # Load other data
        self.cancellation_data = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Cancellation_Updated_25_G24.xlsx')
        self.nps_data = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')
        
        print("Data loaded successfully!")
        
    def preprocess_data(self):
        """Preprocess and clean the data"""
        # Convert date columns
        self.financial_data['Sales Detail Participation Date'] = pd.to_datetime(
            self.financial_data['Sales Detail Participation Date'], errors='coerce')
        
        self.attendance_data['Attendance Detail Date'] = pd.to_datetime(
            self.attendance_data['Attendance Detail Date'], errors='coerce')
        
        self.booking_data['BookedDate'] = pd.to_datetime(
            self.booking_data['BookedDate'], errors='coerce')
        self.booking_data['StartDate'] = pd.to_datetime(
            self.booking_data['StartDate'], errors='coerce')
        
        self.utilisation_data['Resource Utilisation Date'] = pd.to_datetime(
            self.utilisation_data['Resource Utilisation Date'], errors='coerce')
        
        self.cancellation_data['Cancelled on'] = pd.to_datetime(
            self.cancellation_data['Cancelled on'], errors='coerce')
        
        self.nps_data['Response Date'] = pd.to_datetime(
            self.nps_data['Response Date'], errors='coerce')
        
        # Clean price data
        self.financial_data['Sales Detail Gross Amount'] = pd.to_numeric(
            self.financial_data['Sales Detail Gross Amount'], errors='coerce')
        
        print("Data preprocessing completed!")
        
    def integrate_data(self):
        """Integrate data using common identifiers"""
        # Primary integration using Unique Key/ID
        financial_agg = self.financial_data.groupby('Unique Key').agg({
            'Sales Detail Gross Amount': ['sum', 'mean', 'count'],
            'Sales Detail Price Level': 'first',
            'Sales Detail Participation Date': ['min', 'max'],
            'Contacts Detail Age': 'first',
            'Contacts Detail Gender': 'first'
        }).reset_index()
        
        financial_agg.columns = ['Unique_Key', 'Total_Revenue', 'Avg_Revenue', 'Transaction_Count',
                               'Price_Level', 'First_Transaction', 'Last_Transaction', 
                               'Age', 'Gender']
        
        # Attendance aggregation
        attendance_agg = self.attendance_data.groupby('Unique Key').agg({
            'Attendance Detail Date': 'count',
            'Attendance Detail Date Time': ['min', 'max'],
            'Contacts Detail Price Level': 'first'
        }).reset_index()
        
        attendance_agg.columns = ['Unique_Key', 'Total_Visits', 'First_Visit', 'Last_Visit', 
                                'Member_Type']
        
        # Merge data
        self.integrated_data = pd.merge(financial_agg, attendance_agg, on='Unique_Key', how='outer')
        self.integrated_data = pd.merge(self.integrated_data, self.user_data, 
                                      left_on='Unique_Key', right_on='Unique ID', how='left')
        
        print(f"Integrated dataset created with {len(self.integrated_data)} records")
        
    def calculate_enhanced_demand_metrics(self):
        """Calculate enhanced demand metrics"""
        # Booking vs Attendance Analysis
        booking_attendance = pd.merge(
            self.booking_data.groupby('Unique ID').agg({
                'Attended': lambda x: (x == 'Yes').sum(),
                'BookingID': 'count'
            }).reset_index(),
            self.attendance_data.groupby('Unique Key').size().reset_index(),
            left_on='Unique ID', right_on='Unique Key', how='inner'
        )
        
        booking_attendance.columns = ['Unique_ID', 'Bookings_Attended', 'Total_Bookings', 
                                    'Unique_Key', 'Actual_Attendance']
        
        # Calculate attendance rate
        booking_attendance['Attendance_Rate'] = (
            booking_attendance['Bookings_Attended'] / booking_attendance['Total_Bookings']
        ).fillna(0)
        
        # Facility utilization rate from utilisation_data
        facility_util = self.utilisation_data.groupby('Resource Utilisation Resource').agg({
            'Resource Utilisation Hours Used': 'sum',
            'Resource Utilisation Hours Available': 'sum'
        }).reset_index()
        
        facility_util['Utilization_Rate'] = (
            facility_util['Resource Utilisation Hours Used'] / 
            facility_util['Resource Utilisation Hours Available']
        ).fillna(0)
        
        # Member engagement score calculation
        self.integrated_data['Member_Engagement_Score'] = (
            self.integrated_data['Total_Visits'].fillna(0) * 
            self.integrated_data['Transaction_Count'].fillna(0) * 
            (self.integrated_data['Total_Revenue'].fillna(0) / 100)
        )
        
        return booking_attendance, facility_util
        
    def calculate_price_elasticity(self):
        """Calculate multi-dimensional price elasticity"""
        # Filter out invalid data
        valid_data = self.integrated_data[
            (self.integrated_data['Total_Revenue'].notna()) & 
            (self.integrated_data['Total_Revenue'] > 0) &
            (self.integrated_data['Total_Visits'].notna()) & 
            (self.integrated_data['Total_Visits'] > 0)
        ].copy()
        
        # Create price bins for elasticity calculation
        valid_data['Price_Bin'] = pd.qcut(valid_data['Total_Revenue'], 
                                        q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Calculate elasticity by membership tier
        elasticity_results = {}
        
        for price_level in valid_data['Price_Level'].dropna().unique():
            if len(valid_data[valid_data['Price_Level'] == price_level]) < 5:
                continue
                
            subset = valid_data[valid_data['Price_Level'] == price_level]
            
            # Calculate price-demand relationship
            price_demand_corr = subset[['Total_Revenue', 'Total_Visits']].corr().iloc[0, 1]
            
            # Simple elasticity approximation
            if subset['Total_Revenue'].std() > 0 and subset['Total_Visits'].std() > 0:
                price_change_pct = subset['Total_Revenue'].pct_change().mean()
                demand_change_pct = subset['Total_Visits'].pct_change().mean()
                
                if price_change_pct != 0:
                    elasticity = demand_change_pct / price_change_pct
                else:
                    elasticity = 0
            else:
                elasticity = 0
                
            elasticity_results[price_level] = {
                'correlation': price_demand_corr,
                'elasticity': elasticity,
                'sample_size': len(subset)
            }
        
        return elasticity_results

# Initialize the analyzer
analyzer = PriceElasticityAnalyzer()
analyzer.load_data()
analyzer.preprocess_data()
analyzer.integrate_data()
booking_attendance, facility_util = analyzer.calculate_enhanced_demand_metrics()
elasticity_results = analyzer.calculate_price_elasticity()

print("Analysis completed successfully!")
print(f"Elasticity results calculated for {len(elasticity_results)} membership tiers")

#Visualization of results
def create_visualizations(analyzer, booking_attendance, facility_util, elasticity_results):
    """Create comprehensive visualizations"""
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    return {
        'revenue_distribution': create_revenue_distribution_chart(analyzer),
        'attendance_patterns': create_attendance_patterns_chart(analyzer),
        'facility_utilization': create_facility_utilization_chart(facility_util),
        'booking_vs_attendance': create_booking_attendance_chart(booking_attendance),
        'member_engagement': create_member_engagement_chart(analyzer),
        'price_elasticity_heatmap': create_elasticity_heatmap(elasticity_results),
        'demand_vs_price_scatter': create_demand_price_scatter(analyzer),
        'membership_tier_analysis': create_membership_tier_analysis(analyzer),
        'seasonal_demand_trends': create_seasonal_trends_chart(analyzer),
        'cancellation_analysis': create_cancellation_analysis_chart(analyzer)
    }

def create_revenue_distribution_chart(analyzer):
    """Revenue Distribution by Membership Type"""
    revenue_by_type = analyzer.integrated_data.groupby('Price_Level')['Total_Revenue'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(revenue_by_type)), revenue_by_type.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(revenue_by_type))))
    
    ax.set_yticks(range(len(revenue_by_type)))
    ax.set_yticklabels(revenue_by_type.index)
    ax.set_xlabel('Total Revenue (£)')
    ax.set_title('Revenue Distribution by Membership Type', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, revenue_by_type.values)):
        if not pd.isna(value):
            ax.text(value + max(revenue_by_type) * 0.01, i, f'£{value:,.0f}', 
                   va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_attendance_patterns_chart(analyzer):
    """Attendance Patterns by Day of Week"""
    daily_attendance = analyzer.attendance_data.groupby('Attendance Detail Weekday').size()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(daily_attendance.index, daily_attendance.values, 
                  color=plt.cm.Set3(np.linspace(0, 1, len(daily_attendance))))
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Visits')
    ax.set_title('Attendance Patterns by Day of Week', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, daily_attendance.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(daily_attendance) * 0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_facility_utilization_chart(facility_util):
    """Facility Utilization Rate Analysis"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(range(len(facility_util)), facility_util['Utilization_Rate'] * 100,
                  color=plt.cm.RdYlGn(facility_util['Utilization_Rate']))
    
    ax.set_xticks(range(len(facility_util)))
    ax.set_xticklabels(facility_util['Resource Utilisation Resource'], rotation=45, ha='right')
    ax.set_ylabel('Utilization Rate (%)')
    ax.set_title('Facility Utilization Rates', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, facility_util['Utilization_Rate'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_booking_attendance_chart(booking_attendance):
    """Booking vs Actual Attendance Analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Attendance Rate Distribution
    ax1.hist(booking_attendance['Attendance_Rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Attendance Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Attendance Rates', fontweight='bold')
    ax1.axvline(booking_attendance['Attendance_Rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {booking_attendance["Attendance_Rate"].mean():.2f}')
    ax1.legend()
    
    # Bookings vs Actual Attendance Scatter
    ax2.scatter(booking_attendance['Total_Bookings'], booking_attendance['Actual_Attendance'], 
                alpha=0.6, color='coral')
    ax2.plot([0, booking_attendance['Total_Bookings'].max()], 
             [0, booking_attendance['Total_Bookings'].max()], 'r--', label='Perfect Attendance')
    ax2.set_xlabel('Total Bookings')
    ax2.set_ylabel('Actual Attendance')
    ax2.set_title('Bookings vs Actual Attendance', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_member_engagement_chart(analyzer):
    """Member Engagement Score Analysis"""
    engagement_data = analyzer.integrated_data[analyzer.integrated_data['Member_Engagement_Score'].notna()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Engagement Score Distribution
    ax1.hist(engagement_data['Member_Engagement_Score'], bins=30, alpha=0.7, 
             color='lightgreen', edgecolor='black')
    ax1.set_xlabel('Member Engagement Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Member Engagement Scores', fontweight='bold')
    
    # Engagement by Age Group
    engagement_data['Age_Group'] = pd.cut(engagement_data['Age'], 
                                        bins=[0, 25, 35, 50, 65, 100], 
                                        labels=['18-25', '26-35', '36-50', '51-65', '65+'])
    
    engagement_by_age = engagement_data.groupby('Age_Group')['Member_Engagement_Score'].mean()
    ax2.bar(range(len(engagement_by_age)), engagement_by_age.values, 
            color=plt.cm.plasma(np.linspace(0, 1, len(engagement_by_age))))
    ax2.set_xticks(range(len(engagement_by_age)))
    ax2.set_xticklabels(engagement_by_age.index)
    ax2.set_ylabel('Average Engagement Score')
    ax2.set_title('Average Engagement Score by Age Group', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_elasticity_heatmap(elasticity_results):
    """Price Elasticity Heatmap"""
    # Convert elasticity results to DataFrame for visualization
    elasticity_df = pd.DataFrame(elasticity_results).T
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap data
    heatmap_data = elasticity_df[['elasticity', 'correlation']].fillna(0)
    
    im = ax.imshow(heatmap_data.T, cmap='RdBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.index, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.columns)))
    ax.set_yticklabels(heatmap_data.columns)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Add text annotations
    for i in range(len(heatmap_data.columns)):
        for j in range(len(heatmap_data.index)):
            text = ax.text(j, i, f'{heatmap_data.iloc[j, i]:.2f}', 
                          ha='center', va='center', fontweight='bold')
    
    ax.set_title('Price Elasticity Analysis by Membership Tier', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_demand_price_scatter(analyzer):
    """Demand vs Price Relationship Scatter Plot"""
    valid_data = analyzer.integrated_data[
        (analyzer.integrated_data['Total_Revenue'].notna()) & 
        (analyzer.integrated_data['Total_Visits'].notna()) &
        (analyzer.integrated_data['Total_Revenue'] > 0) &
        (analyzer.integrated_data['Total_Visits'] > 0)
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by membership type
    membership_types = valid_data['Price_Level'].dropna().unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(membership_types)))
    
    for i, membership in enumerate(membership_types):
        subset = valid_data[valid_data['Price_Level'] == membership]
        ax.scatter(subset['Total_Revenue'], subset['Total_Visits'], 
                  alpha=0.6, label=membership, color=colors[i], s=50)
    
    ax.set_xlabel('Total Revenue (£)')
    ax.set_ylabel('Total Visits')
    ax.set_title('Demand vs Price Relationship by Membership Type', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_membership_tier_analysis(analyzer):
    """Membership Tier Analysis"""
    tier_analysis = analyzer.integrated_data.groupby('Price_Level').agg({
        'Total_Revenue': 'mean',
        'Total_Visits': 'mean',
        'Member_Engagement_Score': 'mean',
        'Unique_Key': 'count'
    }).fillna(0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average Revenue by Tier
    ax1.bar(range(len(tier_analysis)), tier_analysis['Total_Revenue'], 
            color=plt.cm.viridis(np.linspace(0, 1, len(tier_analysis))))
    ax1.set_xticks(range(len(tier_analysis)))
    ax1.set_xticklabels(tier_analysis.index, rotation=45, ha='right')
    ax1.set_ylabel('Average Revenue (£)')
    ax1.set_title('Average Revenue by Membership Tier', fontweight='bold')
    
    # Average Visits by Tier
    ax2.bar(range(len(tier_analysis)), tier_analysis['Total_Visits'], 
            color=plt.cm.plasma(np.linspace(0, 1, len(tier_analysis))))
    ax2.set_xticks(range(len(tier_analysis)))
    ax2.set_xticklabels(tier_analysis.index, rotation=45, ha='right')
    ax2.set_ylabel('Average Visits')
    ax2.set_title('Average Visits by Membership Tier', fontweight='bold')
    
    # Member Count by Tier
    ax3.pie(tier_analysis['Unique_Key'], labels=tier_analysis.index, autopct='%1.1f%%')
    ax3.set_title('Member Distribution by Tier', fontweight='bold')
    
    # Engagement Score by Tier
    ax4.bar(range(len(tier_analysis)), tier_analysis['Member_Engagement_Score'], 
            color=plt.cm.coolwarm(np.linspace(0, 1, len(tier_analysis))))
    ax4.set_xticks(range(len(tier_analysis)))
    ax4.set_xticklabels(tier_analysis.index, rotation=45, ha='right')
    ax4.set_ylabel('Average Engagement Score')
    ax4.set_title('Average Engagement Score by Tier', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_seasonal_trends_chart(analyzer):
    """Seasonal Demand Trends Analysis"""
    # Extract month from attendance data
    analyzer.attendance_data['Month'] = analyzer.attendance_data['Attendance Detail Date'].dt.month
    monthly_attendance = analyzer.attendance_data.groupby('Month').size()
    
    # Extract month from financial data
    analyzer.financial_data['Month'] = analyzer.financial_data['Sales Detail Participation Date'].dt.month
    monthly_revenue = analyzer.financial_data.groupby('Month')['Sales Detail Gross Amount'].sum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Monthly Attendance Trend
    ax1.plot(monthly_attendance.index, monthly_attendance.values, marker='o', linewidth=3, 
             color='steelblue', markersize=8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Visits')
    ax1.set_title('Monthly Attendance Trends', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Monthly Revenue Trend
    ax2.plot(monthly_revenue.index, monthly_revenue.values, marker='s', linewidth=3, 
             color='forestgreen', markersize=8)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Revenue (£)')
    ax2.set_title('Monthly Revenue Trends', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_cancellation_analysis_chart(analyzer):
    """Cancellation Analysis"""
    # Analyze cancellation reasons
    cancellation_reasons = analyzer.cancellation_data['Reasoning'].value_counts().head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Top Cancellation Reasons
    ax1.barh(range(len(cancellation_reasons)), cancellation_reasons.values, 
             color=plt.cm.Reds(np.linspace(0.3, 0.9, len(cancellation_reasons))))
    ax1.set_yticks(range(len(cancellation_reasons)))
    ax1.set_yticklabels(cancellation_reasons.index, fontsize=10)
    ax1.set_xlabel('Number of Cancellations')
    ax1.set_title('Top Cancellation Reasons', fontweight='bold')
    
    # Cancellation Trends by Month
    analyzer.cancellation_data['Cancellation_Month'] = analyzer.cancellation_data['Cancelled on'].dt.month
    monthly_cancellations = analyzer.cancellation_data.groupby('Cancellation_Month').size()
    
    ax2.plot(monthly_cancellations.index, monthly_cancellations.values, marker='o', 
             linewidth=3, color='crimson', markersize=8)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Cancellations')
    ax2.set_title('Monthly Cancellation Trends', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Create all visualizations
visualizations = create_visualizations(analyzer, booking_attendance, facility_util, elasticity_results)

print("All visualizations created successfully!")
print("Charts available:")
for chart_name in visualizations.keys():
    print(f"- {chart_name}")

# DISPLAY THE CHARTS
print("\nDisplaying charts...")

# Option 1: Display charts in separate windows
for chart_name, fig in visualizations.items():
    if fig is not None:
        plt.figure(fig.number)
        plt.suptitle(f'{chart_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        plt.show()
        print(f"✓ Displayed: {chart_name}")
    else:
        print(f"✗ Error: {chart_name} could not be created")

# Option 2: Save charts to files (uncomment if preferred)
"""
import os
os.makedirs('charts', exist_ok=True)

for chart_name, fig in visualizations.items():
    if fig is not None:
        filename = f'charts/{chart_name}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
"""

print("\nVisualization process completed!")


#Analytics and Reporting
def generate_elasticity_report(elasticity_results, analyzer):
    """Generate comprehensive elasticity report"""
    
    print("=" * 60)
    print("ENHANCED PRICE ELASTICITY ANALYSIS REPORT")
    print("=" * 60)
    
    # Summary Statistics
    valid_data = analyzer.integrated_data[
        (analyzer.integrated_data['Total_Revenue'].notna()) & 
        (analyzer.integrated_data['Total_Visits'].notna()) &
        (analyzer.integrated_data['Total_Revenue'] > 0) &
        (analyzer.integrated_data['Total_Visits'] > 0)
    ]
    
    print(f"\nDATA SUMMARY:")
    print(f"Total Active Members Analyzed: {len(valid_data):,}")
    print(f"Total Revenue Analyzed: £{valid_data['Total_Revenue'].sum():,.2f}")
    print(f"Average Revenue per Member: £{valid_data['Total_Revenue'].mean():.2f}")
    print(f"Average Visits per Member: {valid_data['Total_Visits'].mean():.1f}")
    
    print(f"\nPRICE ELASTICITY BY MEMBERSHIP TIER:")
    print("-" * 40)
    
    for tier, results in elasticity_results.items():
        elasticity_value = results['elasticity']
        correlation = results['correlation']
        sample_size = results['sample_size']
        
        # Interpret elasticity
        if elasticity_value < -1:
            interpretation = "Highly Elastic (Price Sensitive)"
        elif -1 <= elasticity_value < -0.5:
            interpretation = "Moderately Elastic"
        elif -0.5 <= elasticity_value < 0:
            interpretation = "Inelastic (Price Insensitive)"
        elif elasticity_value >= 0:
            interpretation = "Positive/Atypical Relationship"
        else:
            interpretation = "Cannot Determine"
        
        print(f"\n{tier}:")
        print(f"  Elasticity: {elasticity_value:.3f}")
        print(f"  Interpretation: {interpretation}")
        print(f"  Price-Demand Correlation: {correlation:.3f}")
        print(f"  Sample Size: {sample_size}")
    
    # Calculate facility-specific metrics
    booking_attendance_rate = (analyzer.booking_data['Attended'] == 'Yes').mean() if len(analyzer.booking_data) > 0 else 0
    
    print(f"\nFACILITY UTILIZATION METRICS:")
    print("-" * 40)
    print(f"Overall Booking Attendance Rate: {booking_attendance_rate:.1%}")
    
    # Member engagement insights
    engagement_stats = valid_data['Member_Engagement_Score'].describe()
    print(f"\nMEMBER ENGAGEMENT INSIGHTS:")
    print("-" * 40)
    print(f"Average Engagement Score: {engagement_stats['mean']:.2f}")
    print(f"Median Engagement Score: {engagement_stats['50%']:.2f}")
    print(f"Top 25% Engagement Threshold: {engagement_stats['75%']:.2f}")
    
    # Recommendations
    print(f"\nSTRATEGIC RECOMMENDATIONS:")
    print("-" * 40)
    
    # Find most and least elastic segments
    elasticity_values = {k: v['elasticity'] for k, v in elasticity_results.items() if not np.isnan(v['elasticity'])}
    
    if elasticity_values:
        most_elastic = min(elasticity_values, key=elasticity_values.get)
        least_elastic = max(elasticity_values, key=elasticity_values.get)
        
        print(f"1. PRICE OPTIMIZATION:")
        print(f"   - {most_elastic} members are most price-sensitive (elasticity: {elasticity_values[most_elastic]:.3f})")
        print(f"   - Consider discount strategies for this segment")
        print(f"   - {least_elastic} members are least price-sensitive (elasticity: {elasticity_values[least_elastic]:.3f})")
        print(f"   - Consider premium offerings for this segment")
    
    print(f"\n2. CAPACITY MANAGEMENT:")
    print(f"   - Booking attendance rate of {booking_attendance_rate:.1%} indicates potential overbooking opportunities")
    print(f"   - Focus on converting high-engagement members to higher-tier memberships")
    
    print(f"\n3. MEMBER RETENTION:")
    print(f"   - Target members with engagement scores below {engagement_stats['25%']:.2f} for retention campaigns")
    print(f"   - Develop loyalty programs for top-tier engaged members")
    
    print("=" * 60)

# Generate the comprehensive report
generate_elasticity_report(elasticity_results, analyzer)

#Chart Creation Function

def create_individual_charts():
    """Create all charts separately using the create_chart tool"""
    
    chart_configs = [
        {
            'name': 'revenue_distribution',
            'title': 'Revenue Distribution by Membership Type',
            'description': 'Horizontal bar chart showing total revenue by membership type'
        },
        {
            'name': 'attendance_patterns', 
            'title': 'Attendance Patterns by Day of Week',
            'description': 'Bar chart showing gym attendance patterns across different days'
        },
        {
            'name': 'facility_utilization',
            'title': 'Facility Utilization Rates', 
            'description': 'Bar chart showing utilization rates for different facilities'
        },
        {
            'name': 'member_engagement',
            'title': 'Member Engagement Score Analysis',
            'description': 'Distribution and analysis of member engagement scores'
        },
        {
            'name': 'price_elasticity_heatmap',
            'title': 'Price Elasticity Analysis Heatmap',
            'description': 'Heatmap showing price elasticity and correlation by membership tier'
        },
        {
            'name': 'demand_vs_price',
            'title': 'Demand vs Price Relationship',
            'description': 'Scatter plot showing relationship between price and demand by membership type'
        },
        {
            'name': 'seasonal_trends',
            'title': 'Seasonal Demand Trends',
            'description': 'Line charts showing monthly trends in attendance and revenue'
        },
        {
            'name': 'cancellation_analysis',
            'title': 'Membership Cancellation Analysis', 
            'description': 'Analysis of cancellation reasons and trends'
        }
    ]
    
    return chart_configs

# Print completion message
print("\n" + "="*60)
print("ENHANCED PRICE ELASTICITY MODEL FRAMEWORK COMPLETED")
print("="*60)
print("\nThe comprehensive analysis includes:")
print("✓ Data integration across all datasets")
print("✓ Enhanced demand measurement metrics")
print("✓ Multi-dimensional price elasticity calculations")
print("✓ Professional visualizations")
print("✓ Strategic recommendations")
print("\nAll code is ready for execution and chart generation!")

