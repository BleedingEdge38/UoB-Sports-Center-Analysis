from operator import attrgetter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SportscentreAnalytics:
    def __init__(self):
        """Initialize the analytics class and loading all datasets"""
        self.load_datasets()
        self.prepare_data()
    
    def load_datasets(self):
        """Load all required datasets from Excel files"""
        # Load Attendance data
        self.attendance_gym = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Gym')
        self.attendance_gym2 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Gym 2')
        self.attendance_pool = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Pool')
        self.attendance_reception1 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 1')
        self.attendance_reception2 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 2')
        self.attendance_reception3 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Recpetion Barrier 3')
        self.attendance_reception4 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 4')
        
        # Load Bookings data
        self.bookings_squash = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Squash')
        self.bookings_classes = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Classes')
        self.bookings_alt_sessions = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Alternative sessions')
        self.bookings_other_activities = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Other Activities')

        # Load Financial data
        self.financial_sf = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='S&F')
        self.financial_tiverton = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='Tiverton')
        self.financial_non_member = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='Non member payments')

        # Load Cancellation and NPS data
        self.cancellations = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Cancellation_Updated_25_G24.xlsx', sheet_name='Sheet1')
        self.nps = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')

    def prepare_data(self):
        """Clean and prepare datasets for analysis"""
        # Standardize column names for attendance data
        attendance_datasets = [
            self.attendance_gym, self.attendance_gym2, 
            self.attendance_pool, self.attendance_reception1, self.attendance_reception2,
            self.attendance_reception3, self.attendance_reception4
            ]
        
        for df in attendance_datasets:
            df.rename(columns={
                'Unique Key': 'unique_key',
                'Attendance Detail Date': 'date',
                'Attendance Detail Date Time': 'date_time',
                'Attendance Detail Entry Point': 'entry_point',
                'Attendance Detail Weekday': 'weekday',
                'Contacts Detail Age': 'age',
                'Contacts Detail Gender': 'gender',
                'Contacts Detail Price Level': 'price_level'
            }, inplace=True)
        
        # Add facility type to each dataset
        self.attendance_gym['facility_type'] = 'Gym'
        self.attendance_gym2['facility_type'] = 'Gym_2'
        self.attendance_pool['facility_type'] = 'Pool'
        self.attendance_reception1['facility_type'] = 'Reception_1'
        self.attendance_reception2['facility_type'] = 'Reception_2'
        self.attendance_reception3['facility_type'] = 'Reception_3'
        self.attendance_reception4['facility_type'] = 'Reception_4'

        # Combine all attendance data
        self.attendance = pd.concat([
            self.attendance_gym, self.attendance_gym2, 
            self.attendance_pool, self.attendance_reception1,
            self.attendance_reception2, self.attendance_reception3,
            self.attendance_reception4
        ], ignore_index=True)
        
        # Convert date columns
        self.attendance['date'] = pd.to_datetime(self.attendance['date'])
        self.attendance['date_time'] = pd.to_datetime(self.attendance['date_time'])
        
        # Prepare financial data
        self.financial = pd.concat([
            self.financial_sf, self.financial_tiverton, self.financial_non_member
        ], ignore_index=True)
        
        # Clean financial column names
        self.financial.columns = self.financial.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Prepare cancellation data
        self.cancellations.columns = self.cancellations.columns.str.strip().str.lower().str.replace(' ', '_')
        self.cancellations['effective_from'] = pd.to_datetime(self.cancellations['effective_from'], errors='coerce')
        
        # Prepare NPS data
        self.nps['Response Date'] = pd.to_datetime(self.nps['Response Date'], format='%d/%m/%Y')

    def membership_tier_performance_analysis(self):
        """Perform comprehensive membership tier performance analysis"""
        
        # 1. Price Level Distribution Analysis
        price_level_counts = self.attendance['price_level'].value_counts().reset_index()
        price_level_counts.columns = ['price_level', 'attendance_count']
        
        # Calculate revenue by price level from financial data
        if 'sales_detail_price_level' in self.financial.columns:
            revenue_by_tier = self.financial.groupby('sales_detail_price_level')['sales_detail_gross_amount'].sum().reset_index()
            revenue_by_tier.columns = ['price_level', 'total_revenue']
        else:
            # Alternative approach if column name is different
            revenue_cols = [col for col in self.financial.columns if 'price' in col.lower()]
            amount_cols = [col for col in self.financial.columns if 'amount' in col.lower() or 'gross' in col.lower()]
            
            if revenue_cols and amount_cols:
                revenue_by_tier = self.financial.groupby(revenue_cols[0])[amount_cols[0]].sum().reset_index()
                revenue_by_tier.columns = ['price_level', 'total_revenue']
            else:
                revenue_by_tier = pd.DataFrame({'price_level': [], 'total_revenue': []})
        
        # 2. Age-Price Level Cross-Analysis
        age_price_analysis = self.attendance.groupby(['price_level', 'age']).size().reset_index(name='count')
        age_stats_by_tier = self.attendance.groupby('price_level')['age'].agg(['mean', 'median', 'std']).reset_index()
        
        # 3. Membership Conversion Patterns
        customer_journey = self.attendance.sort_values(['unique_key', 'date']).groupby('unique_key').agg({
            'price_level': ['first', 'last'],
            'date': ['first', 'last'],
            'facility_type': lambda x: list(x.unique())
        }).reset_index()
        
        customer_journey.columns = ['unique_key', 'first_tier', 'last_tier', 'first_visit', 'last_visit', 'facilities_used']
        customer_journey['tier_changed'] = customer_journey['first_tier'] != customer_journey['last_tier']
        customer_journey['days_active'] = (customer_journey['last_visit'] - customer_journey['first_visit']).dt.days
        
        # Conversion matrix
        conversion_matrix = customer_journey.groupby(['first_tier', 'last_tier']).size().unstack(fill_value=0)
        
        # 4. Retention Analysis
        total_customers_by_tier = self.attendance.groupby('price_level')['unique_key'].nunique().reset_index()
        total_customers_by_tier.columns = ['price_level', 'total_customers']
        
        # Calculate cancellation rates
        if 'sub_title' in self.cancellations.columns:
            cancellation_counts = self.cancellations['sub_title'].value_counts().reset_index()
            cancellation_counts.columns = ['price_level', 'cancellations']
            
            retention_data = pd.merge(total_customers_by_tier, cancellation_counts, on='price_level', how='left')
            retention_data['cancellations'] = retention_data['cancellations'].fillna(0)
            retention_data['retention_rate'] = 1 - (retention_data['cancellations'] / retention_data['total_customers'])
            retention_data['churn_rate'] = retention_data['cancellations'] / retention_data['total_customers']
        else:
            retention_data = total_customers_by_tier.copy()
            retention_data['cancellations'] = 0
            retention_data['retention_rate'] = 1.0
            retention_data['churn_rate'] = 0.0
        
        return {
            'price_level_counts': price_level_counts,
            'revenue_by_tier': revenue_by_tier,
            'age_price_analysis': age_price_analysis,
            'age_stats_by_tier': age_stats_by_tier,
            'customer_journey': customer_journey,
            'conversion_matrix': conversion_matrix,
            'retention_data': retention_data
        }

    def customer_journey_mapping(self):
        """Analyze customer journey across facilities and time"""
        
        # 1. Facility Usage Progression
        facility_progression = self.attendance.sort_values(['unique_key', 'date']).groupby('unique_key').agg({
            'facility_type': lambda x: ' → '.join(x.unique()),
            'date': ['first', 'last'],
            'price_level': 'first'
        }).reset_index()
        
        facility_progression.columns = ['unique_key', 'facility_journey', 'first_visit', 'last_visit', 'membership_tier']
        
        # Most common facility journeys
        journey_patterns = facility_progression['facility_journey'].value_counts().head(10)
        
        # 2. Cross-Facility Utilization
        customer_facility_usage = self.attendance.groupby('unique_key').agg({
            'facility_type': lambda x: list(x.unique()),
            'price_level': 'first',
            'age': 'first',
            'gender': 'first'
        }).reset_index()
        
        customer_facility_usage['num_facilities'] = customer_facility_usage['facility_type'].apply(len)
        customer_facility_usage['is_multi_facility'] = customer_facility_usage['num_facilities'] > 1
        
        # Cross-facility usage by membership tier
        cross_facility_by_tier = customer_facility_usage.groupby(['price_level', 'is_multi_facility']).size().unstack(fill_value=0)
        
        # 3. Booking Behavior Evolution (if booking data available)
        booking_evolution = self.analyze_booking_patterns()
        
        return {
            'facility_progression': facility_progression,
            'journey_patterns': journey_patterns,
            'customer_facility_usage': customer_facility_usage,
            'cross_facility_by_tier': cross_facility_by_tier,
            'booking_evolution': booking_evolution
        }
    
    def analyze_booking_patterns(self):
        """Analyze booking patterns across different activities"""
        # Combine booking data
        booking_datasets = []
        
        for df_name, activity_type in [
            (self.bookings_squash, 'Squash'),
            (self.bookings_classes, 'Classes'),
            (self.bookings_alt_sessions, 'Alternative_Sessions'),
            (self.bookings_other_activities, 'Other_Activities')
        ]:
            if not df_name.empty:
                df_copy = df_name.copy()
                df_copy['activity_type'] = activity_type
                booking_datasets.append(df_copy)
        
        if booking_datasets:
            bookings_combined = pd.concat(booking_datasets, ignore_index=True)
            
            # Standardize column names
            if 'Unique' in bookings_combined.columns:
                bookings_combined.rename(columns={'Unique': 'unique_key'}, inplace=True)
            
            # Analyze booking patterns by customer
            if 'unique_key' in bookings_combined.columns:
                booking_patterns = bookings_combined.groupby('unique_key').agg({
                    'activity_type': lambda x: list(x.unique()),
                    'Bookings Detail Start Date': 'count' if 'Bookings Detail Start Date' in bookings_combined.columns else lambda x: len(x)
                }).reset_index()
                
                return booking_patterns
        
        return pd.DataFrame()

    def create_visualizations(self, analysis_results):
        """Create professional visualizations for the analysis"""
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        
        # Create subplots for comprehensive dashboard
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Membership Tier Distribution (Attendance vs Revenue)
        ax1 = plt.subplot(4, 2, 1)
        price_data = analysis_results['price_level_counts'].head(10)
        bars = ax1.bar(range(len(price_data)), price_data['attendance_count'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(price_data))))
        ax1.set_title('Attendance Count by Membership Tier', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Membership Tier')
        ax1.set_ylabel('Attendance Count')
        ax1.set_xticks(range(len(price_data)))
        ax1.set_xticklabels(price_data['price_level'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Age Distribution by Membership Tier
        ax2 = plt.subplot(4, 2, 2)
        age_stats = analysis_results['age_stats_by_tier']
        if not age_stats.empty:
            bars = ax2.bar(range(len(age_stats)), age_stats['mean'], 
                          yerr=age_stats['std'], capsize=5,
                          color=plt.cm.plasma(np.linspace(0, 1, len(age_stats))))
            ax2.set_title('Average Age by Membership Tier', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Membership Tier')
            ax2.set_ylabel('Average Age')
            ax2.set_xticks(range(len(age_stats)))
            ax2.set_xticklabels(age_stats['price_level'], rotation=45, ha='right')
        
        # 3. Retention Rate by Membership Tier
        ax3 = plt.subplot(4, 2, 3)
        retention_data = analysis_results['retention_data']
        colors = ['#2E8B57' if rate >= 0.8 else '#DAA520' if rate >= 0.6 else '#DC143C' 
                 for rate in retention_data['retention_rate']]
        bars = ax3.bar(range(len(retention_data)), retention_data['retention_rate'], color=colors)
        ax3.set_title('Retention Rate by Membership Tier', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Membership Tier')
        ax3.set_ylabel('Retention Rate')
        ax3.set_xticks(range(len(retention_data)))
        ax3.set_xticklabels(retention_data['price_level'], rotation=45, ha='right')
        ax3.set_ylim(0, 1.1)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Customer Lifetime Value (Revenue per Customer)
        ax4 = plt.subplot(4, 2, 4)
        if not analysis_results['revenue_by_tier'].empty:
            revenue_data = analysis_results['revenue_by_tier']
            total_customers = analysis_results['retention_data']
            
            # Merge revenue and customer data
            clv_data = pd.merge(revenue_data, total_customers[['price_level', 'total_customers']], 
                               on='price_level', how='inner')
            clv_data['clv'] = clv_data['total_revenue'] / clv_data['total_customers']
            
            bars = ax4.bar(range(len(clv_data)), clv_data['clv'],
                          color=plt.cm.coolwarm(np.linspace(0, 1, len(clv_data))))
            ax4.set_title('Customer Lifetime Value by Membership Tier', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Membership Tier')
            ax4.set_ylabel('CLV (£)')
            ax4.set_xticks(range(len(clv_data)))
            ax4.set_xticklabels(clv_data['price_level'], rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'£{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Facility Usage Patterns
        ax5 = plt.subplot(4, 2, 5)
        journey_results = self.customer_journey_mapping()
        if not journey_results['journey_patterns'].empty:
            journey_data = journey_results['journey_patterns'].head(8)
            bars = ax5.barh(range(len(journey_data)), journey_data.values,
                           color=plt.cm.Set3(np.linspace(0, 1, len(journey_data))))
            ax5.set_title('Most Common Facility Usage Patterns', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Number of Customers')
            ax5.set_ylabel('Facility Journey')
            ax5.set_yticks(range(len(journey_data)))
            ax5.set_yticklabels(journey_data.index)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # 6. Multi-Facility Usage by Membership Tier
        ax6 = plt.subplot(4, 2, 6)
        if not journey_results['cross_facility_by_tier'].empty:
            cross_facility_data = journey_results['cross_facility_by_tier']
            
            # Create stacked bar chart
            single_facility = cross_facility_data[False] if False in cross_facility_data.columns else pd.Series(0, index=cross_facility_data.index)
            multi_facility = cross_facility_data[True] if True in cross_facility_data.columns else pd.Series(0, index=cross_facility_data.index)
            
            x_pos = range(len(cross_facility_data.index))
            ax6.bar(x_pos, single_facility, label='Single Facility', color='#87CEEB')
            ax6.bar(x_pos, multi_facility, bottom=single_facility, label='Multi Facility', color='#4682B4')
            
            ax6.set_title('Single vs Multi-Facility Usage by Membership Tier', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Membership Tier')
            ax6.set_ylabel('Number of Customers')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(cross_facility_data.index, rotation=45, ha='right')
            ax6.legend()
        
        # 7. Customer Tenure Distribution
        ax7 = plt.subplot(4, 2, 7)
        customer_journey = analysis_results['customer_journey']
        if not customer_journey.empty and 'days_active' in customer_journey.columns:
            tenure_bins = [0, 30, 90, 180, 365, 730, float('inf')]
            tenure_labels = ['0-30 days', '31-90 days', '91-180 days', '181-365 days', '1-2 years', '2+ years']
            customer_journey['tenure_category'] = pd.cut(customer_journey['days_active'], 
                                                        bins=tenure_bins, labels=tenure_labels, right=False)
            
            tenure_counts = customer_journey['tenure_category'].value_counts()
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(tenure_counts)))
            
            wedges, texts, autotexts = ax7.pie(tenure_counts.values, labels=tenure_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax7.set_title('Customer Tenure Distribution', fontsize=14, fontweight='bold')
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 8. Membership Tier Conversion Heatmap
        ax8 = plt.subplot(4, 2, 8)
        conversion_matrix = analysis_results['conversion_matrix']
        if not conversion_matrix.empty:
            # Normalize by row to show conversion percentages
            conversion_pct = conversion_matrix.div(conversion_matrix.sum(axis=1), axis=0) * 100
            
            im = ax8.imshow(conversion_pct.values, cmap='YlOrRd', aspect='auto')
            ax8.set_title('Membership Tier Conversion Matrix (%)', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Final Membership Tier')
            ax8.set_ylabel('Initial Membership Tier')
            ax8.set_xticks(range(len(conversion_pct.columns)))
            ax8.set_yticks(range(len(conversion_pct.index)))
            ax8.set_xticklabels(conversion_pct.columns, rotation=45, ha='right')
            ax8.set_yticklabels(conversion_pct.index)
            
            # Add text annotations
            for i in range(len(conversion_pct.index)):
                for j in range(len(conversion_pct.columns)):
                    text = ax8.text(j, i, f'{conversion_pct.iloc[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Conversion Percentage', rotation=270, labelpad=20)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        return fig

    def generate_insights_report(self, analysis_results):
        """Generate comprehensive insights report"""
        
        insights = {
            'membership_performance': {},
            'customer_behavior': {},
            'retention_insights': {},
            'revenue_insights': {}
        }
        
        # Membership Performance Insights
        price_level_counts = analysis_results['price_level_counts']
        top_tier = price_level_counts.iloc[0]
        insights['membership_performance']['dominant_tier'] = f"{top_tier['price_level']} ({top_tier['attendance_count']} visits)"
        
        # Customer Behavior Insights
        customer_journey = analysis_results['customer_journey']
        if not customer_journey.empty:
            avg_tenure = customer_journey['days_active'].mean()
            insights['customer_behavior']['average_tenure'] = f"{avg_tenure:.0f} days"
            
            tier_changes = customer_journey['tier_changed'].sum()
            total_customers = len(customer_journey)
            insights['customer_behavior']['conversion_rate'] = f"{(tier_changes/total_customers)*100:.1f}%"
        
        # Retention Insights
        retention_data = analysis_results['retention_data']
        if not retention_data.empty:
            avg_retention = retention_data['retention_rate'].mean()
            best_retention_tier = retention_data.loc[retention_data['retention_rate'].idxmax(), 'price_level']
            insights['retention_insights']['average_retention'] = f"{avg_retention:.1%}"
            insights['retention_insights']['best_retention_tier'] = best_retention_tier
        
        # Revenue Insights
        revenue_data = analysis_results['revenue_by_tier']
        if not revenue_data.empty:
            top_revenue_tier = revenue_data.loc[revenue_data['total_revenue'].idxmax()]
            insights['revenue_insights']['top_revenue_tier'] = f"{top_revenue_tier['price_level']} (£{top_revenue_tier['total_revenue']:,.2f})"
        
        return insights

    def run_complete_analysis(self):
        """Run the complete customer lifecycle and segmentation analysis"""
        print("🏃‍♂️ Running Customer Lifecycle and Segmentation Analysis...")
        print("=" * 60)
        
        # Perform membership tier performance analysis
        print("📊 Analyzing Membership Tier Performance...")
        membership_results = self.membership_tier_performance_analysis()
        
        # Perform customer journey mapping
        print("🗺️ Mapping Customer Journeys...")
        journey_results = self.customer_journey_mapping()
        
        # Combine results
        analysis_results = {**membership_results, **journey_results}
        
        # Create visualizations
        print("📈 Creating Professional Visualizations...")
        fig = self.create_visualizations(analysis_results)
        
        # Generate insights
        print("💡 Generating Insights Report...")
        insights = self.generate_insights_report(analysis_results)
        
        # Print key insights
        print("\n🎯 KEY INSIGHTS:")
        print("-" * 40)
        for category, insight_dict in insights.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for key, value in insight_dict.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print("\n✅ Analysis Complete!")
        
        return analysis_results, insights

# Initialize and run the analysis
if __name__ == "__main__":
    # Create analytics instance
    analytics = SportscentreAnalytics()
    
    # Run complete analysis
    results, insights = analytics.run_complete_analysis()
    
    # Additional detailed analysis functions
    def detailed_cohort_analysis(analytics_instance):
        """Perform detailed cohort analysis for customer retention"""
        attendance = analytics_instance.attendance
        
        # Create monthly cohorts based on first visit
        attendance['first_visit_month'] = attendance.groupby('unique_key')['date'].transform('min').dt.to_period('M')
        attendance['visit_month'] = attendance['date'].dt.to_period('M')
        
        # Calculate period number for each visit
        attendance['period_number'] = (attendance['visit_month'] - attendance['first_visit_month']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = attendance.groupby(['first_visit_month', 'period_number'])['unique_key'].nunique().reset_index()
        cohort_table = cohort_data.pivot(index='first_visit_month', columns='period_number', values='unique_key')
        
        # Calculate cohort sizes
        cohort_sizes = attendance.groupby('first_visit_month')['unique_key'].nunique()
        cohort_table = cohort_table.divide(cohort_sizes, axis=0)
        
        return cohort_table
    
    def advanced_segmentation_analysis(analytics_instance):
        """Perform advanced customer segmentation using RFM analysis"""
        attendance = analytics_instance.attendance
        
        # Calculate RFM metrics
        current_date = attendance['date'].max()
        
        rfm = attendance.groupby('unique_key').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'unique_key': 'count',  # Frequency
            'price_level': 'first'  # Monetary proxy
        }).reset_index()
        
        rfm.columns = ['unique_key', 'recency', 'frequency', 'price_level']
        
        # Create RFM scores
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str)
        
        # Define customer segments
        def segment_customers(row):
            if row['rfm_score'] in ['55', '54', '45', '44']:
                return 'Champions'
            elif row['rfm_score'] in ['53', '43', '52', '42']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['51', '41', '35', '34']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['33', '32', '31']:
                return 'New Customers'
            elif row['rfm_score'] in ['25', '24', '23', '22']:
                return 'Promising'
            elif row['rfm_score'] in ['15', '14', '13', '12']:
                return 'Need Attention'
            elif row['rfm_score'] in ['21', '11']:
                return 'About to Sleep'
            else:
                return 'At Risk'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        
        return rfm
    
    # Run additional analyses
    print("\n🔍 Running Advanced Cohort Analysis...")
    cohort_analysis = detailed_cohort_analysis(analytics)
    
    print("🎯 Running RFM Segmentation Analysis...")
    rfm_analysis = advanced_segmentation_analysis(analytics)
    
    print("\n📋 ADVANCED ANALYSIS SUMMARY:")
    print("-" * 50)
    print(f"Customer Segments Identified: {rfm_analysis['segment'].nunique()}")
    print(f"Cohort Periods Analyzed: {cohort_analysis.shape[1]}")
    print(f"Total Unique Customers: {analytics.attendance['unique_key'].nunique()}")
    print(f"Date Range: {analytics.attendance['date'].min()} to {analytics.attendance['date'].max()}")
