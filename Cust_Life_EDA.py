# enhanced_analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SportscentreAnalytics:
    def __init__(self, data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        """Initialize the analytics class and load prepared datasets"""
        self.data_dir = data_dir
        self.load_prepared_data()
        
    def load_prepared_data(self):
        """Load prepared datasets from files"""
        print("📂 Loading prepared data...")
        
        try:
            # Load main datasets
            with open(f'{self.data_dir}/attendance.pkl', 'rb') as f:
                self.attendance = pickle.load(f)
            
            with open(f'{self.data_dir}/financial.pkl', 'rb') as f:
                self.financial = pickle.load(f)
            
            with open(f'{self.data_dir}/cancellations.pkl', 'rb') as f:
                self.cancellations = pickle.load(f)
            
            with open(f'{self.data_dir}/nps.pkl', 'rb') as f:
                self.nps = pickle.load(f)
            
            # Load booking datasets
            with open(f'{self.data_dir}/bookings_squash.pkl', 'rb') as f:
                self.bookings_squash = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_classes.pkl', 'rb') as f:
                self.bookings_classes = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_alt_sessions.pkl', 'rb') as f:
                self.bookings_alt_sessions = pickle.load(f)
            
            with open(f'{self.data_dir}/bookings_other_activities.pkl', 'rb') as f:
                self.bookings_other_activities = pickle.load(f)
            
            # Load data info
            with open(f'{self.data_dir}/data_info.pkl', 'rb') as f:
                self.data_info = pickle.load(f)
            
            print("✅ All prepared data loaded successfully!")
            print(f"📊 Data prepared on: {self.data_info['preparation_date']}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading prepared data: {e}")
            print("🔄 Please run the data preparation script first!")
            raise

    def fix_data_types(self):
        """Fix data type issues in the loaded data"""
        print("🔧 Fixing data types...")
        
        # Fix age column
        print("Fixing age column...")
        self.attendance['age'] = pd.to_numeric(self.attendance['age'], errors='coerce')
        
        # Fill missing ages with median (or drop rows - your choice)
        if self.attendance['age'].isnull().sum() > 0:
            median_age = self.attendance['age'].median()
            self.attendance['age'].fillna(median_age, inplace=True)
            print(f"Filled {self.attendance['age'].isnull().sum()} missing age values with median: {median_age}")
        
        # Fix other potential numeric columns
        numeric_columns = ['unique_key']
        for col in numeric_columns:
            if col in self.attendance.columns:
                self.attendance[col] = pd.to_numeric(self.attendance[col], errors='coerce')
        
        print("✅ Data types fixed!")

    def membership_tier_performance_analysis(self):
        """Perform comprehensive membership tier performance analysis"""
        print("🎯 Analyzing membership tier performance...")
        
        # Fix data types first
        self.fix_data_types()
        
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
        print("🗺️ Mapping customer journeys...")
        
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
        
        # 3. Booking Behavior Evolution
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
            (self.bookings_alt_sessions, 'Alternative_Sessions')
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

    def create_separate_visualizations(self, analysis_results):
        """Create separate professional visualizations for each analysis"""
        print("📈 Creating separate professional visualizations...")
        
        # Set up consistent styling
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Color palettes
        colors_primary = ['#2E8B57', '#4682B4', '#DC143C', '#DAA520', '#9370DB', '#FF6347', '#20B2AA', '#FF69B4']
        colors_secondary = ['#87CEEB', '#98FB98', '#F0E68C', '#DDA0DD', '#F5DEB3', '#FFB6C1', '#AFEEEE', '#D3D3D3']
        
        # 1. MEMBERSHIP TIER ATTENDANCE DISTRIBUTION
        print("Creating Chart 1: Membership Tier Attendance Distribution")
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        price_data = analysis_results['price_level_counts'].head(10)
        
        bars = ax1.bar(range(len(price_data)), price_data['attendance_count'], 
                      color=colors_primary[:len(price_data)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_title('Attendance Count by Membership Tier', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Membership Tier', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Attendance Count', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(price_data)))
        ax1.set_xticklabels(price_data['price_level'], rotation=45, ha='right', fontsize=12)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. AGE DISTRIBUTION BY MEMBERSHIP TIER
        print("Creating Chart 2: Age Distribution by Membership Tier")
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        age_stats = analysis_results['age_stats_by_tier']
        
        if not age_stats.empty:
            bars = ax2.bar(range(len(age_stats)), age_stats['mean'], 
                          yerr=age_stats['std'], capsize=5, alpha=0.8,
                          color=colors_primary[:len(age_stats)], edgecolor='black', linewidth=0.5)
            
            ax2.set_title('Average Age by Membership Tier', fontsize=18, fontweight='bold', pad=20)
            ax2.set_xlabel('Membership Tier', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Average Age (years)', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(age_stats)))
            ax2.set_xticklabels(age_stats['price_level'], rotation=45, ha='right', fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 3. RETENTION RATE BY MEMBERSHIP TIER
        print("Creating Chart 3: Retention Rate by Membership Tier")
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        retention_data = analysis_results['retention_data']
        
        # Color code based on retention rate
        colors = ['#2E8B57' if rate >= 0.8 else '#DAA520' if rate >= 0.6 else '#DC143C' 
                 for rate in retention_data['retention_rate']]
        
        bars = ax3.bar(range(len(retention_data)), retention_data['retention_rate'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_title('Retention Rate by Membership Tier', fontsize=18, fontweight='bold', pad=20)
        ax3.set_xlabel('Membership Tier', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Retention Rate', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(retention_data)))
        ax3.set_xticklabels(retention_data['price_level'], rotation=45, ha='right', fontsize=12)
        ax3.set_ylim(0, 1.1)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add horizontal lines for benchmarks
        ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (80%)')
        ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Average (60%)')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 4. CUSTOMER LIFETIME VALUE
        print("Creating Chart 4: Customer Lifetime Value by Membership Tier")
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        
        if not analysis_results['revenue_by_tier'].empty:
            revenue_data = analysis_results['revenue_by_tier']
            total_customers = analysis_results['retention_data']
            
            # Merge revenue and customer data
            clv_data = pd.merge(revenue_data, total_customers[['price_level', 'total_customers']], 
                               on='price_level', how='inner')
            clv_data['clv'] = clv_data['total_revenue'] / clv_data['total_customers']
            
            bars = ax4.bar(range(len(clv_data)), clv_data['clv'],
                          color=colors_primary[:len(clv_data)], alpha=0.8, 
                          edgecolor='black', linewidth=0.5)
            
            ax4.set_title('Customer Lifetime Value by Membership Tier', fontsize=18, fontweight='bold', pad=20)
            ax4.set_xlabel('Membership Tier', fontsize=14, fontweight='bold')
            ax4.set_ylabel('CLV (£)', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(clv_data)))
            ax4.set_xticklabels(clv_data['price_level'], rotation=45, ha='right', fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'£{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 5. FACILITY USAGE PATTERNS
        print("Creating Chart 5: Most Common Facility Usage Patterns")
        fig5, ax5 = plt.subplots(figsize=(14, 10))
        journey_results = self.customer_journey_mapping()
        
        if not journey_results['journey_patterns'].empty:
            journey_data = journey_results['journey_patterns'].head(10)
            
            bars = ax5.barh(range(len(journey_data)), journey_data.values,
                           color=colors_primary[:len(journey_data)], alpha=0.8, 
                           edgecolor='black', linewidth=0.5)
            
            ax5.set_title('Most Common Facility Usage Patterns', fontsize=18, fontweight='bold', pad=20)
            ax5.set_xlabel('Number of Customers', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Facility Journey', fontsize=14, fontweight='bold')
            ax5.set_yticks(range(len(journey_data)))
            ax5.set_yticklabels(journey_data.index, fontsize=12)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                        f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=11)
            
            ax5.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 6. MULTI-FACILITY USAGE
        print("Creating Chart 6: Single vs Multi-Facility Usage by Membership Tier")
        fig6, ax6 = plt.subplots(figsize=(14, 8))
        
        if not journey_results['cross_facility_by_tier'].empty:
            cross_facility_data = journey_results['cross_facility_by_tier']
            
            # Create stacked bar chart
            single_facility = cross_facility_data[False] if False in cross_facility_data.columns else pd.Series(0, index=cross_facility_data.index)
            multi_facility = cross_facility_data[True] if True in cross_facility_data.columns else pd.Series(0, index=cross_facility_data.index)
            
            x_pos = range(len(cross_facility_data.index))
            bars1 = ax6.bar(x_pos, single_facility, label='Single Facility', color='#87CEEB', alpha=0.8)
            bars2 = ax6.bar(x_pos, multi_facility, bottom=single_facility, label='Multi Facility', color='#4682B4', alpha=0.8)
            
            ax6.set_title('Single vs Multi-Facility Usage by Membership Tier', fontsize=18, fontweight='bold', pad=20)
            ax6.set_xlabel('Membership Tier', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Number of Customers', fontsize=14, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(cross_facility_data.index, rotation=45, ha='right', fontsize=12)
            ax6.legend(fontsize=12)
            ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 7. CUSTOMER TENURE DISTRIBUTION
        print("Creating Chart 7: Customer Tenure Distribution")
        fig7, ax7 = plt.subplots(figsize=(12, 10))
        customer_journey = analysis_results['customer_journey']
        
        if not customer_journey.empty and 'days_active' in customer_journey.columns:
            tenure_bins = [0, 30, 90, 180, 365, 730, float('inf')]
            tenure_labels = ['0-30 days', '31-90 days', '91-180 days', '181-365 days', '1-2 years', '2+ years']
            customer_journey['tenure_category'] = pd.cut(customer_journey['days_active'], 
                                                        bins=tenure_bins, labels=tenure_labels, right=False)
            
            tenure_counts = customer_journey['tenure_category'].value_counts()
            
            wedges, texts, autotexts = ax7.pie(tenure_counts.values, labels=tenure_counts.index, 
                                              autopct='%1.1f%%', colors=colors_primary[:len(tenure_counts)], 
                                              startangle=90, textprops={'fontsize': 12})
            
            ax7.set_title('Customer Tenure Distribution', fontsize=18, fontweight='bold', pad=20)
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
        
        plt.tight_layout()
        plt.show()
        
        # 8. MEMBERSHIP TIER CONVERSION HEATMAP
        print("Creating Chart 8: Membership Tier Conversion Matrix")
        fig8, ax8 = plt.subplots(figsize=(12, 10))
        conversion_matrix = analysis_results['conversion_matrix']
        
        if not conversion_matrix.empty:
            # Normalize by row to show conversion percentages
            conversion_pct = conversion_matrix.div(conversion_matrix.sum(axis=1), axis=0) * 100
            
            im = ax8.imshow(conversion_pct.values, cmap='YlOrRd', aspect='auto')
            ax8.set_title('Membership Tier Conversion Matrix (%)', fontsize=18, fontweight='bold', pad=20)
            ax8.set_xlabel('Final Membership Tier', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Initial Membership Tier', fontsize=14, fontweight='bold')
            ax8.set_xticks(range(len(conversion_pct.columns)))
            ax8.set_yticks(range(len(conversion_pct.index)))
            ax8.set_xticklabels(conversion_pct.columns, rotation=45, ha='right', fontsize=12)
            ax8.set_yticklabels(conversion_pct.index, fontsize=12)
            
            # Add text annotations
            for i in range(len(conversion_pct.index)):
                for j in range(len(conversion_pct.columns)):
                    text = ax8.text(j, i, f'{conversion_pct.iloc[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontweight='bold', fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Conversion Percentage', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ All visualizations created successfully!")

    def generate_insights_report(self, analysis_results):
        """Generate comprehensive insights report"""
        print("💡 Generating insights report...")
        
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
        print("=" * 80)
        
        # Perform membership tier performance analysis
        membership_results = self.membership_tier_performance_analysis()
        
        # Perform customer journey mapping
        journey_results = self.customer_journey_mapping()
        
        # Combine results
        analysis_results = {**membership_results, **journey_results}
        
        # Create separate visualizations
        print("\n" + "="*80)
        print("📊 CREATING SEPARATE VISUALIZATIONS")
        print("="*80)
        self.create_separate_visualizations(analysis_results)
        
        # Generate insights
        insights = self.generate_insights_report(analysis_results)
        
        # Print key insights
        print("\n" + "="*80)
        print("🎯 KEY INSIGHTS SUMMARY")
        print("="*80)
        for category, insight_dict in insights.items():
            print(f"\n📈 {category.upper().replace('_', ' ')}:")
            for key, value in insight_dict.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("📊 All visualizations displayed as separate charts")
        print("💡 Insights generated successfully")
        print("="*80)
        
        return analysis_results, insights

# Run the analytics
if __name__ == "__main__":
    # Check if prepared data exists
    import os
    if not os.path.exists('prepared_data'):
        print("❌ Prepared data directory not found!")
        print("🔄 Please run the data preparation script first:")
        print("   python data_preparation.py")
        exit(1)
    
    # Create analytics instance
    analytics = SportscentreAnalytics()
    
    # Run complete analysis
    results, insights = analytics.run_complete_analysis()
    
    print("\n🎉 Enhanced Analytics completed successfully!")
    print("📊 All visualizations are now displayed as separate, clearly visible charts!")
