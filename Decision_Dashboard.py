import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Data Loading Class (Based on Revenue_Opt.py)
class EnhancedDataLoader:
    def __init__(self, pkl_data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        self.pkl_data_dir = Path(pkl_data_dir)
        self.cache_dir = Path('./data_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        self.primary_profiles = [
            'Student Member', 'Staff Member', 'Alumni Member',
            'Community Member', 'Associate Member',
            'Community Over 65 Member', 'Staff Non Member'
        ]
        
        self.financial_data = None
        self.attendance_data = None
        self.nps_data = None
        self.cancellation_data = None
        self.user_data = None

    def drop_duplicate_columns(self, df):
        if df is None or df.empty:
            return df
        v = pd.Series(df.columns)
        dups = v[v.duplicated()].unique()
        if len(dups):
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df

    def canonicalize_financial_columns(self, df):
        if df is None or df.empty:
            return df
        
        column_mapping = {
            'sales_detail_participation_date': 'date',
            'sales_detail_participation_date_time': 'date',
            'sales_detail_gross_amount': 'price',
            'sales_detail_net_amount': 'price',
            'sales_detail_quantity': 'quantity',
            'sales_detail_price_level': 'customer_type',
            'contacts_detail_price_level': 'customer_type',
            'unique_key': 'unique_id',
            'product_hierarchy_product': 'product'
        }
        
        lower_col_map = {k.lower(): v for k, v in column_mapping.items()}
        new_cols = {}
        for col in df.columns:
            canonical_name = lower_col_map.get(col.lower())
            if canonical_name:
                new_cols[col] = canonical_name
        
        if new_cols:
            df = df.rename(columns=new_cols)
        
        return self.drop_duplicate_columns(df)

    def load_pkl_datasets(self):
        datasets = {}
        pkl_files = {
            'financial': 'financial.pkl',
            'attendance': 'attendance.pkl',
            'bookings_squash': 'bookings_squash.pkl',
            'bookings_classes': 'bookings_classes.pkl',
            'nps': 'nps.pkl',
            'cancellation': 'cancellations.pkl'
        }
        
        for name, filename in pkl_files.items():
            try:
                pkl_path = self.pkl_data_dir / filename
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        datasets[name] = pickle.load(f)
                else:
                    datasets[name] = None
            except Exception as e:
                datasets[name] = None
        
        return datasets

    def load_excel_fallback(self, dataset_name):
        try:
            fallback_paths = {
                'financial': 'Financial_Data.xlsx',
                'attendance': 'Att_data.xlsx',
                'nps': 'NPS_Updated_25_G24.csv',
                'cancellation': 'Cancellation_Updated_25_G24.xlsx',
                'user': 'F:/UoB Study/Capstone Project/Final Project/Datasets/User_25_G24.xlsx'
            }
            
            path = fallback_paths.get(dataset_name)
            if not path:
                return None
                
            if dataset_name == 'nps':
                return pd.read_csv(path)
            elif dataset_name == 'financial':
                # Handle multiple sheets
                xls = pd.ExcelFile(path)
                sheets = []
                for sheet in ['S&F', 'Tiverton']:
                    if sheet in xls.sheet_names:
                        sheets.append(pd.read_excel(xls, sheet))
                return pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
            elif dataset_name == 'attendance':
                # Handle multiple sheets for attendance
                xls = pd.ExcelFile(path)
                sheets = []
                for sheet in xls.sheet_names:
                    if 'gym' in sheet.lower() or 'barrier' in sheet.lower():
                        sheets.append(pd.read_excel(xls, sheet))
                return pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
            else:
                return pd.read_excel(path)
                
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return None

    def load_and_prepare_data(self):
        # Load PKL datasets first
        pkl_datasets = self.load_pkl_datasets()
        
        # Load financial data
        if pkl_datasets['financial'] is not None:
            self.financial_data = pkl_datasets['financial']
            if isinstance(self.financial_data, dict):
                sheets = []
                for sheet_data in self.financial_data.values():
                    if sheet_data is not None:
                        sheets.append(sheet_data)
                self.financial_data = pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
        else:
            self.financial_data = self.load_excel_fallback('financial')
        
        if self.financial_data is not None:
            self.financial_data = self.canonicalize_financial_columns(self.financial_data)
            # Filter for primary profiles
            if 'customer_type' in self.financial_data.columns:
                self.financial_data = self.financial_data[
                    self.financial_data['customer_type'].isin(self.primary_profiles)
                ]
                # Clean price data
                if 'price' in self.financial_data.columns:
                    self.financial_data['price'] = pd.to_numeric(self.financial_data['price'], errors='coerce')
                    self.financial_data = self.financial_data[self.financial_data['price'] > 0]
        
        # Load attendance data
        if pkl_datasets['attendance'] is not None:
            self.attendance_data = pkl_datasets['attendance']
            if isinstance(self.attendance_data, dict):
                sheets = []
                for sheet_data in self.attendance_data.values():
                    if sheet_data is not None:
                        sheets.append(sheet_data)
                self.attendance_data = pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
        else:
            self.attendance_data = self.load_excel_fallback('attendance')
        
        if self.attendance_data is not None:
            self.attendance_data = self.canonicalize_financial_columns(self.attendance_data)
        
        # Load other datasets
        nps_data = pkl_datasets.get('nps')
        self.nps_data = nps_data if nps_data is not None and not nps_data.empty else self.load_excel_fallback('nps')
        cancellation_data = pkl_datasets.get('cancellation')
        self.cancellation_data = cancellation_data if cancellation_data is not None and not cancellation_data.empty else self.load_excel_fallback('cancellation')
        self.user_data = self.load_excel_fallback('user')
        
        return {
            'financial': self.financial_data,
            'attendance': self.attendance_data,
            'nps': self.nps_data,
            'cancellation': self.cancellation_data,
            'user': self.user_data
        }

# Advanced Pricing Analytics Engine
class PricingAnalyticsEngine:
    def __init__(self, datasets):
        self.financial_data = datasets.get('financial', pd.DataFrame())
        self.attendance_data = datasets.get('attendance', pd.DataFrame())
        self.nps_data = datasets.get('nps', pd.DataFrame())
        self.cancellation_data = datasets.get('cancellation', pd.DataFrame())
        self.user_data = datasets.get('user', pd.DataFrame())
        
        self.service_elasticity = {}
        self.segment_analysis = {}
        
        # Initialize analytics
        self._initialize_analytics()
    
    def _initialize_analytics(self):
        self._calculate_service_elasticity()
        self._analyze_customer_segments()
        
    def _map_services_to_data(self):
        """Map financial transactions to service categories"""
        service_mapping = {
            'gym': ['gym', 'fitness', 'weight'],
            'swimming': ['swim', 'pool'],
            'classes': ['class', 'session', 'group'],
            'courts': ['court', 'squash', 'badminton']
        }
        
        if self.financial_data.empty or 'product' not in self.financial_data.columns:
            # Fallback: use customer type as service proxy
            return self._map_customer_type_to_service()
        
        self.financial_data['service_category'] = 'Other'
        
        for service, keywords in service_mapping.items():
            for keyword in keywords:
                mask = self.financial_data['product'].str.contains(keyword, case=False, na=False)
                self.financial_data.loc[mask, 'service_category'] = service.title()
        
        return self.financial_data
    
    def _map_customer_type_to_service(self):
        """Fallback mapping using customer types"""
        # Map customer types to primary service usage patterns
        type_service_mapping = {
            'Student Member': 'Gym',
            'Staff Member': 'Classes', 
            'Alumni Member': 'Swimming',
            'Community Member': 'Gym',
            'Associate Member': 'Courts',
            'Community Over 65 Member': 'Swimming',
            'Staff Non Member': 'Classes'
        }
        
        if 'customer_type' in self.financial_data.columns:
            self.financial_data['service_category'] = self.financial_data['customer_type'].map(
                type_service_mapping
            ).fillna('Gym')
        
        return self.financial_data
    
    def _calculate_service_elasticity(self):
        """Calculate price elasticity for each service category"""
        if self.financial_data.empty:
            self.service_elasticity = {'gym': -0.8, 'swimming': -0.6, 'classes': -1.2, 'courts': -0.5}
            return
        
        # Map services
        self._map_services_to_data()
        
        # Calculate usage frequency from attendance
        if not self.attendance_data.empty and 'unique_id' in self.attendance_data.columns:
            usage_freq = self.attendance_data.groupby('unique_id').size().reset_index(name='total_visits')
            
            # Merge with financial data
            merged = self.financial_data.merge(usage_freq, on='unique_id', how='inner')
            
            # Calculate elasticity by service
            for service in merged['service_category'].unique():
                service_data = merged[merged['service_category'] == service]
                
                if len(service_data) > 10:
                    # Calculate correlation between price and usage
                    correlation = service_data['price'].corr(service_data['total_visits'])
                    # Convert correlation to elasticity estimate
                    elasticity = correlation * -1.5 if pd.notna(correlation) else -0.8
                    self.service_elasticity[service.lower()] = max(min(elasticity, 0), -3.0)
                else:
                    self.service_elasticity[service.lower()] = -0.8
        else:
            # Default elasticity values based on typical gym/leisure industry
            self.service_elasticity = {
                'gym': -0.8,
                'swimming': -0.6, 
                'classes': -1.2,
                'courts': -0.5
            }
    
    def _analyze_customer_segments(self):
        """Analyze customer segments for churn risk and value"""
        if self.financial_data.empty:
            return
        
        # Calculate segment metrics
        segment_metrics = self.financial_data.groupby('customer_type').agg({
            'price': ['mean', 'sum', 'count'],
            'unique_id': 'nunique'
        }).round(2)
        
        segment_metrics.columns = ['avg_price', 'total_revenue', 'transaction_count', 'unique_customers']
        segment_metrics = segment_metrics.reset_index()
        
        # Calculate churn risk based on cancellation data
        if not self.cancellation_data.empty:
            # Map cancellations to segments
            for _, row in segment_metrics.iterrows():
                segment = row['customer_type']
                customers = row['unique_customers']
                
                # Calculate churn rate (simplified)
                churn_count = 0
                if 'unique id' in self.cancellation_data.columns:
                    segment_ids = self.financial_data[
                        self.financial_data['customer_type'] == segment
                    ]['unique_id'].unique()
                    
                    canceled_ids = self.cancellation_data['unique id'].dropna().unique()
                    churn_count = len(set(segment_ids) & set(canceled_ids))
                
                churn_rate = churn_count / customers if customers > 0 else 0
                segment_metrics.loc[segment_metrics['customer_type'] == segment, 'churn_risk'] = churn_rate
        else:
            segment_metrics['churn_risk'] = 0.1  # Default assumption
        
        self.segment_analysis = segment_metrics.to_dict('records')

# Dashboard Functions
def calculate_revenue_projections(price_adjustments, time_horizon, analytics_engine):
    """Calculate revenue projections based on elasticity models"""
    projections = []
    
    for service, price_change in price_adjustments.items():
        service_key = service.lower().replace(' ', '_')
        elasticity = analytics_engine.service_elasticity.get(service_key, -0.8)
        
        # Calculate quantity and revenue changes
        quantity_change = elasticity * price_change
        revenue_change = price_change + quantity_change + (price_change * quantity_change)
        
        # Apply time horizon multiplier
        time_multiplier = {'3 months': 0.25, '6 months': 0.5, '12 months': 1.0}.get(time_horizon, 1.0)
        
        projections.append({
            'Service': service,
            'Price_Change_Pct': price_change * 100,
            'Quantity_Change_Pct': quantity_change * 100,
            'Revenue_Change_Pct': revenue_change * 100 * time_multiplier,
            'Elasticity': elasticity
        })
    
    return pd.DataFrame(projections)

def predict_churn_from_prices(price_adjustments, analytics_engine):
    """Predict churn risk based on price changes"""
    churn_data = []
    
    for segment_info in analytics_engine.segment_analysis:
        segment = segment_info['customer_type']
        
        # Find corresponding service for segment (simplified mapping)
        service_map = {
            'Student Member': 'gym',
            'Staff Member': 'classes',
            'Alumni Member': 'swimming', 
            'Community Member': 'gym',
            'Associate Member': 'courts',
            'Community Over 65 Member': 'swimming',
            'Staff Non Member': 'classes'
        }
        
        service = service_map.get(segment, 'gym')
        service_title = service.title()
        
        price_change = price_adjustments.get(service_title, 0)
        base_churn = segment_info.get('churn_risk', 0.1)
        
        # Adjust churn risk based on price change and elasticity
        elasticity = analytics_engine.service_elasticity.get(service, -0.8)
        churn_adjustment = abs(price_change) * (1 + abs(elasticity)) * 0.5
        
        adjusted_churn = min(base_churn + churn_adjustment, 1.0)
        
        churn_data.append({
            'Segment': segment,
            'Price_Change': price_change,
            'Churn_Risk': adjusted_churn,
            'Customer_Count': segment_info.get('unique_customers', 0),
            'Service': service_title
        })
    
    return pd.DataFrame(churn_data)

def generate_dynamic_recommendations(price_adjustments, revenue_projections, churn_predictions):
    """Generate strategic recommendations based on analysis"""
    recommendations = []
    
    # Merge revenue and churn data
    for _, rev_row in revenue_projections.iterrows():
        service = rev_row['Service']
        revenue_change = rev_row['Revenue_Change_Pct']
        price_change = rev_row['Price_Change_Pct']
        
        # Find corresponding churn data
        churn_data = churn_predictions[churn_predictions['Service'] == service]
        max_churn = churn_data['Churn_Risk'].max() if not churn_data.empty else 0
        
        # Generate recommendation based on revenue and churn
        if revenue_change > 5 and max_churn < 0.2:
            title = f"Optimize pricing for {service}"
            impact = f"Expected revenue increase: +{revenue_change:.1f}%, Low churn risk"
            implementation = "Implement gradual price increase with enhanced value proposition"
        elif revenue_change < -5 or max_churn > 0.3:
            title = f"Address pricing concerns for {service}"
            impact = f"Revenue risk: {revenue_change:.1f}%, Churn risk: {max_churn:.1%}"
            implementation = "Consider price reduction, promotions, or service improvements"
        elif abs(revenue_change) <= 5 and max_churn <= 0.3:
            title = f"Maintain current strategy for {service}"
            impact = "Balanced revenue and retention outlook"
            implementation = "Continue monitoring, focus on service quality"
        else:
            title = f"Review pricing strategy for {service}"
            impact = "Mixed signals require careful analysis"
            implementation = "Conduct detailed customer research before changes"
        
        recommendations.append({
            'title': title,
            'impact': impact,
            'implementation': implementation
        })
    
    return recommendations

# Main Dashboard Application
def create_interactive_dashboard():
    """
    Build Streamlit dashboard for real-time scenario analysis
    """
    st.set_page_config(
        page_title="Sports Centre Strategic Pricing Dashboard",
        page_icon="🏋️‍♂️",
        layout="wide"
    )
    
    st.title("🏋️‍♂️ Sports Centre Strategic Pricing Dashboard")
    st.markdown("### Real-time scenario analysis for UoB Sports & Fitness Centre")
    
    # Initialize data loader and analytics
    @st.cache_data
    def load_data():
        loader = EnhancedDataLoader()
        return loader.load_and_prepare_data()
    
    @st.cache_data
    def initialize_analytics(datasets):
        return PricingAnalyticsEngine(datasets)
    
    # Load data
    with st.spinner("Loading data and initializing analytics..."):
        datasets = load_data()
        analytics_engine = initialize_analytics(datasets)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Scenario Configuration")
    
    # Price adjustment sliders
    price_adjustments = {}
    services = ['Gym', 'Swimming', 'Classes', 'Courts']
    
    for service in services:
        price_adjustments[service] = st.sidebar.slider(
            f"{service} Price Change (%)",
            min_value=-20, max_value=30, value=0, step=1
        ) / 100
    
    # Advanced options
    st.sidebar.header("⚙️ Advanced Options")
    time_horizon = st.sidebar.selectbox(
        "Analysis Period", 
        ["3 months", "6 months", "12 months"]
    )
    capacity_constraint = st.sidebar.checkbox("Apply Capacity Constraints")
    show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)
    
    # Main dashboard content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📈 Revenue Impact Projection")
        
        # Calculate and display revenue projections
        revenue_projections = calculate_revenue_projections(
            price_adjustments, time_horizon, analytics_engine
        )
        
        if not revenue_projections.empty:
            # Create revenue impact chart
            fig = px.bar(
                revenue_projections, 
                x='Service', 
                y='Revenue_Change_Pct',
                title="Projected Revenue Change by Service",
                color='Revenue_Change_Pct',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            total_revenue_change = revenue_projections['Revenue_Change_Pct'].sum()
            avg_elasticity = revenue_projections['Elasticity'].mean()
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Total Revenue Impact", f"{total_revenue_change:.1f}%")
            with col1b:
                st.metric("Avg Price Elasticity", f"{avg_elasticity:.2f}")
        else:
            st.warning("Unable to calculate revenue projections with current data")
    
    with col2:
        st.header("👥 Customer Impact Analysis")
        
        # Churn predictions
        churn_predictions = predict_churn_from_prices(price_adjustments, analytics_engine)
        
        if not churn_predictions.empty:
            fig2 = px.scatter(
                churn_predictions,
                x='Price_Change',
                y='Churn_Risk',
                size='Customer_Count',
                color='Segment',
                title="Churn Risk vs Price Changes",
                hover_data=['Service']
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Risk alerts
            high_risk_segments = churn_predictions[churn_predictions['Churn_Risk'] > 0.3]
            if not high_risk_segments.empty:
                st.error(f"⚠️ High churn risk detected for: {', '.join(high_risk_segments['Segment'].tolist())}")
            
            moderate_risk = churn_predictions[
                (churn_predictions['Churn_Risk'] > 0.15) & 
                (churn_predictions['Churn_Risk'] <= 0.3)
            ]
            if not moderate_risk.empty:
                st.warning(f"⚡ Moderate churn risk: {', '.join(moderate_risk['Segment'].tolist())}")
        else:
            st.warning("Unable to calculate churn predictions with current data")
    
    # Detailed metrics section
    if show_detailed_metrics and not revenue_projections.empty:
        st.header("📊 Detailed Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Revenue Projections Table")
            st.dataframe(
                revenue_projections[['Service', 'Price_Change_Pct', 'Revenue_Change_Pct', 'Elasticity']].round(2),
                use_container_width=True
            )
        
        with col4:
            st.subheader("Churn Risk Analysis")
            if not churn_predictions.empty:
                st.dataframe(
                    churn_predictions[['Segment', 'Churn_Risk', 'Customer_Count']].round(3),
                    use_container_width=True
                )
    
    # Strategic recommendations section
    st.header("🎯 Strategic Recommendations")
    
    if not revenue_projections.empty and not churn_predictions.empty:
        recommendations = generate_dynamic_recommendations(
            price_adjustments, revenue_projections, churn_predictions
        )
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"**{i}. {rec['title']}**"):
                st.write(f"**Impact:** {rec['impact']}")
                st.write(f"**Implementation:** {rec['implementation']}")
    else:
        st.info("Configure price adjustments to see recommendations")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>UoB Sports & Fitness Centre - Strategic Pricing Dashboard</small><br>
        <small>Based on advanced price elasticity analysis and customer behavior modeling</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Run the dashboard
if __name__ == "__main__":
    create_interactive_dashboard()
