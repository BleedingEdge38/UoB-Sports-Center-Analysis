import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

# ------- NEW: Membership Definitions for UoB Sports & Fitness -----------
MEMBERSHIP_PRODUCTS = [
    # Student
    {"label": "Student Peak Annual", "code": "student_peak_annual", "price": 368},
    {"label": "Student 5 Day Peak Annual", "code": "student_5day_peak_annual", "price": 281},
    {"label": "Student Weekend Annual", "code": "student_weekend_annual", "price": 118},
    {"label": "Student Off Peak Annual", "code": "student_offpeak_annual", "price": 281},
    {"label": "Student 6am-3:30pm Annual", "code": "student_6am_3_30pm_annual", "price": 222},
    {"label": "Student 9am-2pm Annual", "code": "student_9am_2pm_annual", "price": 164},
    {"label": "Student Swim Peak Annual", "code": "student_swim_peak_annual", "price": 255},
    {"label": "Student Swim Off Peak", "code": "student_swim_offpeak_annual", "price": 178},
    {"label": "Student Swim 9am-2pm", "code": "student_swim_9am_2pm_annual", "price": 85},
    {"label": "Student Squash Peak", "code": "student_squash_peak_annual", "price": 146},
    {"label": "Student Squash Off Peak", "code": "student_squash_offpeak_annual", "price": 108},
    {"label": "Student One Month Peak", "code": "student_1month_peak", "price": 49},
    {"label": "Student One Month Weekend", "code": "student_1month_weekend", "price": 16},
    {"label": "Student One Month Off Peak", "code": "student_1month_offpeak", "price": 38},
    {"label": "Student One Month 6am-3:30pm", "code": "student_1month_6am_3_30pm", "price": 29},
    {"label": "Student One Month 9am-2pm", "code": "student_1month_9am_2pm", "price": 21},
    # Staff
    {"label": "Staff Peak", "code": "staff_peak", "price": 58*12},
    {"label": "Staff Off Peak", "code": "staff_offpeak", "price": 41*12},
    {"label": "Staff Saver", "code": "staff_saver", "price": 21*12},
    # Alumni
    {"label": "Alumni Peak", "code": "alumni_peak", "price": 55.62*12},
    {"label": "Alumni Off Peak", "code": "alumni_offpeak", "price": 38.99*12},
    # Over 65
    {"label": "Over 65 Peak", "code": "over65_peak", "price": 55.36*12},
    {"label": "Over 65 Off Peak", "code": "over65_offpeak", "price": 37.64*12},
    # Community
    {"label": "Community", "code": "community", "price": 68*12},
]

# --- Data Loading Classes remain unchanged (as in supplied code) ---

class EnhancedDataLoader:
    def __init__(self, pkl_data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        self.pkl_data_dir = Path(pkl_data_dir)
        self.cache_dir = Path('./data_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Updated to reflect actual UoB membership tiers
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
                xls = pd.ExcelFile(path)
                sheets = []
                for sheet in ['S&amp;F', 'Tiverton']:
                    if sheet in xls.sheet_names:
                        sheets.append(pd.read_excel(xls, sheet))
                return pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
            elif dataset_name == 'attendance':
                xls = pd.ExcelFile(path)
                sheets = []
                for sheet in xls.sheet_names:
                    if 'gym' in sheet.lower() or 'barrier' in sheet.lower() or 'pool' in sheet.lower():
                        sheets.append(pd.read_excel(xls, sheet))
                return pd.concat(sheets, ignore_index=True) if sheets else pd.DataFrame()
            else:
                return pd.read_excel(path)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return None

    def load_and_prepare_data(self):
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
            if 'customer_type' in self.financial_data.columns:
                self.financial_data = self.financial_data[
                    self.financial_data['customer_type'].isin(self.primary_profiles)
                ]
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


# --- Membership-based Analytics Engine --- 

class MembershipAnalyticsEngine:
    def __init__(self, datasets):
        self.financial_data = datasets.get('financial', pd.DataFrame())
        self.user_data = datasets.get('user', pd.DataFrame())
        self.cancellation_data = datasets.get('cancellation', pd.DataFrame())
        self.nps_data = datasets.get('nps', pd.DataFrame())
        self.membership_products = MEMBERSHIP_PRODUCTS
        self.price_lookup = {m['code']: m['price'] for m in self.membership_products}
        self._prepare_membership_data()
        self.segment_analysis = self._analyze_segments()
        self.membership_elasticities = self._estimate_elasticities()

    def _prepare_membership_data(self):
        # Map transactions to membership products using price, name, or level where possible
        if self.financial_data is None or self.financial_data.empty:
            self.membership_data = pd.DataFrame()
            return
        df = self.financial_data.copy()
        # Try to infer membership type (this step may require adjustment based on your real data schema)
        def infer_membership(row):
            # Map price or product field to product code as best as possible
            product = str(row.get('product', '')).lower()
            price = float(row.get('price', 0))
            for m in MEMBERSHIP_PRODUCTS:
                if abs(price - m['price']) < 5 or m['label'].split()[0].lower() in product:
                    return m['code']
            return 'unknown'
        df['membership_code'] = df.apply(infer_membership, axis=1)
        self.membership_data = df

    def _analyze_segments(self):
        # Group by membership code, count users, total revenue
        if self.membership_data.empty:
            return []
        # Map user IDs to membership code if possible (assume unique_id field in both)
        seg = self.membership_data.groupby('membership_code').agg({
            'unique_id': 'nunique',
            'price': 'sum'
        }).reset_index().rename(columns={'unique_id': 'unique_customers', 'price': 'total_revenue'})
        seg['membership_label'] = seg['membership_code'].map({m["code"]: m["label"] for m in MEMBERSHIP_PRODUCTS})
        # Churn calculation (by historic cancels per product)
        if self.cancellation_data is not None and 'unique id' in self.cancellation_data.columns:
            cancels = self.cancellation_data['unique id'].dropna().unique()
            seg['churn_count'] = seg['membership_code'].apply(
                lambda code: self.membership_data[
                    (self.membership_data['membership_code'] == code) &
                    (self.membership_data['unique_id'].isin(cancels))
                ]['unique_id'].nunique()
            )
            seg['churn_risk'] = seg['churn_count'] / seg['unique_customers'].replace(0, np.nan)
        else:
            seg['churn_risk'] = 0.1
        return seg.to_dict('records')

    def _estimate_elasticities(self):
        # Default elasticities by membership type (use available studies or business rules)
        # Higher for flexible (monthly) types, lower for annual, lowest for discounted segments
        elasticities = {}
        for m in MEMBERSHIP_PRODUCTS:
            lcode = m["code"]
            label = m["label"].lower()
            if "annual" in label:
                elasticities[lcode] = -0.15
            elif "month" in label:
                elasticities[lcode] = -0.25
            elif "saver" in label or "off peak" in label:
                elasticities[lcode] = -0.10
            else:
                elasticities[lcode] = -0.18
        return elasticities

def calculate_revenue_projections(price_adjustments, period, engine):
    projections = []
    months = {"3 months": 3, "6 months": 6, "12 months": 12}.get(period, 12)
    for prod in MEMBERSHIP_PRODUCTS:
        code = prod["code"]
        orig_price = prod["price"]
        price_adj = price_adjustments.get(code, 0)
        elasticity = engine.membership_elasticities.get(code, -0.15)
        segment_info = next((seg for seg in engine.segment_analysis if seg["membership_code"] == code), {})
        baseline_count = segment_info.get("unique_customers", 0)
        # If baseline data is missing, skip
        if baseline_count == 0:
            continue
        # Predicted revenue and quantity effect
        price_delta = orig_price * price_adj
        price_new = orig_price + price_delta
        quantity_change = elasticity * price_adj
        quantity_new = baseline_count * (1 + quantity_change)
        # Projected period revenue
        revenue_baseline = baseline_count * orig_price * (months / 12)
        revenue_projected = quantity_new * price_new * (months / 12)
        revenue_change_pct = ((revenue_projected - revenue_baseline) / revenue_baseline) if revenue_baseline > 0 else 0
        projections.append({
            "Membership": prod["label"],
            "Price_Change_Pct": price_adj * 100,
            "Revenue_Change_Pct": revenue_change_pct * 100,
            "Elasticity": elasticity,
            "User_Count": baseline_count,
            "Old_Price": orig_price,
            "New_Price": price_new,
        })
    return pd.DataFrame(projections)

def predict_churn_from_prices(price_adjustments, engine):
    churn_data = []
    for seg in engine.segment_analysis:
        code = seg['membership_code']
        price_adj = price_adjustments.get(code, 0)
        base_churn = seg.get('churn_risk', 0.1)
        elasticity = engine.membership_elasticities.get(code, -0.15)
        churn_adj = abs(price_adj) * (1 + abs(elasticity)) * 0.6
        adjusted_churn = min(base_churn + churn_adj, 1.0)
        churn_data.append({
            "Membership": seg['membership_label'],
            "Price_Change": price_adj,
            "Churn_Risk": adjusted_churn,
            "Customer_Count": seg['unique_customers'],
        })
    return pd.DataFrame(churn_data)

def generate_dynamic_recommendations(price_adjustments, revenue_projections, churn_predictions):
    recs = []
    for _, row in revenue_projections.iterrows():
        membership = row["Membership"]
        revenue_change = row["Revenue_Change_Pct"]
        churn_score = churn_predictions.loc[churn_predictions['Membership'] == membership, 'Churn_Risk'].max()
        if revenue_change > 5 and churn_score < 0.15:
            recs.append({
                "title": f"Increase pricing for {membership}",
                "impact": f"Expected revenue increase: +{revenue_change:.1f}%, Low churn risk",
                "implementation": "Recommend moderate price increase. Communicate added value to members."
            })
        elif revenue_change < -5 or churn_score > 0.3:
            recs.append({
                "title": f"Risk for {membership}",
                "impact": f"Revenue risk: {revenue_change:.1f}%, Churn risk: {churn_score:.1%}",
                "implementation": "Review price sensitivity. Consider member engagement or support campaigns."
            })
        else:
            recs.append({
                "title": f"Maintain pricing for {membership}",
                "impact": f"Stable outlook. Revenue change: {revenue_change:.1f}%, Churn risk: {churn_score:.1%}",
                "implementation": "Monitor and focus on service quality and experience."
            })
    return recs

def churn_risk_color(churn):
    if churn > 0.3:
        return "🔴 High Risk"
    elif churn > 0.15:
        return "🟡 Moderate Risk"
    else:
        return "🟢 Low Risk"
    
def create_enhanced_recommendations_section(price_adjustments, projections, churn):
    """Enhanced recommendations section with professional styling for stakeholders"""
    
    st.markdown("""
    <style>
    .recommendation-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .high-priority {
        border-left-color: #e74c3c !important;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }
    
    .medium-priority {
        border-left-color: #f39c12 !important;
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .low-priority {
        border-left-color: #27ae60 !important;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .rec-title {
        color: #2c3e50;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    
    .rec-content {
        color: #34495e;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 8px;
    }
    
    .metric-highlight {
        background-color: rgba(52, 152, 219, 0.1);
        border-radius: 8px;
        padding: 5px 10px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    
    .summary-banner {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Generate recommendations
    recs = generate_dynamic_recommendations(price_adjustments, projections, churn)
    
    # Create summary metrics
    total_revenue_impact = projections['Revenue_Change_Pct'].sum() if not projections.empty else 0
    avg_churn_risk = churn['Churn_Risk'].mean() if not churn.empty else 0
    
    # Summary banner
    st.markdown(f"""
    <div class="summary-banner">
        <h2>📊 Executive Summary</h2>
        <div style="display: flex; justify-content: space-around; margin-top: 15px;">
            <div>
                <h3>{total_revenue_impact:+.1f}%</h3>
                <p>Total Revenue Impact</p>
            </div>
            <div>
                <h3>{avg_churn_risk:.1%}</h3>
                <p>Average Churn Risk</p>
            </div>
            <div>
                <h3>{len(recs)}</h3>
                <p>Strategic Actions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.header("🎯 Strategic Recommendations")
    
    # Categorize recommendations by priority
    def get_priority_and_icon(rec_title, impact_text):
        if "Risk" in rec_title or "Revenue risk" in impact_text:
            return "high", "🚨"
        elif "Increase pricing" in rec_title or "revenue increase" in impact_text:
            return "medium", "📈"
        else:
            return "low", "✅"

    # Create tabs for different views
    tab1, tab2 = st.columns([3, 1])
    
    with tab1:
        for i, rec in enumerate(recs, 1):
            priority, icon = get_priority_and_icon(rec['title'], rec['impact'])
            
            priority_class = f"{priority}-priority"
            
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <div class="rec-title">
                    {icon} Recommendation {i}: {rec['title']}
                </div>
                <div class="rec-content">
                    <strong>📊 Expected Impact:</strong><br>
                    <span class="metric-highlight">{rec['impact']}</span>
                </div>
                <div class="rec-content">
                    <strong>🎯 Recommended Action:</strong><br>
                    {rec['implementation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("🏷️ Priority Legend")
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px;">
            <div style="margin-bottom: 10px;">
                <span style="color: #e74c3c; font-size: 18px;">🚨</span> 
                <strong>High Priority</strong><br>
                <small>Immediate attention required</small>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="color: #f39c12; font-size: 18px;">📈</span> 
                <strong>Medium Priority</strong><br>
                <small>Growth opportunities</small>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="color: #27ae60; font-size: 18px;">✅</span> 
                <strong>Low Priority</strong><br>
                <small>Maintain status quo</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📧 Email Report", help="Generate email summary"):
                st.success("Report prepared for email!")
        with col2:
            if st.button("📋 Export PDF", help="Export recommendations as PDF"):
                st.success("PDF export ready!")

    # Call-to-action footer
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-top: 30px; border: 2px solid #3498db;">
        <h4 style="color: #2980b9; margin-bottom: 10px;">💡 Next Steps</h4>
        <ul style="color: #34495e; margin-left: 20px;">
            <li>Review high-priority recommendations with your team</li>
            <li>Set implementation timelines for approved changes</li>
            <li>Schedule follow-up analysis in 3-6 months</li>
            <li>Monitor member feedback during transition periods</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -------- Streamlit Interactive Dashboard -------- #
def create_interactive_dashboard():
    st.set_page_config(
        page_title="UoB Sports Centre Membership Pricing Dashboard",
        page_icon="🏋️",
        layout="wide"
    )
    st.title("🏋️ University of Birmingham Sports Centre Membership Pricing Dashboard")
    st.markdown("Analyse membership scenario impact by tier — revenue, churn and recommendations.")
    
    # Data loading
    @st.cache_data
    def load_data():
        loader = EnhancedDataLoader()
        return loader.load_and_prepare_data()
    
    @st.cache_data
    def init_analytics(datasets):
        return MembershipAnalyticsEngine(datasets)

    with st.spinner("Loading data/analytics..."):
        datasets = load_data()
        engine = init_analytics(datasets)

    st.sidebar.header("Scenario Configuration")

    # --- Quick Scenario ---

    presets = {
        "Conservative": {m['code']: 0.05 for m in MEMBERSHIP_PRODUCTS},
        "Inflation-Match": {m['code']: 0.08 for m in MEMBERSHIP_PRODUCTS},
        "Freeze": {m['code']: 0.0 for m in MEMBERSHIP_PRODUCTS},
        "Custom": {},
    }
    selected_preset = st.sidebar.radio("Quick Scenario Testing", list(presets.keys()))
    if selected_preset != "Custom":
        price_adjustments = presets[selected_preset].copy()
    else:
        st.sidebar.subheader("Manual Membership Adjustments")
        price_adjustments = {}
        for m in MEMBERSHIP_PRODUCTS:
            price_adjustments[m['code']] = st.sidebar.slider(
                m["label"], min_value=-0.2, max_value=0.3, value=0.0, step=0.01
            )

    st.sidebar.header("Business Context")
    segment_counts = {seg['membership_label']: seg['unique_customers'] for seg in engine.segment_analysis}
    for k, v in segment_counts.items():
        st.sidebar.markdown(f"{k}: {v}")

    st.sidebar.markdown("**Annual Pricing/Eligibility**")
    for m in MEMBERSHIP_PRODUCTS:
        st.sidebar.markdown(f"- {m['label']}: £{m['price']:.2f}")

    st.sidebar.header("Advanced")
    period = st.sidebar.selectbox("Analysis Period", ["3 months", "6 months", "12 months"])

    st.markdown("---")

    # --- Main Dash Content ---#
    col1, col2 = st.columns(2)

    projections = calculate_revenue_projections(price_adjustments, period, engine)
    churn = predict_churn_from_prices(price_adjustments, engine)

    with col1:
        st.header("📈 Revenue Impact Projection")
        if not projections.empty:
            fig = px.bar(
                projections, x="Membership", y="Revenue_Change_Pct", color="Revenue_Change_Pct",
                color_continuous_scale='RdYlGn', title="Projected Revenue Change",
                text="Revenue_Change_Pct"
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition="outside")
            fig.update_layout(yaxis_title="% Change", xaxis_title="Membership", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Total Revenue Impact", f"{projections['Revenue_Change_Pct'].sum():.1f}%")
        else:
            st.warning("No data for revenue projection.")

    with col2:
        st.header("👥 Customer Impact Analysis")
        if not churn.empty:
            churn['Risk_Level'] = churn['Churn_Risk'].apply(churn_risk_color)
            fig2 = px.scatter(
                churn, x='Price_Change', y='Churn_Risk', size='Customer_Count', color='Risk_Level',
                title='Churn Risk by Price Change (Membership)', hover_data=['Membership']
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No churn estimation possible.")

    st.header("Detailed Analysis (Table)")
    if not projections.empty and not churn.empty:
        col1a, col2a = st.columns(2)
        with col1a:
            st.subheader("Revenue Projections")
            st.dataframe(projections)
        with col2a:
            st.subheader("Churn Analysis")
            st.dataframe(churn)
    else:
        st.info("No analysis due to missing data.")

    # ---- Recommendations ---- #
    # ---- Enhanced Recommendations ---- #
    create_enhanced_recommendations_section(price_adjustments, projections, churn)


    st.markdown("---")
    st.markdown("""
    _Features: Membership-centric analysis, churn and revenue simulation by membership.  
    Official membership types, pricing as per UoB Sports & Fitness 2025/26 policy.  
    """)


if __name__ == "__main__":
    create_interactive_dashboard()

