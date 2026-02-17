import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

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
    return df

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
        # Drop any new duplicates after renaming!
        df = drop_duplicate_columns(df)

    return df

class PredictiveChurnModel:
    def __init__(self, cache_dir='./data_cache', pkl_data_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pkl_data_dir = pkl_data_dir
        
        # Data containers
        self.financial_data = None
        self.attendance_data = None
        self.user_data = None
        self.cancellation_data = None
        self.nps_data = None
        self.qualitative_feedback = None
        self.booking_data = None
        
        # Model containers
        self.churn_features = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Primary member profiles to focus on
        self.primary_profiles = [
            'Student Member', 'Staff Member', 'Alumni Member',
            'Community Member', 'Associate Member',
            'Community Over 65 Member', 'Staff Non Member'
        ]

    def load_pkl_datasets(self):
        """Load datasets from pickle files with fallback to Excel"""
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
                pkl_path = f'{self.pkl_data_dir}/{filename}'
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        datasets[name] = pickle.load(f)
                    print(f"✅ Loaded {name} data from {filename}")
                else:
                    print(f"⚠️ {filename} not found, will try alternative loading...")
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")

        # Load qualitative feedback
        try:
            qualitative_df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')
            datasets['qualitative'] = qualitative_df
            print(f"✅ Loaded qualitative feedback: {len(qualitative_df)} records")
        except Exception as e:
            print(f"❌ Error loading qualitative feedback: {str(e)}")

        return datasets

    def load_excel_fallback(self, dataset_name):
        """Fallback method to load data from Excel files"""
        try:
            if dataset_name == 'financial':
                financial_sheets = pd.read_excel('Financial_Data.xlsx', sheet_name=['S&F', 'Non Member Payments'])
                df = pd.concat([financial_sheets['S&F'], financial_sheets['Non Member Payments']], ignore_index=True)
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'attendance':
                attendance_sheets = pd.read_excel('Att_data.xlsx',
                    sheet_name=['Gym', 'Gym 2','Reception Barrier 1','Reception Barrier 2','Recpetion Barrier 3', 'Reception Barrier 4'])
                dfs = []
                for sheet_name, sheet_data in attendance_sheets.items():
                    if sheet_data is not None:
                        dfs.append(sheet_data)
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'user':
                df = pd.read_excel('User_data.xlsx', sheet_name='Sheet1')
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'booking':
                booking_sheets = pd.read_excel('Bookings_data.xlsx', sheet_name=['Squash', 'Classes'])
                df = pd.concat([booking_sheets['Squash'], booking_sheets['Classes']], ignore_index=True)
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'cancellation':
                df = pd.read_excel('Cancellation_Updated_25_G24.xlsx')
                df = drop_duplicate_columns(df)
                return df
            elif dataset_name == 'nps':
                df = pd.read_csv('NPS_Updated_25_G24.csv')
                df = drop_duplicate_columns(df)
                return df
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} from Excel: {e}")
            return None

    def load_and_preprocess_data(self, force_reload=False):
        """Load and preprocess all required datasets"""
        print("\n🔄 Starting data loading for churn prediction model...")
        
        try:
            # Load from PKL files first
            pkl_datasets = self.load_pkl_datasets()
            
            # Load financial data
            print("📈 Loading financial data...")
            if 'financial' in pkl_datasets and pkl_datasets['financial'] is not None:
                self.financial_data = pkl_datasets['financial']
                if isinstance(self.financial_data, dict):
                    dfs = []
                    for sheet_name, sheet_data in self.financial_data.items():
                        if sheet_data is not None:
                            dfs.append(sheet_data)
                    self.financial_data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                self.financial_data = self.load_excel_fallback('financial')
            
            if self.financial_data is not None:
                self.financial_data = drop_duplicate_columns(self.financial_data)
                self.financial_data = drop_duplicate_index(self.financial_data)
                self.financial_data = canonicalize_financial_columns(self.financial_data)
                self.financial_data = drop_duplicate_columns(self.financial_data)

            # Load attendance data
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
            
            if self.attendance_data is not None:
                self.attendance_data = drop_duplicate_columns(self.attendance_data)
                self.attendance_data = drop_duplicate_index(self.attendance_data)
                self.attendance_data = canonicalize_financial_columns(self.attendance_data)
                self.attendance_data = drop_duplicate_columns(self.attendance_data)

            # Load user data
            print("👤 Loading user profile data...")
            self.user_data = self.load_excel_fallback('user')
            if self.user_data is not None:
                self.user_data = drop_duplicate_columns(self.user_data)
                self.user_data = drop_duplicate_index(self.user_data)

            # Load cancellation data
            print("❌ Loading cancellation data...")
            if 'cancellation' in pkl_datasets and pkl_datasets['cancellation'] is not None:
                self.cancellation_data = pkl_datasets['cancellation']
            else:
                self.cancellation_data = self.load_excel_fallback('cancellation')
            
            if self.cancellation_data is not None:
                self.cancellation_data = drop_duplicate_columns(self.cancellation_data)
                self.cancellation_data = drop_duplicate_index(self.cancellation_data)

            # Load NPS data
            print("⭐ Loading NPS data...")
            if 'nps' in pkl_datasets and pkl_datasets['nps'] is not None:
                self.nps_data = pkl_datasets['nps']
            else:
                self.nps_data = self.load_excel_fallback('nps')

            # Load qualitative feedback
            if 'qualitative' in pkl_datasets:
                self.qualitative_feedback = pkl_datasets['qualitative']

            # Load booking data
            print("📅 Loading booking data...")
            self.booking_data = self.load_excel_fallback('booking')

            # Filter for primary profiles
            self._filter_primary_profiles()
            
            # Preprocess dates
            print("📅 Preprocessing dates...")
            self._preprocess_dates()
            
            print("✅ Data loaded and preprocessed successfully!")
            
        except Exception as e:
            print(f"❌ Error during data loading: {e}")
            raise

    def _filter_primary_profiles(self):
        """Filter data for primary member profiles"""
        print("🎯 Filtering data for primary member profiles...")
        
        if self.financial_data is not None:
            self.financial_data = drop_duplicate_columns(self.financial_data)
            initial_financial_count = len(self.financial_data)
            if 'customer_type' in self.financial_data.columns:
                self.financial_data = self.financial_data[
                    self.financial_data['customer_type'].isin(self.primary_profiles)
                ]
                self.financial_data = drop_duplicate_columns(self.financial_data)
                print(f" Financial data: {initial_financial_count} → {len(self.financial_data)} records")

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
                print(f" User data: {initial_user_count} → {len(self.user_data)} records")

    def _preprocess_dates(self):
        """Preprocess date columns"""
        try:
            if self.financial_data is not None and 'date' in self.financial_data.columns:
                self.financial_data['date'] = pd.to_datetime(self.financial_data['date'], errors='coerce')
            
            if self.attendance_data is not None and 'date' in self.attendance_data.columns:
                self.attendance_data['date'] = pd.to_datetime(self.attendance_data['date'], errors='coerce')
            
            if self.cancellation_data is not None:
                for col in self.cancellation_data.columns:
                    if 'date' in col.lower():
                        self.cancellation_data[col] = pd.to_datetime(self.cancellation_data[col], errors='coerce')
            
            if self.nps_data is not None:
                for col in self.nps_data.columns:
                    if 'date' in col.lower():
                        self.nps_data[col] = pd.to_datetime(self.nps_data[col], errors='coerce')
                        
        except Exception as e:
            print(f"⚠️ Warning during date preprocessing: {e}")

    def create_churn_features(self):
        """Create features for churn prediction"""
        print("🔧 Creating churn features...")
        
        # Define churn based on cancellation data
        churned_members = set()
        if self.cancellation_data is not None and 'Unique ID' in self.cancellation_data.columns:
            churned_members = set(self.cancellation_data['Unique ID'].dropna())
        
        # Get all unique members from financial data
        if self.financial_data is None or 'unique_id' not in self.financial_data.columns:
            print("❌ Financial data or unique_id column not available")
            return None
        
        all_members = self.financial_data['unique_id'].unique()
        
        # Create feature dataframe
        features_list = []
        
        for member_id in all_members:
            if pd.isna(member_id):
                continue
                
            member_data = {}
            member_data['unique_id'] = member_id
            member_data['churned'] = 1 if member_id in churned_members else 0
            
            # Financial features
            member_financial = self.financial_data[self.financial_data['unique_id'] == member_id]
            if not member_financial.empty:
                member_data['total_revenue'] = member_financial['price'].sum() if 'price' in member_financial.columns else 0
                member_data['avg_transaction_amount'] = member_financial['price'].mean() if 'price' in member_financial.columns else 0
                member_data['transaction_count'] = len(member_financial)
                member_data['customer_type'] = member_financial['customer_type'].iloc[0] if 'customer_type' in member_financial.columns else 'Unknown'
                
                # Recency of last transaction
                if 'date' in member_financial.columns:
                    last_transaction = member_financial['date'].max()
                    if pd.notna(last_transaction):
                        days_since_last_transaction = (datetime.now() - last_transaction).days
                        member_data['days_since_last_transaction'] = days_since_last_transaction
                    else:
                        member_data['days_since_last_transaction'] = 999
                else:
                    member_data['days_since_last_transaction'] = 999
            else:
                member_data['total_revenue'] = 0
                member_data['avg_transaction_amount'] = 0
                member_data['transaction_count'] = 0
                member_data['customer_type'] = 'Unknown'
                member_data['days_since_last_transaction'] = 999
            
            # Attendance features
            if self.attendance_data is not None and 'unique_id' in self.attendance_data.columns:
                member_attendance = self.attendance_data[self.attendance_data['unique_id'] == member_id]
                member_data['total_visits'] = len(member_attendance)
                
                if not member_attendance.empty and 'date' in member_attendance.columns:
                    # Visit frequency
                    visit_dates = member_attendance['date'].dropna()
                    if len(visit_dates) > 1:
                        date_range = (visit_dates.max() - visit_dates.min()).days
                        member_data['visit_frequency'] = len(visit_dates) / max(date_range, 1) * 30  # visits per month
                        member_data['days_since_last_visit'] = (datetime.now() - visit_dates.max()).days
                    else:
                        member_data['visit_frequency'] = 0
                        member_data['days_since_last_visit'] = 999
                else:
                    member_data['visit_frequency'] = 0
                    member_data['days_since_last_visit'] = 999
            else:
                member_data['total_visits'] = 0
                member_data['visit_frequency'] = 0
                member_data['days_since_last_visit'] = 999
            
            # User profile features
            if self.user_data is not None and 'Unique ID' in self.user_data.columns:
                member_profile = self.user_data[self.user_data['Unique ID'] == member_id]
                if not member_profile.empty:
                    member_data['age'] = member_profile['Contacts Detail Age'].iloc[0] if 'Contacts Detail Age' in member_profile.columns else 0
                    member_data['gender'] = member_profile['Contacts Detail Gender'].iloc if 'Contacts Detail Gender' in member_profile.columns else 'Unknown'
                    member_data['fitness_goal'] = member_profile['Fitnesss Goal'].iloc if 'Fitnesss Goal' in member_profile.columns else 'Unknown'
                    member_data['fitness_level'] = member_profile['Fitness Level'].iloc if 'Fitness Level' in member_profile.columns else 'Unknown'
                else:
                    member_data['age'] = 0
                    member_data['gender'] = 'Unknown'
                    member_data['fitness_goal'] = 'Unknown'
                    member_data['fitness_level'] = 'Unknown'
            else:
                member_data['age'] = 0
                member_data['gender'] = 'Unknown'
                member_data['fitness_goal'] = 'Unknown'
                member_data['fitness_level'] = 'Unknown'
            
            # NPS features
            if self.nps_data is not None and 'Unique ID' in self.nps_data.columns:
                member_nps = self.nps_data[self.nps_data['Unique ID'] == member_id]
                if not member_nps.empty and 'Score' in member_nps.columns:
                    member_data['avg_nps_score'] = member_nps['Score'].mean()
                    member_data['nps_responses'] = len(member_nps)
                else:
                    member_data['avg_nps_score'] = 5  # neutral
                    member_data['nps_responses'] = 0
            else:
                member_data['avg_nps_score'] = 5  # neutral
                member_data['nps_responses'] = 0
            
            # Booking features
            if self.booking_data is not None and 'Unique' in self.booking_data.columns:
                member_bookings = self.booking_data[self.booking_data['Unique'] == member_id]
                member_data['total_bookings'] = len(member_bookings)
                if not member_bookings.empty:
                    attended_bookings = member_bookings[member_bookings.get('Bookings Detail Attended', '') == 'Yes']
                    member_data['booking_attendance_rate'] = len(attended_bookings) / len(member_bookings) if len(member_bookings) > 0 else 0
                else:
                    member_data['booking_attendance_rate'] = 0
            else:
                member_data['total_bookings'] = 0
                member_data['booking_attendance_rate'] = 0
            
            features_list.append(member_data)
        
        # Create DataFrame
        self.churn_features = pd.DataFrame(features_list)
        
        # Handle missing values
        numeric_columns = ['total_revenue', 'avg_transaction_amount', 'transaction_count', 
                          'days_since_last_transaction', 'total_visits', 'visit_frequency',
                          'days_since_last_visit', 'age', 'avg_nps_score', 'nps_responses',
                          'total_bookings', 'booking_attendance_rate']
        
        for col in numeric_columns:
            if col in self.churn_features.columns:
                self.churn_features[col] = pd.to_numeric(self.churn_features[col], errors='coerce').fillna(0)
        
        print(f"✅ Created features for {len(self.churn_features)} members")
        print(f"Churn rate: {self.churn_features['churned'].mean():.2%}")
        
        return self.churn_features

    def prepare_model_data(self):
        """Prepare data for model training"""
        if self.churn_features is None:
            print("❌ No features available. Run create_churn_features() first.")
            return None, None
        
        # Encode categorical variables
        le_customer_type = LabelEncoder()
        le_gender = LabelEncoder()
        le_fitness_goal = LabelEncoder()
        le_fitness_level = LabelEncoder()
        
        model_data = self.churn_features.copy()
        
        # Handle categorical encoding
        model_data['customer_type_encoded'] = le_customer_type.fit_transform(model_data['customer_type'].astype(str))
        model_data['gender_encoded'] = le_gender.fit_transform(model_data['gender'].astype(str))
        model_data['fitness_goal_encoded'] = le_fitness_goal.fit_transform(model_data['fitness_goal'].astype(str))
        model_data['fitness_level_encoded'] = le_fitness_level.fit_transform(model_data['fitness_level'].astype(str))
        
        # Select features for modeling
        feature_columns = [
            'total_revenue', 'avg_transaction_amount', 'transaction_count',
            'days_since_last_transaction', 'total_visits', 'visit_frequency',
            'days_since_last_visit', 'age', 'avg_nps_score', 'nps_responses',
            'total_bookings', 'booking_attendance_rate',
            'customer_type_encoded', 'gender_encoded', 'fitness_goal_encoded', 'fitness_level_encoded'
        ]
        
        X = model_data[feature_columns]
        y = model_data['churned']
        
        # Handle any remaining missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        return X_imputed, y

    def train_churn_model(self):
        """Train the churn prediction model"""
        print("🤖 Training churn prediction models...")
        
        X, y = self.prepare_model_data()
        if X is None or y is None:
            return None
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                proba_output = model.predict_proba(X_test)
                if proba_output.shape[1] > 1:
                    y_pred_proba = proba_output[:, 1]  # Probability of positive class
                else:
                    # Handle single class case
                    print(f"Warning: Model only learned one class: {model.classes_}")
                    y_pred_proba = model.predict(X_test).astype(float)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_data': (X_test_scaled if name == 'Logistic Regression' else X_test, y_test)
            }
            
            print(f"{name} AUC Score: {auc_score:.3f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        print(f"✅ Best model: {best_model_name} (AUC: {best_score:.3f})")
        
        return results, best_model_name, (X_test, y_test)

    def plot_model_results(self, results, best_model_name):
        """Plot model performance results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, result in results.items():
            X_test, y_test = result['test_data']
            y_pred_proba = result['probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = result['auc_score']
            
            linewidth = 3 if name == best_model_name else 1
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=linewidth)
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Churn Prediction Models')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature Importance (for best model)
        ax2 = axes[0, 1]
        if hasattr(self.model, 'feature_importances_'):
            X, _ = self.prepare_model_data()
            feature_names = X.columns
            importances = self.model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            top_10 = indices[:10]
            
            ax2.bar(range(len(top_10)), importances[top_10])
            ax2.set_xticks(range(len(top_10)))
            ax2.set_xticklabels([feature_names[i] for i in top_10], rotation=45, ha='right')
            ax2.set_title(f'Top 10 Feature Importance - {best_model_name}')
            ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax3 = axes[1, 0]
        best_result = results[best_model_name]
        _, y_test = best_result['test_data']
        y_pred = best_result['predictions']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model_name}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Churn Risk Distribution
        ax4 = axes[1, 1]
        churn_probs = best_result['probabilities']
        
        ax4.hist(churn_probs[y_test == 0], bins=30, alpha=0.5, label='Non-Churners', density=True)
        ax4.hist(churn_probs[y_test == 1], bins=30, alpha=0.5, label='Churners', density=True)
        ax4.set_xlabel('Churn Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Churn Risk Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('churn_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def identify_high_risk_members(self, threshold=0.7):
        """Identify members at high risk of churning"""
        if self.model is None:
            print("❌ No trained model available")
            return None
        
        X, _ = self.prepare_model_data()
        
        if hasattr(self.model, 'predict_proba'):
            if isinstance(self.model, LogisticRegression):
                X_scaled = self.scaler.transform(X)
                churn_probabilities = self.model.predict_proba(X_scaled)[:, 1]
            else:
                churn_probabilities = self.model.predict_proba(X)[:, 1]
        else:
            print("❌ Model doesn't support probability prediction")
            return None
        
        # Add probabilities to feature dataframe
        risk_df = self.churn_features.copy()
        risk_df['churn_probability'] = churn_probabilities
        
        # Identify high-risk members
        high_risk = risk_df[
            (risk_df['churn_probability'] >= threshold) & 
            (risk_df['churned'] == 0)  # Only current members
        ].sort_values('churn_probability', ascending=False)
        
        print(f"🚨 Identified {len(high_risk)} high-risk members (threshold: {threshold})")
        
        return high_risk

    def generate_churn_insights(self):
        """Generate insights about churn patterns"""
        if self.churn_features is None:
            print("❌ No features available")
            return
        
        print("\n" + "="*60)
        print("📊 CHURN ANALYSIS INSIGHTS")
        print("="*60)
        
        # Overall churn rate
        churn_rate = self.churn_features['churned'].mean()
        print(f"Overall Churn Rate: {churn_rate:.2%}")
        
        # Churn by customer type
        print("\n📋 Churn Rate by Customer Type:")
        churn_by_type = self.churn_features.groupby('customer_type')['churned'].agg(['mean', 'count']).round(3)
        churn_by_type.columns = ['Churn_Rate', 'Count']
        churn_by_type = churn_by_type.sort_values('Churn_Rate', ascending=False)
        print(churn_by_type)
        
        # Key differences between churners and non-churners
        print("\n📊 Key Metrics - Churners vs Non-Churners:")
        churners = self.churn_features[self.churn_features['churned'] == 1]
        non_churners = self.churn_features[self.churn_features['churned'] == 0]
        
        metrics = ['total_revenue', 'avg_transaction_amount', 'total_visits', 
                  'visit_frequency', 'days_since_last_visit', 'avg_nps_score']
        
        comparison = pd.DataFrame({
            'Churners': churners[metrics].mean(),
            'Non_Churners': non_churners[metrics].mean()
        }).round(2)
        
        comparison['Difference'] = (comparison['Churners'] - comparison['Non_Churners']).round(2)
        print(comparison)
        
        print("="*60)

    def save_model(self, filename='churn_model.pkl'):
        """Save the trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.prepare_model_data()[0].columns.tolist()
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved to {filename}")
        else:
            print("❌ No trained model to save")

    def load_model(self, filename='churn_model.pkl'):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"✅ Model loaded from {filename}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def create_churn_dashboard(self):
        """Create comprehensive churn prediction dashboard"""
        print("\n🎯 Creating Churn Prediction Dashboard...")
        
        # Create features
        self.create_churn_features()
        
        # Train model
        results, best_model_name, test_data = self.train_churn_model()
        
        # Plot results
        self.plot_model_results(results, best_model_name)
        
        # Generate insights
        self.generate_churn_insights()
        
        # Identify high-risk members
        high_risk_members = self.identify_high_risk_members()
        
        if high_risk_members is not None and not high_risk_members.empty:
            print(f"\n🚨 Top 10 High-Risk Members:")
            risk_summary = high_risk_members[['unique_id', 'customer_type', 'churn_probability', 
                                            'total_revenue', 'days_since_last_visit']].head(10)
            print(risk_summary.to_string(index=False))
        
        # Save model
        self.save_model()
        
        print("\n✅ Churn prediction dashboard completed!")
        print("📁 Files generated:")
        print(" - churn_model_results.png")
        print(" - churn_model.pkl")

def main():
    """Main execution function"""
    print("🤖 UoB Sports & Fitness Centre - Predictive Churn Model")
    print("🎯 Focus: PRIMARY MEMBER PROFILES ONLY")
    print("📊 Strategy: Data-Driven Churn Prevention")
    print("=" * 80)
    
    # Initialize model
    churn_model = PredictiveChurnModel()
    
    try:
        # Load and preprocess data
        churn_model.load_and_preprocess_data()
        
        # Create dashboard
        churn_model.create_churn_dashboard()
        
        print("✅ Churn prediction analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Additional utility functions
def quick_churn_analysis():
    """Quick churn analysis function"""
    churn_model = PredictiveChurnModel()
    churn_model.load_and_preprocess_data()
    return churn_model.create_churn_features()

def get_high_risk_members():
    """Get high-risk members for immediate action"""
    churn_model = PredictiveChurnModel()
    churn_model.load_and_preprocess_data()
    churn_model.create_churn_features()
    churn_model.train_churn_model()
    return churn_model.identify_high_risk_members()
