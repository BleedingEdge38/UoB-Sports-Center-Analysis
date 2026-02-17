import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self):
        """Initialize the data preparation class"""
        self.attendance_gym = None
        self.attendance_gym2 = None
        self.attendance_pool = None
        self.attendance_reception1 = None
        self.attendance_reception2 = None
        self.attendance_reception3 = None
        self.attendance_reception4 = None
        self.bookings_squash = None
        self.bookings_classes = None
        self.bookings_alt_sessions = None
        self.bookings_other_activities = None
        self.financial_sf = None
        self.financial_tiverton = None
        self.financial_non_member = None
        self.cancellations = None
        self.nps = None
        
        # Prepared datasets
        self.attendance = None
        self.financial = None
        
    def load_datasets(self):
        """Load all required datasets from Excel files"""
        print("📂 Loading datasets...")
        
        # Load Attendance data
        print("  Loading attendance data...")
        self.attendance_gym = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Gym')
        self.attendance_gym2 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Gym 2')
        self.attendance_pool = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Pool')
        self.attendance_reception1 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 1')
        self.attendance_reception2 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 2')
        self.attendance_reception3 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Recpetion Barrier 3')
        self.attendance_reception4 = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Att_Data_25_G24.xlsx', sheet_name='Reception Barrier 4')

        # Load Bookings data
        print("  Loading bookings data...")
        self.bookings_squash = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Squash')
        self.bookings_classes = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Classes')
        self.bookings_alt_sessions = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Alternative sessions')
        self.bookings_other_activities = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Bookings_Data_25_G24.xlsx', sheet_name='Other Activities')

        # Load Financial data
        print("  Loading financial data...")
        self.financial_sf = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='S&F')
        self.financial_tiverton = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='Tiverton')
        self.financial_non_member = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Financial_Data_25_G24.xlsx', sheet_name='Non member payments')

        # Load Cancellation and NPS data
        print("  Loading cancellation and NPS data...")
        self.cancellations = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Cancellation_Updated_25_G24.xlsx', sheet_name='Sheet1')
        self.nps = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')

        print("✅ All datasets loaded successfully!")
    
    def prepare_data(self):
        """Clean and prepare datasets for analysis"""
        print("🔧 Preparing and cleaning data...")
        
        # Standardize column names for attendance data
        attendance_datasets = [
            self.attendance_gym, self.attendance_gym2, 
            self.attendance_pool, self.attendance_reception1, 
            self.attendance_reception2, self.attendance_reception3,
            self.attendance_reception4
        ]
        
        print("  Standardizing attendance data...")
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
        self.attendance_reception1['facility_type'] = 'Reception_Barrier_1'
        self.attendance_reception2['facility_type'] = 'Reception_Barrier_2'
        self.attendance_reception3['facility_type'] = 'Reception_Barrier_3'
        self.attendance_reception4['facility_type'] = 'Reception_Barrier_4'

        # Combine all attendance data
        print("  Combining attendance data...")
        self.attendance = pd.concat([
            self.attendance_gym, self.attendance_gym2, 
            self.attendance_pool, self.attendance_reception1,
            self.attendance_reception2, self.attendance_reception3,
            self.attendance_reception4
        ], ignore_index=True)
        
        # Convert date columns
        print("  Converting date columns...")
        self.attendance['date'] = pd.to_datetime(self.attendance['date'])
        self.attendance['date_time'] = pd.to_datetime(self.attendance['date_time'])
        
        # Prepare financial data
        print("  Preparing financial data...")
        self.financial = pd.concat([
            self.financial_sf, self.financial_tiverton, self.financial_non_member
        ], ignore_index=True)
        
        # Clean financial column names
        self.financial.columns = self.financial.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Prepare cancellation data
        print("  Preparing cancellation data...")
        self.cancellations.columns = self.cancellations.columns.str.strip().str.lower().str.replace(' ', '_')
        self.cancellations['effective_from'] = pd.to_datetime(self.cancellations['effective_from'], errors='coerce')
        
        # Prepare NPS data
        print("  Preparing NPS data...")
        self.nps['Response Date'] = pd.to_datetime(self.nps['Response Date'], format='%d/%m/%Y')
        
        print("✅ Data preparation completed!")
    
    def save_prepared_data(self, output_dir='F:/UoB Study/Capstone Project/Final Project/Datasets/prepared_data'):
        """Save prepared datasets to files"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving prepared data to {output_dir}/...")
        
        # Save main prepared datasets
        datasets_to_save = {
            'attendance': self.attendance,
            'financial': self.financial,
            'cancellations': self.cancellations,
            'nps': self.nps,
            'bookings_squash': self.bookings_squash,
            'bookings_classes': self.bookings_classes,
            'bookings_alt_sessions': self.bookings_alt_sessions,
            'other_activities': self.bookings_other_activities
        }
        
        for name, dataset in datasets_to_save.items():
            if dataset is not None and not dataset.empty:
                # Save as pickle for faster loading and preserving data types
                with open(f'{output_dir}/{name}.pkl', 'wb') as f:
                    pickle.dump(dataset, f)
                print(f"  ✅ Saved {name}.pkl")
        
        # Save data info for reference
        data_info = {
            'preparation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'attendance_shape': self.attendance.shape if self.attendance is not None else None,
            'financial_shape': self.financial.shape if self.financial is not None else None,
            'cancellations_shape': self.cancellations.shape if self.cancellations is not None else None,
            'nps_shape': self.nps.shape if self.nps is not None else None,
            'date_range': {
                'min_date': self.attendance['date'].min().strftime('%Y-%m-%d') if self.attendance is not None else None,
                'max_date': self.attendance['date'].max().strftime('%Y-%m-%d') if self.attendance is not None else None
            }
        }
        
        with open(f'{output_dir}/data_info.pkl', 'wb') as f:
            pickle.dump(data_info, f)
        
        print("✅ All prepared data saved successfully!")
        return data_info
    
    def run_preparation(self):
        """Run the complete data preparation pipeline"""
        print("🚀 Starting Data Preparation Pipeline...")
        print("=" * 50)
        
        # Load datasets
        self.load_datasets()
        
        # Prepare data
        self.prepare_data()
        
        # Save prepared data
        data_info = self.save_prepared_data()
        
        print("\n📊 DATA PREPARATION SUMMARY:")
        print("-" * 40)
        print(f"Preparation completed: {data_info['preparation_date']}")
        print(f"Attendance records: {data_info['attendance_shape']}")
        print(f"Financial records: {data_info['financial_shape']}")
        print(f"Cancellation records: {data_info['cancellations_shape']}")
        print(f"NPS records: {data_info['nps_shape']}")
        print(f"Date range: {data_info['date_range']['min_date']} to {data_info['date_range']['max_date']}")
        
        print("\n✅ Data preparation pipeline completed successfully!")
        print("📁 Prepared data files saved in 'prepared_data' directory")
        print("🔄 You can now run the analytics code without reloading data")

# Run the data preparation
if __name__ == "__main__":
    prep = DataPreparation()
    prep.run_preparation()
