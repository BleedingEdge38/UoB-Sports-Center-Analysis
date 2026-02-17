import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Attendance Data
gym = pd.read_excel("Att_data.xlsx", sheet_name="Gym", parse_dates=["Attendance Detail Date Time"])
gym2 = pd.read_excel("Att_data.xlsx", sheet_name="Gym 2", parse_dates=["Attendance Detail Date Time"])
pool = pd.read_excel("Att_data.xlsx", sheet_name="Pool", parse_dates=["Attendance Detail Date Time"])
reception = pd.read_excel("Att_data.xlsx", sheet_name="Reception Barrier 1", parse_dates=["Attendance Detail Date Time"])

# Combine gym sheets
gym_all = pd.concat([gym, gym2], ignore_index=True)
attendance = pd.concat([gym_all, pool, reception], ignore_index=True)

attendance['Hour'] = attendance['Attendance Detail Date Time'].dt.hour
attendance['Date'] = attendance['Attendance Detail Date Time'].dt.date
attendance['Weekday'] = attendance['Attendance Detail Date Time'].dt.strftime("%A")

## 1. Peak Hour Analysis (Congestion Patterns)
plt.figure(figsize=(14,6))
sns.countplot(data=attendance, x='Hour', hue='Attendance Detail Entry Point')
plt.title("Attendance Distribution by Hour Across All Facilities")
plt.xlabel("Hour of Day")
plt.ylabel("Visit Count")
plt.legend(title="Facility")
plt.tight_layout()
plt.show()

# 2. Capacity Utilization Rates
# Example: Assume static theoretical maximums (replace below with real ones if known)
facility_max = {
    'Gym Barrier 1': 80,
    'Gym Barrier 2': 80,
    'Gym Barrier 3': 80,
    'Pool Lhs Barrier': 60,
    'Pool Rhs Barrier': 60,
    'Reception1': 200
}
attendance['Facility_max'] = attendance['Attendance Detail Entry Point'].map(facility_max)
utilization = attendance.groupby(['Attendance Detail Entry Point', 'Hour', 'Weekday']).size().reset_index(name='Count')
utilization['Capacity Utilization %'] = utilization.apply(
    lambda x: 100 * x['Count'] / x['Attendance Detail Entry Point'].map(facility_max) if x['Attendance Detail Entry Point'] in facility_max else np.nan,
    axis=1
)
fig = px.box(utilization, x="Weekday", y="Capacity Utilization %", color="Attendance Detail Entry Point",
             title="Facility Capacity Utilization by Day and Hour (as % of theoretical maximum)")
fig.show()

# 3. Equipment/Space Turnover Analysis (Squash/Classes Bookings vs. Capacity)
squash = pd.read_excel("Bookings_data.xlsx", sheet_name="Squash", parse_dates=["Bookings Detail Start Date Time"])
classes = pd.read_excel("Bookings_data.xlsx", sheet_name="Classes", parse_dates=["Bookings Detail Start Date Time"])

squash['Hour'] = squash['Bookings Detail Start Date Time'].dt.hour
squash_turnover = squash.groupby('Hour').size()
plt.figure(figsize=(10,5))
squash_turnover.plot(kind='bar', color='orange')
plt.title("Squash Court Bookings by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Bookings")
plt.tight_layout()
plt.show()

classes['Hour'] = classes['Bookings Detail Start Date Time'].dt.hour
classes_turnover = classes.groupby('Hour').size()
plt.figure(figsize=(10,5))
classes_turnover.plot(kind='bar', color='green')
plt.title("Class Bookings by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Class Sessions")
plt.tight_layout()
plt.show()

# 4. Queue/Wait Time Analysis
# Approximate by examining delta between entries for the same user within a short window
attendance_sorted = attendance.sort_values(["Unique Key", "Attendance Detail Date Time"])
attendance_sorted['Prev Time'] = attendance_sorted.groupby('Unique Key')['Attendance Detail Date Time'].shift()
attendance_sorted['Wait Time (Minutes)'] = (attendance_sorted['Attendance Detail Date Time'] - attendance_sorted['Prev Time']).dt.total_seconds()/60
queue_hist = attendance_sorted['Wait Time (Minutes)'].dropna()[attendance_sorted['Wait Time (Minutes)'] < 60]

plt.figure(figsize=(10,5))
sns.histplot(queue_hist, bins=50, kde=True)
plt.title("Distribution of Wait Times Between Visits (<60 mins)")
plt.xlabel("Wait Time (minutes between same-user facility accesses)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5. Revenue per Square Foot/Hour, Cost per Visit (using Financial Data)
fin_snf = pd.read_excel("Financial_Data.xlsx", sheet_name="S&F", parse_dates=["Sales Detail Participation Date Time"])
fin_snf['Hour'] = fin_snf['Sales Detail Participation Date Time'].dt.hour
fin_snf['Date'] = fin_snf['Sales Detail Participation Date Time'].dt.date

revenue_hour = fin_snf.groupby(['Sales Detail Participation Site', 'Date', 'Hour'])['Sales Detail Net Amount'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=revenue_hour, x='Hour', y='Sales Detail Net Amount', hue='Sales Detail Participation Site', ci=None)
plt.title("Hourly Revenue by Facility/Site")
plt.xlabel("Hour")
plt.ylabel("Revenue (£)")
plt.tight_layout()
plt.show()

# 6. Member Engagement Score: Frequency of Facility Usage
visits_per_user = attendance.groupby("Unique Key")['Date'].nunique()
plt.figure(figsize=(12,6))
sns.histplot(visits_per_user, bins=30, color='navy')
plt.title("Frequency of Facility Usage per User (Unique Visit Days)")
plt.xlabel("Unique Visit Days per User")
plt.ylabel("Number of Members")
plt.tight_layout()
plt.show()

# 7. Cross-Facility Synergies: Members Using Multiple Facilities
facility_member_df = attendance[['Unique Key', 'Attendance Detail Entry Point']].drop_duplicates()
member_multi_facility = facility_member_df.groupby('Unique Key')['Attendance Detail Entry Point'].nunique().value_counts().sort_index()

plt.figure(figsize=(8,5))
member_multi_facility.plot(kind='bar', color='purple')
plt.title("Number of Facilities Used per Member")
plt.xlabel("Distinct Facilities Used")
plt.ylabel("Number of Members")
plt.tight_layout()
plt.show()
