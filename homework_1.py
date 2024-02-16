
#I'll be starting be importing any libraries I need.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#starting with reading each csv file into a dataframe, and later joining them
df_2024 = pd.read_csv('https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/21c83b32-d5a8-4106-a54f-010dbe49f6f2/resource/ffd20867-6e3c-4074-8427-d63810edf231/download/Daily%20shelter%20overnight%20occupancy.csv')
df_2024.head(2)

# I ran into some issues changing the occupancy_date column to a date format, 
# so I decided to do so with each individual csv to be able to determine the issue easier

df_2024['OCCUPANCY_DATE'] = pd.to_datetime(df_2024['OCCUPANCY_DATE'])
df_2024['OCCUPANCY_DATE']

df_2023['OCCUPANCY_DATE']= df_2023['OCCUPANCY_DATE'].str.split('T').str[0]
df_2023

df_2023['OCCUPANCY_DATE'] = pd.to_datetime(df_2023['OCCUPANCY_DATE'])
df_2023['OCCUPANCY_DATE']

df_2022 = pd.read_csv('https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/21c83b32-d5a8-4106-a54f-010dbe49f6f2/resource/55d58477-50f5-490c-8da8-5953e3b26ca4/download/daily-shelter-overnight-service-occupancy-capacity-2022.csv')
df_2022

#The issue seems to be python misinturperting the 2 digit years when converting to the date format
#I'll see if adding 20 to the beginning of the year will fix that.

df_2022['OCCUPANCY_DATE'] = '20'+ df_2022['OCCUPANCY_DATE']
df_2022.head()

df_2022['OCCUPANCY_DATE'] = pd.to_datetime(df_2022['OCCUPANCY_DATE'])
df_2022['OCCUPANCY_DATE']

df_2021 = pd.read_csv('https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/21c83b32-d5a8-4106-a54f-010dbe49f6f2/resource/df7d621d-a7a0-4854-81b9-8a6dc29d73a6/download/daily-shelter-overnight-service-occupancy-capacity-2021.csv')
df_2021.head(2)

df_2021['OCCUPANCY_DATE'] = '20'+ df_2021['OCCUPANCY_DATE']
df_2021['OCCUPANCY_DATE'] = pd.to_datetime(df_2021['OCCUPANCY_DATE'])
df_2021['OCCUPANCY_DATE']

#now we'll concat the dataframes into a single dataframe and ignore their original indexes 

df = pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)

df.head()
df.tail()
df.columns
df.dtypes
df.isna().sum()
df.shape
df.describe(exclude='object')

#the iloc[0] at the end is so that we only look at the first mode in each column in case there are more than one.

df.select_dtypes(include=['object']).mode().iloc[0]

df.select_dtypes(include=['object']).nunique()

df=df.rename(columns=str.lower)
df.columns

df = df.rename(columns={'_id':'id'})
df.columns

df['organization_name'].unique()

df['organization_name'].value_counts()

df['occupancy_date']=df['occupancy_date'].astype('str')
df['occupancy_date']
df['occupancy_date']=pd.to_datetime(df['occupancy_date'])
df['occupancy_date']

df.to_json('assignment_2.json')

# we'll create a month column and see which month has the most occurances

df['month']=df['occupancy_date'].dt.month
df.dtypes

df['month'].nunique()

df['month'].value_counts().sort_values

# I'll remove the location_province column since it only has a single value 
del df['location_province']
df.columns

org_name= 'City of Toronto'
df_cot = df.query('organization_name == @org_name')
df_new_1 = df_cot[['month','organization_name','location_name','program_name','program_model','occupied_beds','occupied_rooms']]
df_new_1.head(10)

df_new_2 = df.loc[df['organization_name'] == 'City of Toronto', ['month', 'organization_name', 'location_name', 'program_name', 'program_model', 'occupied_beds', 'occupied_rooms']]
df_new_2

df_nulls = df[df.isna().any(axis=1)]
df_nulls.head()

df_nulls_subset = df[pd.isna(df['occupied_beds'])]
df_nulls_subset.head()

df.isna().sum()

nan_df = df.isnull()
sns.heatmap(nan_df)

nan_df = df.isnull()
sns.heatmap(nan_df)
plt.savefig("nulls_heatmap.pdf", format="pdf")
plt.show()

# looking at a heatmap of null values gives us great insight into where we are missing data.

# it seems like wherever I'm missing data in the 'capacity_actual_bed', 'capacity_funding_bed',
# 'occupied_beds', 'unoccupied_beds' 'unavailable_beds', and 'occupancy_rate_beds' columns;
# I have information the columns referring to rooms. Therefor I will not be getting rid of the Nans.

#  for the 'shelter_group', 'location_id', 'location_name', 'location_address', 'location_postal_code',
# 'location_city', 'program_name', 'program_model', 'overnight_service_type' and 'program_area'
# columns I will be removing the records containing nulls as they are a small fraction of all the records
# and will not impact results much.

columns_to_check = ['shelter_group', 'location_id', 'location_name', 'location_address', 'location_postal_code', 'location_city', 'program_name', 'program_model', 'overnight_service_type', 'program_area']
df.dropna(subset=columns_to_check, inplace=True)

nan_df = df.isnull()
sns.heatmap(nan_df)
plt.savefig("nulls_heatmap_after.pdf", format="pdf")
plt.show()

df_daily = df.groupby('occupancy_date')
df_daily.sum()

df.to_csv('homework1.csv')

summary = df.groupby('occupancy_date').agg({
    'occupied_beds': ['sum', 'mean', 'std'],
    'unoccupied_beds': ['sum', 'mean'],
    'capacity_actual_room': ['max', 'min'],
    'occupancy_rate_beds': ['mean', 'max'],
})

print('summary')


# Extracting the specific aggregated data for plotting
occupied_beds_data = summary['occupied_beds']


# Creating subplots for each metric of 'occupied_beds' against the date
fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

# Plot for sum
axes[0].plot(occupied_beds_data.index, occupied_beds_data['sum'], label='Sum', color='blue')
axes[0].set_title('Sum of Occupied Beds Over Time')
axes[0].legend()
axes[0].grid(True)

# Plot for mean
axes[1].plot(occupied_beds_data.index, occupied_beds_data['mean'], label='Mean', color='green')
axes[1].set_title('Mean of Occupied Beds Over Time')
axes[1].legend()
axes[1].grid(True)

# Plot for standard deviation
axes[2].plot(occupied_beds_data.index, occupied_beds_data['std'], label='Standard Deviation', color='red')
axes[2].set_title('Standard Deviation of Occupied Beds Over Time')
axes[2].legend()
axes[2].grid(True)

# Setting common labels and adjusting layout
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to not cut off labels

plt.show()
plt.savefig("occupied_beds_overtime.pdf", format="pdf")
