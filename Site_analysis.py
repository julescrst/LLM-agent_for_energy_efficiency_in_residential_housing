import pandas as pd 
import numpy as np
import os 
import joblib
from datetime import datetime
import pandas as pd 
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import sys
import argparse
import plotly.graph_objects as go 
warnings.filterwarnings('ignore')



###### LOAD DATA FOR A SINGLE SITE ######

# Base path and filenames
base_root = "residential/processed"
interval = "60min"
load_fname = f"data_pivot_{interval}.csv"
temp_fname = f"temperature_pivot_{interval}.csv"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze energy consumption for a residential site')
parser.add_argument('--site', type=str, help='Site ID to analyze (e.g., 727)')
parser.add_argument('--state', type=str, help='State code (e.g., WA, OR, ID, MT)')
parser.add_argument('--list', action='store_true', help='List all available sites')
parser.add_argument('--all', action='store_true', help='Process all sites in the directory')

args = parser.parse_args()

# Handle --list option to show available sites
if args.list:
    print("\n=== AVAILABLE SITES ===")
    for state in os.listdir(base_root):
        state_path = os.path.join(base_root, state)
        if os.path.isdir(state_path) and not state.startswith('.'):
            sites = [d for d in os.listdir(state_path) 
                    if os.path.isdir(os.path.join(state_path, d)) and not d.startswith('.')]
            if sites:
                print(f"\n{state}: ({len(sites)} sites)")
                for site in sorted(sites)[:10]:  # Show first 10
                    print(f"  - {site}")
                if len(sites) > 10:
                    print(f"  ... and {len(sites)-10} more")
    print("\n")
    sys.exit(0)

# Determine target site from arguments or prompt user
if args.site and args.state:
    target_state = args.state.upper()
    target_site = args.site
    print(f"\n=== ANALYZING SITE: {target_state}/{target_site} ===\n")
elif args.site or args.state:
    print("\n✗ Error: Both --site and --state are required")
    print("\nUsage examples:")
    print("  python Site_analysis.py --site 727 --state WA")
    print("  python Site_analysis.py --list")
    sys.exit(1)
else:
    # No arguments - use default or prompt
    print("\n=== SITE ANALYSIS TOOL ===")
    print("No arguments provided. You can:")
    print("  1. Use default site (ID/2631)")
    print("  2. Enter a custom site")
    print("  3. List available sites")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == "2":
        target_state = input("Enter state code (WA, OR, ID, MT): ").strip().upper()
        target_site = input("Enter site ID: ").strip()
        print(f"\n=== ANALYZING SITE: {target_state}/{target_site} ===\n")
    elif choice == "3":
        print("\n=== AVAILABLE SITES ===")
        for state in os.listdir(base_root):
            state_path = os.path.join(base_root, state)
            if os.path.isdir(state_path) and not state.startswith('.'):
                sites = [d for d in os.listdir(state_path) 
                        if os.path.isdir(os.path.join(state_path, d)) and not d.startswith('.')]
                if sites:
                    print(f"\n{state}: ({len(sites)} sites)")
                    for site in sorted(sites)[:10]:  # Show first 10
                        print(f"  - {site}")
                    if len(sites) > 10:
                        print(f"  ... and {len(sites)-10} more")
        sys.exit(0)
    else:
        # Default site
        target_state = "ID"
        target_site = "2631"
        print(f"\n=== USING DEFAULT SITE: {target_state}/{target_site} ===\n")

# Target site (now set by user input or command-line args)

site_data = {}
total_sites = 0

# Path to that specific site
state_path = os.path.join(base_root, target_state)
site_path = os.path.join(state_path, target_site)

if os.path.isdir(site_path):
    print(f"Processing single site: {target_state}/{target_site}")

    load_path = os.path.join(site_path, "loads", load_fname)
    temp_path = os.path.join(site_path, "temps", temp_fname)
    weather_path = os.path.join(site_path, "weather", "weather_station.csv")

    if os.path.exists(load_path):
        try:
            load_df = pd.read_csv(load_path)
        except Exception as e:
            print(f"! Failed to read loads for {target_state}/{target_site}: {e}")
            load_df = None
    else:
        print(f"! Missing load file: {load_path}")
        load_df = None

    temp_df, weather_df = None, None

    if os.path.exists(temp_path):
        try:
            temp_df = pd.read_csv(temp_path)
        except Exception as e:
            print(f"! Warning: could not read temps ({e})")

    if os.path.exists(weather_path):
        try:
            weather_df = pd.read_csv(weather_path)
        except Exception as e:
            print(f"! Warning: could not read weather ({e})")

    if load_df is not None:
        key = f"{target_state}/{target_site}"
        site_data[key] = {
            "site_id": target_site,
            "state": target_state,
            "load": load_df,
            "temp": temp_df,
            "weather": weather_df,
        }
        total_sites = 1
        print(f"✓ Loaded {target_state}/{target_site}")
    else:
        print(f"! No valid load data for {target_state}/{target_site}")
else:
    print(f"! Directory does not exist: {site_path}")

print(f"\nTotal sites loaded: {total_sites}")


######### HELPER FUNCTIONS AND ANALYSES #########

def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return np.nan

def current_season(date=None):
    if date is None:
        date = datetime.now()

    month = date.month
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Fall"
    



def return_days(df:pd.DataFrame):
    monday = df[df['dow']==0].groupby('hour')['total_kW'].mean()
    tuesday = df[df['dow']==1].groupby('hour')['total_kW'].mean()
    wednesday = df[df['dow']==2].groupby('hour')['total_kW'].mean()
    thursday = df[df['dow']==3].groupby('hour')['total_kW'].mean()
    friday = df[df['dow']==4].groupby('hour')['total_kW'].mean()
    saturday = df[df['dow']==5].groupby('hour')['total_kW'].mean()
    sunday = df[df['dow']==6].groupby('hour')['total_kW'].mean()

    return monday, tuesday, wednesday, thursday, friday, saturday, sunday



all_possible_appliances = [
    'Central AC',
    'Clothes Dryer', 
    'Clothes Washer',
    'Dishwasher',
    'Ducted Heatpump',
    'Ductless Heatpump',
    'Electric Baseboard Heaters',
    'Electric Furnace',
    'Electric Resistance Storage Water Heaters',
    'Electric Vehicle Charger',
    'Garbage Disposal',
    'Gas Furnace (Component)',
    'Heat Pump Water Heater',
    'Hot Tub',
    'Instantaneous Water Heater (Elec)',
    'Instantaneous Water Heater (Gas)',
    'Mains',
    'Mains With Solar',
    'Microwave',
    'Needs Review',
    'Other',
    'Other With Solar',
    'Other Zonal Heat',
    'Refrigerator/Freezer',
    'Room AC',
    'Solar',
    'Stove/Oven/Range'
    ]



load_df['date'] = pd.to_datetime(load_df['date'])
load_df['month'] = load_df['date'].dt.month
load_df['season'] = load_df['month'].apply(get_season)

season = current_season(date=None)

seasonal_df = load_df[load_df['season']==season]


seasonal_df = load_df[load_df['season'] == 'Summer']

output_dir = f"analyzed_sites/{target_site}"
os.makedirs(output_dir, exist_ok=True)




######### SITE -SPECIFIC ANALYSIS FUNCTIONS #########

def seasonal_changes(df: pd.DataFrame):   ### return avergage seasonal load
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month    
    df['season'] = df['month'].apply(get_season)
    average_seasonal = df.groupby('season')['total_kW'].mean()


    ### figure with daily profile for each season
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    fig = go.Figure()
    for season in seasons:
        seasonal_data = df[df['season'] == season]
        hourly_avg = seasonal_data.groupby('hour')['total_kW'].mean()
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg.values,
            mode='lines+markers',
            name=season,
            hovertemplate='<b>%{x}</b> hrs<br>%{y:.2f} kW<extra></extra>'
        ))
    fig.update_layout(
        title=dict(
            text="Average Daily Load Profile by Season",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Hour of Day", standoff=10),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Average Load (kW)", standoff=16),
            automargin=True,
        ),
        template="plotly_white",
        hovermode='x unified',
        height=520,
        margin=dict(l=90, r=40, t=90, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
    )
    # Export as HTML for Streamlit (interactive)
    fig.write_html(f"{output_dir}/Seasonal_load_profile_{target_site}.html")
    
    
    print(f"✓ Saved seasonal load plot (HTML + PNG) for {target_state}/{target_site}")
    
    return average_seasonal





def dow_with_most_consumption(df : pd.DataFrame):  ## return the average total consumption for each day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
    dow_consumption = df.groupby('dow')['total_kW'].mean().reset_index()
    dow_consumption['day'] = dow_consumption['dow'].apply(lambda x: days[x])
    dow_consumption['total_kWh'] = dow_consumption['total_kW'] * 24 # convert to kWh
    

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days,
        y=dow_consumption['total_kWh'],
        mode='lines+markers',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10, color='#e74c3c', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='<b>%{x}</b><br>%{y:.2f} kW<extra></extra>'
    ))
    fig.update_layout(
        title=dict(
            text="Weekly Energy Consumption Profile",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Day of Week", standoff=10),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Total Consumption (kWh)", standoff=16),
            automargin=True,
        ),
        template="plotly_white",
        hovermode='x unified',
        height=520,
        margin=dict(l=90, r=40, t=90, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
    )
    # Export as HTML for Streamlit (interactive)
    fig.write_html(f"{output_dir}/Week_profile_{target_site}.html")
    
    print(f"✓ Saved weekly profile (HTML + PNG) for site {target_state}/{target_site}")
        
    return dow_consumption[['total_kWh','day']].sort_values(by='total_kWh',ascending=False).reset_index(drop=True)





def day_profile(df:pd.DataFrame):  ## return th peak hour for each day
    monday, tuesday, wednesday, thursday, friday, saturday, sunday = return_days(df)
    
    monday_peak = monday.idxmax()
    tuesday_peak = tuesday.idxmax()
    wednesday_peak = wednesday.idxmax()
    thursday_peak = thursday.idxmax()
    friday_peak = friday.idxmax()
    saturday_peak = saturday.idxmax()
    sunday_peak = sunday.idxmax()

    # Calculate average load for each day
    monday_avg = monday.mean()
    tuesday_avg = tuesday.mean()
    wednesday_avg = wednesday.mean()
    thursday_avg = thursday.mean()
    friday_avg = friday.mean()
    saturday_avg = saturday.mean()
    sunday_avg = sunday.mean()
    
    peak_hour = pd.DataFrame(
        {
            'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'Peak Hour': [monday_peak, tuesday_peak, wednesday_peak, thursday_peak, friday_peak, saturday_peak, sunday_peak],
            'Peak Value (kW)': [monday.max(), tuesday.max(), wednesday.max(), thursday.max(), friday.max(), saturday.max(), sunday.max()],
            'Average Load (kW)': [monday_avg, tuesday_avg, wednesday_avg, thursday_avg, friday_avg, saturday_avg, sunday_avg]
        }
    )
    peak_day = peak_hour.loc[peak_hour['Peak Value (kW)'].idxmax()]

    fig = go.Figure()
    
    # Add peak values trace
    fig.add_trace(go.Scatter(
        x=peak_hour['Day'],
        y=peak_hour['Peak Value (kW)'],
        mode='lines+markers',
        name='Peak Load',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10, color='#e74c3c', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>Peak Hour: %{customdata[0]}<br>Peak: %{y:.2f} kW<extra></extra>',
        customdata=np.array(peak_hour['Peak Hour']).reshape(-1, 1)
    ))
    
    # Add average load trace
    fig.add_trace(go.Scatter(
        x=peak_hour['Day'],
        y=peak_hour['Average Load (kW)'],
        mode='lines+markers',
        name='Average Load',
        line=dict(color='#3498db', width=3, dash='dash'),
        marker=dict(size=8, color='#3498db', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>Average: %{y:.2f} kW<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Peak vs Average Load by Day of Week",
            x=0.5,
            xanchor="center",
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis=dict(
            title=dict(text="Day of Week", font=dict(size=14), standoff=10),
            tickfont=dict(size=12),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Load (kW)", font=dict(size=14), standoff=16),
            tickfont=dict(size=12),
            gridcolor='#ecf0f1',
            automargin=True,
        ),
        template="plotly_white",
        plot_bgcolor='#f8f9fa',
        hovermode='x unified',
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=90, r=40, t=90, b=70),
    )
    # Export as HTML for Streamlit (interactive)
    fig.write_html(f"{output_dir}/peak_hour_load_{target_site}.html")
    # Export as PNG for static use
    print(f"✓ Saved peak hour load plot (HTML) for site {target_state}/{target_site}")

    return peak_hour , peak_day

peak_hours, peak_day = day_profile(seasonal_df)



def appliance_correlation_at_peak_hours(df: pd.DataFrame, desired_day: str):
    """
    Function to create correlation matrices for appliances at peak hours for each day
    """
    # Get available appliances in the dataset
    appliances = [appliance for appliance in all_possible_appliances if appliance in df.columns]
    
    
    # Ensure we have timestamp and date columns
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['date'] = pd.to_datetime(df['date'])
    

    if desired_day.lower() == 'monday':
        df = df[df['dow'] == 0]
    elif desired_day.lower() == 'tuesday':
        df = df[df['dow'] == 1]
    elif desired_day.lower() == 'wednesday':    
        df = df[df['dow'] == 2]
    elif desired_day.lower() == 'thursday':
        df = df[df['dow'] == 3]
    elif desired_day.lower() == 'friday':
        df = df[df['dow'] == 4]
    elif desired_day.lower() == 'saturday':
        df = df[df['dow'] == 5]
    elif desired_day.lower() == 'sunday':
        df = df[df['dow'] == 6]
    

    day_peak = df.loc[df.groupby("date")["total_kW"].idxmax()][["date", "total_kW", "hour",'dow'] + [x for x in appliances]].reset_index(drop=True)

    corr_matrix = day_peak[appliances + ['total_kW']].corr()

    

    return corr_matrix['total_kW'].sort_values(ascending= False)

corr_matrix = appliance_correlation_at_peak_hours(seasonal_df,peak_day['Day'])



def weekend_vs_weekday_consumption(df : pd.DataFrame):  
    weekday = df[df['is_weekend']==0]
    weekend = df[df['is_weekend']==1]

    weekday = weekday.groupby('hour')['total_kW'].mean()
    weekend = weekend.groupby('hour')['total_kW'].mean()

    difference = sum(weekday-weekend)/24
    return difference

weekend_vs_weekday = weekend_vs_weekday_consumption(seasonal_df)



def day_vs_night_consumption(df: pd.DataFrame): #return day vs night consumption ratio
    day_time = df[(df['hour'] >= 7) & (df['hour'] < 19)]
    night_time = df[(df['hour'] < 7) | (df['hour'] >= 19)]
    day_avg = day_time['total_kW'].mean() * 12
    night_avg = night_time['total_kW'].mean() * 12
    
    ratio = day_avg/night_avg
    return ratio
   

day_vs_night_ratio = day_vs_night_consumption(seasonal_df)



def temp_hvac_regression(df: pd.DataFrame, weather_df: pd.DataFrame):
    
    weather_df['min_t'] = pd.to_datetime(weather_df['min_t'])
    weather_df['max_t'] = pd.to_datetime(weather_df['max_t'])

    # Option 1: Use start time's hour
    weather_df['hour'] = weather_df['min_t'].dt.hour


    # Option 2: Use the midpoint between start and end times
    weather_df['mid_t'] = weather_df['min_t'] + (weather_df['max_t'] - weather_df['min_t']) / 2
    weather_df['hour_mid'] = weather_df['mid_t'].dt.hour
    weather_df['date'] = weather_df['mid_t'].dt.date
        
    
    df['date'] = pd.to_datetime(df['date']).dt.date


    weather_df = weather_df.groupby('date')['temp'].mean().reset_index()


    df = df.groupby(['season','date'])['hvac_total_kW'].mean().reset_index()

    df_weather_merged = pd.merge(
        df,
        weather_df[['temp','date']],
        on='date',
        how='left'
    )
    

    summer = df_weather_merged[df_weather_merged['season'] =='Summer']  
    winter = df_weather_merged[df_weather_merged['season'] =='Winter']

    # Summer linear regression - ensure same shape by dropping NaN together
    summer_clean = summer[['temp', 'hvac_total_kW']].dropna()
    summer_clean = summer_clean[summer_clean['hvac_total_kW'] > 0]
    X_summer = summer_clean[['temp']]
    y_summer = summer_clean['hvac_total_kW']
    


    model = LinearRegression(fit_intercept=True)
    model.fit(X_summer, y_summer)
    y_pred = model.predict(X_summer)
    mse = mean_squared_error(y_summer, y_pred)
    r2 = r2_score(y_summer, y_pred)
    
    coefs_summer = model.coef_
    if model.fit_intercept:
        
        summer_intercept = model.intercept_
    else:
        summer_intercept = 0
        
    
    

    # Winter linear regression - ensure same shape by dropping NaN together
    winter_clean = winter[['temp', 'hvac_total_kW']].dropna()
    winter_clean = winter_clean[winter_clean['hvac_total_kW'] > 0]
    X_winter = winter_clean[['temp']]
    y_winter = winter_clean['hvac_total_kW']
    
    
    model_winter = LinearRegression(fit_intercept=True)
    model_winter.fit(X_winter, y_winter)
    y_pred_winter = model_winter.predict(X_winter)
    mse_winter = mean_squared_error(y_winter, y_pred_winter)
    r2_winter = r2_score(y_winter, y_pred_winter)


   
    coefs_winter = model_winter.coef_
    if model.fit_intercept:
        
        winter_intercept = model.intercept_

    else:
        winter_intercept = 0
    
    
    

    summary_df = pd.DataFrame({
        'Season': ['Summer', 'Winter'],
        'Coefficient': [coefs_summer[0], coefs_winter[0]],
        'Intercept': [summer_intercept, winter_intercept],
        'MSE': [mse, mse_winter],
        'R-squared': [r2, r2_winter]
    })

    return summary_df

regression_df = temp_hvac_regression(load_df,weather_df)




def dewpoint_vs_hvac(df:pd.DataFrame,weather:pd.DataFrame):
    df['date'] = pd.to_datetime(df['date']).dt.date

    weather['date'] = pd.to_datetime(weather['date']).dt.date

    df_weather_merged = pd.merge(
        df,
        weather[['dew_point', 'date', 'hour_mid']],
        left_on=['date', 'hour'],
        right_on=['date', 'hour_mid'],
        how='left'
    )

    dewpoint = df_weather_merged[['dew_point','hvac_total_kW']].dropna()
    dewpoint = dewpoint[dewpoint['hvac_total_kW'] > 0.4]
    X_dew = dewpoint[['dew_point']]
    y_dew = dewpoint['hvac_total_kW']
    
    model_dew = LinearRegression()
    model_dew.fit(X_dew, y_dew)
    y_pred_dew = model_dew.predict(X_dew)
    mse_dew = mean_squared_error(y_dew, y_pred_dew)
    r2_dew = r2_score(y_dew, y_pred_dew)
    

    summary_df = pd.DataFrame({
        'Parameter': ['Coefficient', 'Intercept', 'MSE', 'R-squared'],
        'Value': [model_dew.coef_[0], model_dew.intercept_, mse_dew, r2_dew]
    })
    return summary_df

regression_dew_point = dewpoint_vs_hvac(load_df,weather_df)







######## SITE COMPARISON ANALYSIS FUNCTIONS #########

def get_avg_load(df:pd.DataFrame) -> pd.Series:
    hourly_load = df.groupby(['date', 'hour'])['total_kW'].sum().mean()
    
    return float(hourly_load)

avg_load = get_avg_load(load_df)


def get_load_factor(df:pd.DataFrame, avg_daily_load:float) -> float:
    peak_load = df.groupby('date')['total_kW'].max()
    peak_load = peak_load.mean()

    load_factor = avg_daily_load / peak_load
    return float(load_factor)

load_factor = get_load_factor(load_df, avg_load)


def get_hvac_fraction(df:pd.DataFrame):
    hvac = df['hvac_total_kW'].sum()
    total = df['total_kW'].sum()
    fraction = hvac / total
    return float(fraction)

hvac_fraction = get_hvac_fraction(load_df)


def day_vs_night_ratio(df:pd.DataFrame) -> float:
    day_load = df[(df['hour'] >= 7) & (df['hour'] < 19)]['total_kW'].sum()
    night_load = df[(df['hour'] < 7) | (df['hour'] >= 19)]['total_kW'].sum()
    ratio = day_load / night_load
    return float(ratio)
day_night_ratio = day_vs_night_ratio(load_df)


def seasonal_std_variation(df:pd.DataFrame) -> float:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season)
    
    seasonal_mean = df.groupby('season')['total_kW'].mean()
    std_between_seasons = seasonal_mean.std()


    return std_between_seasons

seasonal_std = seasonal_std_variation(load_df)



def weekday_vs_weekend_consumption(df : pd.DataFrame):
    weekday = df[df['is_weekend']==0]
    weekend = df[df['is_weekend']==1]

    weekday = weekday.groupby('hour')['total_kW'].sum().mean()
    weekend = weekend.groupby('hour')['total_kW'].sum().mean()
    

    return float(weekday/weekend)

weekday_vs_weekend = weekday_vs_weekend_consumption(load_df)



keys = list(site_data.keys())

comparison_df = pd.DataFrame({
    'Site_id': keys,
    'avg_daily_load_kW': avg_load,
    'load_factor': load_factor,
    'hvac_fraction': hvac_fraction,
    'day_vs_night_ratio': day_night_ratio,
    'seasonal_std': seasonal_std,
    'weekend_vs_weekday_ratio': weekday_vs_weekend,
    'hvac/temperature regression summer coef': [regression_df['Coefficient'][0] for _ in keys],
    'hvac/temperature regression winter coef': [regression_df['Coefficient'][1] for _ in keys],
    'hvac regression summer R-squared': [regression_df['R-squared'][0] for _ in keys],
    'hvac regression winter R-squared': [regression_df['R-squared'][1] for _ in keys],
    'state': [site_data[k]['state'] for k in keys]   
})

def weight_block(X, w=0.4):
    return X * w


# Try to load clustering model - make it optional if it fails
try:
    pipe = joblib.load('kmeans_pipeline.joblib')
    
    # Predict cluster
    cluster_label = pipe.predict(comparison_df)[0]
    print("Cluster:", cluster_label)
    
    
    # Load the metadata and preprocessing pipeline components
    meta = joblib.load('kmeans_pipeline.meta.joblib')
    features_num = meta['features_num']
    feature_cat = meta['feature_cat']
    w_state = meta['w_state']
    
    # Get the trained pipeline components
    preprocess = pipe.named_steps['preprocess']
    kmeans_model = pipe.named_steps['model']
    
    # Get cluster centers and feature names
    centers = kmeans_model.cluster_centers_
    feature_names_out = []
    feature_names_out.extend(features_num)
    ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
    feature_names_out.extend(ohe.get_feature_names_out([feature_cat]).tolist())
    
    centers_df = pd.DataFrame(centers, columns=feature_names_out)
    centers_df['Cluster'] = range(len(centers_df))
    
    # Inverse-transform numeric features back to original units
    num_scaler = preprocess.named_transformers_['num'].named_steps['scaler']
    num_cols = features_num
    state_cols = [c for c in centers_df.columns if c.startswith(f'{feature_cat}_')]
    
    # Transform numeric centers back to original scale
    num_block_scaled = centers_df[num_cols].to_numpy()
    num_block_orig = num_scaler.inverse_transform(num_block_scaled)
    centers_num_inv = pd.DataFrame(num_block_orig, columns=[f"{c}_orig" for c in num_cols])
    
    # Unweight state block to get prevalence
    centers_state_weighted = centers_df[state_cols].copy()
    centers_state_prev = centers_state_weighted / w_state 
    centers_state_prev = centers_state_prev.clip(0, 1)
    
    # Create cluster summary
    cluster_summary = pd.concat([centers_df[['Cluster']], centers_num_inv, centers_state_prev], axis=1)
    
    print("=== CLUSTER CENTERS (Original Units) ===")
    print(cluster_summary.round(3))
    
    # Compare your site with its assigned cluster center
    print(f"\n=== SITE COMPARISON: Site {target_site} vs Cluster {cluster_label} ===")
    
    # Get the cluster center for your assigned cluster
    cluster_center = cluster_summary[cluster_summary['Cluster'] == cluster_label]
    
    # Create comparison table
    comparison_table = pd.DataFrame({
        'Feature': features_num,
        'Your_Site': [
            comparison_df[col].iloc[0] for col in features_num
        ],
        'Cluster_Center': [
            cluster_center[f"{col}_orig"].iloc[0] for col in features_num
        ]
    })
    
    comparison_table['Difference'] = comparison_table['Your_Site'] - comparison_table['Cluster_Center']
    comparison_table['Percent_Diff'] = (comparison_table['Difference'] / comparison_table['Cluster_Center'] * 100).round(1)
    
    print("\nFeature-by-Feature Comparison:")
    print(comparison_table.round(3))
    
    clustering_available = True
    
except Exception as e:
    print(f"\n! Warning: Could not load clustering model: {e}")
    print("! Continuing without clustering analysis...\n")
    clustering_available = False
    cluster_label = None
    cluster_summary = None
    comparison_table = None
    # Define default features for state comparison
    features_num = [
        'avg_daily_load_kW',
        'load_factor',
        'hvac_fraction',
        'day_vs_night_ratio',
        'seasonal_std',
        'weekend_vs_weekday_ratio',
        'hvac/temperature regression summer coef',
        'hvac/temperature regression winter coef',
        'hvac regression summer R-squared',
        'hvac regression winter R-squared'
    ]


def compare_site_with_state(site_data, state_averages_df):
    """
    Simple comparison function using the saved state averages.
    """
    site_state = site_data['state'].iloc[0]
    state_row = state_averages_df[state_averages_df['state'] == site_state]
    
    if state_row.empty:
        print(f"No data for state: {site_state}")
        return
    
    
    for feature in features_num:
        site_val = site_data[feature].iloc[0]
        state_avg = state_row[f'avg_{feature}'].iloc[0]
        diff = site_val - state_avg
        pct_diff = (diff / state_avg * 100)

    state_comparison_df = pd.DataFrame({
        'Feature': features_num,
        'Your_Site': [site_data[feature].iloc[0] for feature in features_num],
        'State_Average': [state_row[f'avg_{feature}'].iloc[0] for feature in features_num],
        'Difference': [site_data[feature].iloc[0] - state_row[f'avg_{feature}'].iloc[0] for feature in features_num],
        'Percent_Diff': [
            ((site_data[feature].iloc[0] - state_row[f'avg_{feature}'].iloc[0]) / state_row[f'avg_{feature}'].iloc[0] * 100).round(1)
            for feature in features_num
        ]
    })

    return state_comparison_df

state_averages = pd.read_csv('state_averages.csv')
state_comparison = compare_site_with_state(comparison_df, state_averages)

print(state_comparison.round(3))




######### CREATE JSON REPORT #########




def convert_to_json_serializable(obj):
    """Convert numpy types and handle NaN values for JSON serialization"""
    # Handle NaN, inf, -inf values first
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return None  # or "NaN" if you prefer a string representation
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        else:
            return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):  # Handle pandas NaN/NA values
        return None
    else:
        return obj

def save_site_analysis(
    site_id: str,
    state: str,
    kpis: dict,
    cluster_id: int,
    pipeline_info: dict,
    load: pd.DataFrame,
    weather: pd.DataFrame,
    seasonal: pd.DataFrame,
    season,
    file_path: str = "site_analysis.json",

): 
    """
    Creates and saves a JSON file summarizing site-specific analysis results.
    """

    analysis = {
        "schema_version": "1.0",
        "site": {
            "site_id": site_id,
            "state": state,
            "units": {
                "power": "kW",
                "temp": "°C"
            }
        },
        "site specific" : {
            "average seasonal load": seasonal_changes(load).to_dict(),
            "current season" : season,
            f"days with most consumption in {season} ranked(in kWh)": dow_with_most_consumption(seasonal).to_dict(orient='records'),
            f"{season} peak hours per day": day_profile(seasonal)[0].to_dict(orient='records'),
            "appliance correlation at peak hours": appliance_correlation_at_peak_hours(seasonal,day_profile(seasonal)[1]['Day']).to_dict(),
            f"weekend vs weekday consumption during {season}": weekend_vs_weekday_consumption(seasonal),
            f"day vs night consumption ratio during {season}": day_vs_night_consumption(seasonal),
            "temperature vs hvac regression": temp_hvac_regression(load,weather).to_dict(orient='records'),
            "dewpoint vs hvac regression": dewpoint_vs_hvac(load,weather).to_dict(orient='records'),
        },
        "site comparison" : {
            "site characteristics": {
                "avg_daily_load_kW": kpis['avg_daily_load_kW'],
                "load_factor": kpis['load_factor'],
                "hvac_fraction": kpis['hvac_fraction'],
                "day_vs_night_ratio": kpis['day_vs_night_ratio'],
                "seasonal_std": kpis['seasonal_std'],
                "weekend_vs_weekday_ratio": kpis['weekend_vs_weekday_ratio'],
                "hvac/temperature regression summer coef": kpis['hvac/temperature regression summer coef'],
                "hvac/temperature regression winter coef": kpis['hvac/temperature regression winter coef'],
                "hvac regression summer R-squared": kpis['hvac regression summer R-squared'],
                "hvac regression winter R-squared": kpis['hvac regression winter R-squared']
            },
            "cluster_comparison": comparison_table.round(3).to_dict(orient='records'),
            "assigned cluster": cluster_id,
            "cluster centers": cluster_summary.round(3).to_dict(orient='records'),
            "state comparison": state_comparison.round(3).to_dict(orient='records')
        },
        "pipeline": pipeline_info,
    }

    # Convert all numpy types to JSON serializable types
    analysis = convert_to_json_serializable(analysis)
    
    # Save as pretty-printed JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"Site analysis JSON saved to: {file_path}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Get site information
    site_keys = list(site_data.keys())
    site_id = site_data[site_keys[0]]['site_id']
    state = site_data[site_keys[0]]['state']
    
    # KPIs from comparison dataframe
    kpis = {
        "avg_daily_load_kW": comparison_df['avg_daily_load_kW'].iloc[0],
        "load_factor": comparison_df['load_factor'].iloc[0],
        "hvac_fraction": comparison_df['hvac_fraction'].iloc[0],
        "day_vs_night_ratio": comparison_df['day_vs_night_ratio'].iloc[0],
        "seasonal_std": comparison_df['seasonal_std'].iloc[0],
        "weekend_vs_weekday_ratio": comparison_df['weekend_vs_weekday_ratio'].iloc[0],
        "hvac/temperature regression summer coef": comparison_df['hvac/temperature regression summer coef'].iloc[0],
        "hvac/temperature regression winter coef": comparison_df['hvac/temperature regression winter coef'].iloc[0],
        "hvac regression summer R-squared": comparison_df['hvac regression summer R-squared'].iloc[0],
        "hvac regression winter R-squared": comparison_df['hvac regression winter R-squared'].iloc[0],   
    }
    
    # Only create pipeline info and save analysis if clustering is available
    if clustering_available:
        cluster_id = cluster_label
        
        # Pipeline info using actual metadata
        pipeline_info = {
            "model_file": "kmeans_pipeline.joblib",
            "k": len(cluster_summary),  # Actual number of clusters
            "features_used": features_num + [feature_cat],  # From loaded metadata
            "w_state": w_state,  # From loaded metadata
            "model_type": "KMeans",
            "preprocessing": "StandardScaler + OneHotEncoder"
        }

        save_site_analysis(
            site_id=site_id,
            state=state,
            kpis=kpis,
            cluster_id=cluster_id,
            load=load_df,
            weather=weather_df,
            seasonal=seasonal_df,
            pipeline_info=pipeline_info,
            season=season,
            file_path=f"{output_dir}/site_analysis_{site_id.replace('/', '_')}.json"
        )
    else:
        print("\n! Skipping JSON export - clustering data not available")
        print(f"! Site {site_id} analysis complete, but full results require clustering model")


