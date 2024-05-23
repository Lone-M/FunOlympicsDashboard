import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import requests

def fetch_data_from_api(API_KEY, API_ENDPOINT):
    # Assuming API returns data in JSON format
    response = requests.get(API_ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}"})
    if response.status_code == 200:
        data = response.json()
        # Assuming data is a list of lists, where each inner list represents a row of data
        columns = ['timestamp', 'country', 'views', 'sports', 'devices', 'browsers', 'ip_addresses']
        df = pd.DataFrame(data, columns=columns)
        return df
    else:
        print("Failed to fetch data from API.")
        return None

def get_data(API_KEY=None, API_ENDPOINT=None, num_rows=3500, num_days=4):
    if API_KEY is None or API_ENDPOINT is None:
        # Generate test data
        countries = ['USA', 'Canada', 'Japan', 'UK', 'France', 'Germany', 'Russia', 'Botswana']
        sports = ['High Jump', 'Shotput', 'Gymnastics', 'Basketball', 'Football', 'Netball']
        devices = ['Desktop', 'Mobile', 'Tablet']
        browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
        ip_addresses = ['190.1.13.41', '102.12.29.5', '119.3.34.1', '5.8.6.7']

        start_date = datetime.now()
        data = []

        for i in range(num_rows):
            timestamp = start_date + timedelta(days=random.randint(0, num_days-1), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            country = random.choice(countries)
            views = random.randint(1, 100)
            sport = random.choice(sports)
            device = random.choice(devices)
            browser = random.choice(browsers)
            ip_address = random.choice(ip_addresses)
            data.append([timestamp, country, views, sport, device, browser, ip_address])

        # Convert test data to DataFrame
        columns = ['timestamp', 'country', 'views', 'sports', 'devices', 'browsers', 'ip_addresses']
        df = pd.DataFrame(data, columns=columns)

        return df
    else:
        # Fetch data from the API using provided credentials
        return fetch_data_from_api(API_KEY, API_ENDPOINT)

df = pd.read_csv("FunOlympicSheet.csv")

# Initialize the Streamlit app
st.set_page_config(page_title="FunOlympics_Dashboard", page_icon=":bar_chart:", layout="wide")
st.title("FunOlympics Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
#sidebar
st.sidebar.header("Choose your filter: ")
# Create for Country
Country = st.sidebar.multiselect("Pick your Country", df["country"].unique())
if not Country:
    df2 = df.copy()
else:
    df2 = df[df["country"].isin(Country)]

# Create for Sport
Sport = st.sidebar.multiselect("Pick the Sport", df2["sports"].unique())
if not Sport:
    df3 = df2.copy()
else:
    df3 = df2[df2["sports"].isin(Sport)]

# Create for Device
Device = st.sidebar.multiselect("Pick the Device",df3["devices"].unique())

def filter_data(df, Country=None, Sport=None, Device=None):
    if not Country and not Sport and not Device:
        return df
    elif not Sport and not Device:
        return df[df["country"].isin(Country)]
    elif not Country and not Device:
        return df[df["sports"].isin(Sport)]
    elif Sport and Device:
        return df[df["sports"].isin(Sport) & df["devices"].isin(Device)]
    elif Country and Device:
        return df[df["country"].isin(Country) & df["devices"].isin(Device)]
    elif Country and Sport:
        return df[df["country"].isin(Country) & df["sports"].isin(Sport)]
    elif Device:
        return df[df["devices"].isin(Device)]
    else:
        return df[df["country"].isin(Country) & df["sports"].isin(Sport) & df["devices"].isin(Device)]


# Function to create Views Per Sport chart
def create_views_per_sport_chart(filtered_df):
    category_df = filtered_df.groupby(by=["sports"], as_index=False)["views"].sum()
    fig = px.bar(category_df, x="sports", y="views", text=['{:,d}'.format(x) for x in category_df["views"]],
                 template="seaborn") 
    fig.update_layout(title_text="views per sport")
    return fig

# Function to create Views Per Country chart
def create_views_per_country_chart(filtered_df):
    fig = px.pie(filtered_df, values="views", names="country", hole=0.5)
    fig.update_traces(text=filtered_df["country"], textposition="outside")
    fig.update_layout(title_text="views per country")
    return fig

# Function to create Time Series Analysis chart
def create_time_series_chart(filtered_df):
    filtered_df["date"] = filtered_df["timestamp"].dt.date
    linechart = pd.DataFrame(filtered_df.groupby(filtered_df["date"])["views"].sum()).reset_index()
    fig = px.line(linechart, x="date", y="views", labels={"views": "Views"}, height=500, width=1000, template="gridon")
    fig.update_layout(title_text="Time Series Analysis")
    return fig

# Function to create Views By Browsers chart
def create_views_by_browsers_chart(filtered_df):
    fig = px.pie(filtered_df, values="views", names="browsers", template="plotly_dark")
    fig.update_traces(text=filtered_df["browsers"], textposition="inside")
    fig.update_layout(title_text="views by browsers")
    return fig

# Function to create Views By Streaming Device chart
def create_views_by_device_chart(filtered_df):
    fig = px.pie(filtered_df, values="views", names="devices", template="gridon")
    fig.update_traces(text=filtered_df["devices"], textposition="inside")
    fig.update_layout(title_text="views by Streaming Device")
    return fig

# Function to create Olympic Games Streaming Summary table
def create_summary_table(df):
    df_sample = df[0:15][['timestamp', 'country', 'views', 'sports', 'devices', 'browsers']]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    fig.update_layout(title_text="Summary Table")
    return fig

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
placeholder1 = col1.empty()
placeholder2 = col2.empty()
placeholder3 = st.empty()
placeholder4 = col3.empty()
placeholder5 = col4.empty()
placeholder6 = st.empty()

# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download CSV Dataset', data = csv, file_name = "FunOlympicSheet.csv",mime = "text/csv")

# Main loop
while True:

    df = pd.read_csv("FunOlympicSheet.csv")
    df = filter_data(df, Country=None, Sport=None, Device=None)
    # Filter data based on date range
    filtered_df = df.copy()

    # Create charts
    views_per_sport_chart = create_views_per_sport_chart(filtered_df)
    views_per_country_chart = create_views_per_country_chart(filtered_df)
    time_series_chart = create_time_series_chart(filtered_df)
    views_by_browsers_chart = create_views_by_browsers_chart(filtered_df)
    views_by_device_chart = create_views_by_device_chart(filtered_df)
    summary_table = create_summary_table(df)

    # Display charts
    placeholder1.plotly_chart(views_per_sport_chart, use_container_width=True, height=200)

    placeholder2.plotly_chart(views_per_country_chart, use_container_width=True)

    placeholder3.plotly_chart(time_series_chart, use_container_width=True)

    placeholder4.plotly_chart(views_by_browsers_chart, use_container_width=True)

    placeholder5.plotly_chart(views_by_device_chart, use_container_width=True)

    placeholder6.plotly_chart(summary_table, use_container_width=True)

    # Delay for 3 seconds before updating again
    time.sleep(3)
