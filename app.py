import datetime

import streamlit as st
import io
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.chart_container import chart_container
# Custom switch_page function to replace deprecated streamlit_extras
def switch_page(page_name: str):
    """Custom function to switch pages in Streamlit"""
    try:
        # Try using st.switch_page directly
        st.switch_page(f"pages/{page_name}.py")
    except Exception:
        # Fallback: show message and store video_id in session state
        st.session_state['video_id'] = st.session_state.get('video_id')
        st.success(f"Video selected! Use the navigation to view detailed analytics.")
        st.info("💡 Tip: Use the sidebar navigation to access detailed video analytics.")
from streamlit_extras.app_logo import add_logo

from prophet import Prophet

from channelDataExtraction import getChannelData
from channelVideoDataExtraction import *


########################################################################################################################
#                                               FUNCTIONS
########################################################################################################################
@st.cache_data
def download_data(api_key, channel_id):
    channel_details = getChannelData(api_key, channel_id)

    # check if bad channel id
    if channel_details is None:
        return None, None, None, None

    videos = getVideoList(api_key, channel_details["uploads"])
    videos_df = pd.DataFrame(videos)
    video_ids = [video['id'] for video in videos if video['id'] is not None]
    all_video_data = buildVideoListDataframe(api_key, video_ids)

    st.session_state.start_index = 0
    st.session_state.end_index = 10
    st.session_state['video_id'] = None
    st.session_state.all_video_df = all_video_data

    st.session_state.api_key = st.session_state.API_KEY

    return channel_details, videos, all_video_data, videos_df


def display_video_list(video_data, start_index, end_index, search_query=None):
    """Displays a list of videos in a tabular format with custom column order and buttons."""

    # Input widget for searching videos by title
    if search_query is None:
        search_query = ""
    new_search_query = st.text_input("Search Videos by Title", search_query)

    # Initialize start_index and end_index in session_state
    if 'start_index' not in st.session_state:
        st.session_state.start_index = start_index
    if 'end_index' not in st.session_state:
        st.session_state.end_index = end_index

    # If a new search query is entered, reset the start and end indices
    if new_search_query != search_query:
        st.session_state.start_index = start_index
        st.session_state.end_index = end_index

    # Filter videos based on the search query across the entire video_data list
    filtered_videos = [video for video in video_data if new_search_query.lower() in video['title'].lower()]

    # Paginate the filtered results
    paginated_videos = filtered_videos[st.session_state.start_index:st.session_state.end_index]

    for video in paginated_videos:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(video['thumbnail'])
        with col2:
            st.write(video['id'])
        with col3:
            st.write(video['title'])
        with col4:
            video_stats = st.button("Check Video Statistics", key=video['id'])
            if video_stats:
                st.session_state['video_id'] = video['id']
                switch_page("video_data")

    # Display a button to load the next 10 search results
    if st.session_state.end_index < len(filtered_videos):
        if st.button('Load next 10 videos', key='load_next'):
            st.session_state.start_index = st.session_state.end_index
            st.session_state.end_index += 10


########################################################################################################################
#                                       MAIN PAGE CONFIGURATION
########################################################################################################################
st.set_page_config(page_title="YouTube Analytics Dashboard",
                   page_icon="📊",
                   layout="wide")

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern Dashboard Styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chart-container {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,107,53,0.4);
    }
    
    .stExpander {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

########################################################################################################################
#                                       SIDE BAR CONFIGURATION
########################################################################################################################
# Modern header
st.markdown('<div class="main-header">🎯 YouTube Analytics Dashboard</div>', unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)

# Sidebar: Enter Channel ID and YouTube API Key
if 'API_KEY' not in st.session_state:
    st.session_state.API_KEY = ""
if 'CHANNEL_ID' not in st.session_state:
    st.session_state.CHANNEL_ID = ""

st.session_state.API_KEY = st.sidebar.text_input("Enter your YouTube API Key", st.session_state.API_KEY,
                                                 type="password")
st.session_state.CHANNEL_ID = st.sidebar.text_input("Enter the YouTube Channel ID", st.session_state.CHANNEL_ID)

if not st.session_state.API_KEY or not st.session_state.CHANNEL_ID:
    st.warning("Please enter your API Key and Channel ID.")
    # Display the GitHub link for the user manual
    user_manual_link = "https://github.com/zainmz/Youtube-Channel-Analytics-Dashboard"
    st.markdown(f"If you need help, please refer to the the GitHub Repository for the [User Manual]({user_manual_link}).")
    st.stop()

# Data Refresh Button
refresh_button = st.sidebar.button("Refresh Data")

# First Data Load
channel_details, videos, all_video_data, videos_df = download_data(st.session_state.API_KEY, st.session_state.CHANNEL_ID)

if channel_details is None:
    st.warning("Invalid YouTube Channel ID. Please check and enter a valid Channel ID.")
    st.stop()

# Data Filters for fine-tuned data selection
with st.sidebar:
    st.markdown('<div class="sidebar-section">🔍 Data Filters</div>', unsafe_allow_html=True)
    
    num_videos = st.slider("📊 Number of Top Videos:", 1, 50, 10, help="Select how many top videos to display in charts")

    # Convert the 'published_date' column to datetime format
    all_video_data['published_date'] = pd.to_datetime(all_video_data['published_date'])

    # Extract min and max publish dates
    min_date = all_video_data['published_date'].min().date()  # Ensure it's a date object
    max_date = all_video_data['published_date'].max().date()  # Ensure it's a date object

    # Sidebar date input
    st.markdown("**📅 Date Range:**")
    start_date = st.date_input("From:", min_date, help="Select start date for analysis")
    end_date = st.date_input("To:", max_date, help="Select end date for analysis")
    
    if start_date > end_date:
        st.error("⚠️ Start date should be earlier than end date.")
        st.stop()
    
    st.markdown("**🏷️ Tag Search:**")
    tag_search = st.text_input("Search by tags:", placeholder="Enter tag to filter videos", help="Filter videos by specific tags")

if refresh_button:
    with st.spinner("Refreshing data..."):
        channel_details, videos, all_video_data, videos_df = download_data(st.session_state.API_KEY, st.session_state.CHANNEL_ID)

        if channel_details is None:
            st.warning("Invalid YouTube Channel ID. Please check and enter a valid Channel ID.")
            st.stop()

# Filter data based on date range and tags
date_range_start = pd.Timestamp(start_date)
date_range_end = pd.Timestamp(end_date)

filtered_data = all_video_data[(all_video_data['published_date'] >= date_range_start) &
                               (all_video_data['published_date'] <= date_range_end)]

if tag_search:
    # Handle tags properly - convert to string if it's a list
    filtered_data = filtered_data[filtered_data['tags'].apply(lambda x: tag_search in str(x) if isinstance(x, list) else tag_search in str(x))]

########################################################################################################################
#                                       CHANNEL DETAILS AREA CONFIGURATION
########################################################################################################################

# Display channel details with modern styling
st.markdown('<div class="section-header">📺 Channel Overview</div>', unsafe_allow_html=True)

# Channel info in a modern card layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Channel thumbnail and basic info
    channel_thumbnail = channel_details['thumbnail']
    add_logo(channel_thumbnail, height=200)
    
    st.markdown(f"### {channel_details['title']}")
    
    # Channel Description with dropdown visibility
    with st.expander("📝 Channel Description", expanded=False):
        st.markdown(f"{channel_details['description']}")

with col2:
    # Go to Channel Button
    st.link_button("🔗 Visit Channel", f"https://www.youtube.com/channel/{st.session_state.CHANNEL_ID}")

with col3:
    # Quick stats
    view_count = int(channel_details['viewCount'])
    subscriber_count = int(channel_details['subscriberCount'])
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total Videos</h3>
        <h2>{len(videos):,}</h2>
    </div>
    """, unsafe_allow_html=True)

# Key metrics in a modern grid
st.markdown('<div class="section-header">📈 Key Performance Metrics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>👁️ Total Views</h3>
        <h2>{view_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>👥 Subscribers</h3>
        <h2>{subscriber_count:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_views = view_count // len(videos) if len(videos) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Avg Views/Video</h3>
        <h2>{avg_views:,}</h2>
    </div>
    """, unsafe_allow_html=True)

########################################################################################################################
#                                            TOP  VIDEO GRAPHS AREA
########################################################################################################################

st.markdown('<div class="section-header">🎬 Top Performing Videos</div>', unsafe_allow_html=True)

# Create tabs for different metrics
tab1, tab2, tab3 = st.tabs(["👁️ Views", "👍 Likes", "💬 Comments"])

with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    sorted_video_data = filtered_data.sort_values(by='view_count', ascending=False)
    top_views_df = sorted_video_data.head(num_videos)
    
    # Create enhanced bar chart
    fig = px.bar(top_views_df, x='view_count', y='title', orientation='h',
                 title=f"Top {num_videos} Videos by Views",
                 color='view_count',
                 color_continuous_scale='viridis')
    fig.update_layout(
        xaxis_title="View Count",
        yaxis_title="Video Title",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    sorted_video_data = filtered_data.sort_values(by='like_count', ascending=False)
    top_likes_df = sorted_video_data.head(num_videos)
    
    fig = px.bar(top_likes_df, x='like_count', y='title', orientation='h',
                 title=f"Top {num_videos} Videos by Likes",
                 color='like_count',
                 color_continuous_scale='plasma')
    fig.update_layout(
        xaxis_title="Like Count",
        yaxis_title="Video Title",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    sorted_video_data = filtered_data.sort_values(by='comment_count', ascending=False)
    top_comments_df = sorted_video_data.head(num_videos)
    
    fig = px.bar(top_comments_df, x='comment_count', y='title', orientation='h',
                 title=f"Top {num_videos} Videos by Comments",
                 color='comment_count',
                 color_continuous_scale='inferno')
    fig.update_layout(
        xaxis_title="Comment Count",
        yaxis_title="Video Title",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

########################################################################################################################
#                                            CHANNEL GROWTH STATS
########################################################################################################################

st.markdown('<div class="section-header">📈 Channel Growth Analytics</div>', unsafe_allow_html=True)

# Prepare data for pynarrative
growth_data = filtered_data[['published_date', 'view_count']].copy()
growth_data['published_date'] = pd.to_datetime(growth_data['published_date'])
growth_data = growth_data.sort_values('published_date')

# Calculate growth insights
total_views = growth_data['view_count'].sum()
avg_views = growth_data['view_count'].mean()
max_views = growth_data['view_count'].max()
min_views = growth_data['view_count'].min()
growth_rate = ((growth_data['view_count'].iloc[-1] - growth_data['view_count'].iloc[0]) / growth_data['view_count'].iloc[0]) * 100 if len(growth_data) > 1 else 0

# Find peak performance date
peak_date = growth_data.loc[growth_data['view_count'].idxmax(), 'published_date']
peak_views = growth_data['view_count'].max()

# Create enhanced Plotly chart with narrative elements
try:
    # Create a more engaging Plotly chart with narrative elements
    fig = go.Figure()
    
    # Add the main line trace
    fig.add_trace(
        go.Scatter(
            x=growth_data['published_date'], 
            y=growth_data['view_count'], 
            mode='lines+markers', 
            name='Views Over Time', 
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=8, color='#FF6B35')
        )
    )
    
    # Add peak point annotation
    fig.add_annotation(
        x=peak_date,
        y=peak_views,
        text=f"Peak: {peak_views:,} views",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#e74c3c',
        ax=0,
        ay=-40,
        bgcolor='#e74c3c',
        bordercolor='#e74c3c',
        borderwidth=2,
        font=dict(color='white', size=12)
    )
    
    # Update layout with narrative elements
    fig.update_layout(
        title={
            'text': f"Channel Viewership Journey<br><sub>Total: {total_views:,} views | Avg: {avg_views:,.0f} per video</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1a1a1a'}
        },
        xaxis_title="Published Date",
        yaxis_title="Number of Views",
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        annotations=[
            dict(
                text=f"📈 {growth_rate:.1f}% growth trend",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                xanchor='center', yanchor='bottom',
                font=dict(size=14, color='#2ecc71')
            ),
            dict(
                text="💡 Click on points for video details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                font=dict(size=12, color='#3498db')
            )
        ]
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
except Exception as e:
    # Fallback to simple plotly chart if enhanced version fails
    st.error(f"⚠️ Chart error: {str(e)}")
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=growth_data['published_date'], y=growth_data['view_count'], 
                  mode='lines+markers', name='Views Over Time', line=dict(color='orange'))
    )
    fig.update_layout(title='Views Over Time',
                      xaxis_title='Published Date',
                      yaxis_title='Number of Views',
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-header">🔮 Growth Predictions</div>', unsafe_allow_html=True)

with st.spinner("Predicting Views for the next Week"):
    # Prepare dataframe for Prophet
    forecast_df = all_video_data[['published_date', 'view_count']]
    forecast_df.columns = ['ds', 'y']

    # Initialize the Prophet model
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive')

    # Fit the model with the data
    model.fit(forecast_df)

    # Dataframe for future dates
    future_dates = model.make_future_dataframe(periods=30)

    # Predict views for the future dates
    forecast = model.predict(future_dates)
    # Plot the original data and the forecast

    # Plotting using Plotly
    # Filter the forecast dataframe to include only the forecasted period
    forecasted_period = forecast[forecast['ds'] > forecast_df['ds'].max()]

    # Plotting using Plotly
    # Filter the forecast dataframe to include only the forecasted period
    forecasted_period = forecast[forecast['ds'] > forecast_df['ds'].max()]

    # Filter the original dataframe to include only the last 30 days
    last_date = forecast_df['ds'].max()
    start_date = last_date - datetime.timedelta(days=30)
    last_30_days = forecast_df[(forecast_df['ds'] > start_date) & (forecast_df['ds'] <= last_date)]

    # Plotting using Plotly
    trace1 = go.Scatter(x=last_30_days['ds'], y=last_30_days['y'], mode='lines', name='Actual Views (Last 30 Days)')
    trace2 = go.Scatter(x=forecasted_period['ds'], y=forecasted_period['yhat'], mode='lines',
                        name='Predicted Views (Next 30 Days)')
    layout = go.Layout(title="YouTube Views: Last 30 Days and Forecast for Next 30 Days", xaxis_title="Date",
                       yaxis_title="Views")
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Display the combined historical and forecast data in Streamlit using Plotly
    st.plotly_chart(fig, use_container_width=True)
########################################################################################################################
#                                         WORD CLOUD & LIKE TO VIEW RATIO
########################################################################################################################

st.markdown('<div class="section-header">📊 Content Analysis</div>', unsafe_allow_html=True)

# Create tabs for content analysis
tab1, tab2 = st.tabs(["🏷️ Tag Analysis", "📈 Engagement Metrics"])

with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    with st.spinner("Generating Word Cloud..."):
        st.markdown("### Most Common Tags")
        # Extracting tags from DataFrame and creating a single string
        all_tags = " ".join(" ".join(tags) for tags in filtered_data['tags'])

        # Generating the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_tags)

        # Plotting the word cloud using matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        # Saving the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        st.image(buf, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    # Calculating the Like-to-View Ratio - create a copy to avoid SettingWithCopyWarning
    filtered_data_copy = filtered_data.copy()
    filtered_data_copy['like_to_view_ratio'] = filtered_data_copy['like_count'] / filtered_data_copy['view_count']

    # Extracting the like-to-view ratio and published dates from the dataframe
    like_to_view_ratio = filtered_data_copy['like_to_view_ratio']
    dates = filtered_data_copy['published_date']

    st.markdown("### Like-to-View Ratio Over Time")

    # Creating a time series plot for Like-to-View Ratio using Plotly
    fig_ratio = go.Figure()

    fig_ratio.add_trace(go.Scatter(x=dates, y=like_to_view_ratio, mode='lines+markers', name='Like-to-View Ratio',
                                   line=dict(color='#00ff88', width=3),
                                   marker=dict(size=8, color='#00ff88')))

    fig_ratio.update_layout(
        title="Engagement Rate Trends",
        xaxis_title='Published Date',
        yaxis_title='Like-to-View Ratio',
        template="plotly_dark",
        height=400
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_ratio, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

########################################################################################################################
#                                         DETAILED VIDEO STATS SELECTION SECTION
########################################################################################################################

st.markdown('<div class="section-header">🎥 Video Library</div>', unsafe_allow_html=True)

# Video selection with modern styling
st.markdown("""
<div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
    <h4>🔍 Browse Your Videos</h4>
    <p>Click on "Check Video Statistics" to get detailed analytics for any video.</p>
</div>
""", unsafe_allow_html=True)

# latest 10 videos
display_video_list(videos, 0, 10)
