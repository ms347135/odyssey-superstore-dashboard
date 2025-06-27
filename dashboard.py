import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# 1. Add error handling for Prophet import
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Odyssey | Strategic Command Center",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Global styling */
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #0c1228;
        color: #e0e0e0;
    }

    /* Main Title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121a3a;
        border-right: 1px solid #2a3a6b;
    }

    /* KPI Cards */
    .kpi-card {
        background: #121a3a;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2a3a6b;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.1);
        border-color: #00d4ff;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-label {
        font-size: 1rem;
        color: #a0a0a0;
    }

    /* AI Summary Box */
    .ai-summary {
        background: #121a3a;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #2a3a6b;
        margin-bottom: 2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        background-color: transparent;
        border-radius: 8px;
        border: 1px solid #2a3a6b;
	}
	.stTabs [aria-selected="true"] {
  		background-color: #00d4ff;
        color: #0c1228;
        font-weight: bold;
	}
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    # Calculate costs for What-If analysis
    df['cost'] = df['sales'] - df['profit']
    df['profit_margin'] = np.where(df['sales'] != 0, (df['profit'] / df['sales']) * 100, 0)
    df['order_year'] = df['order_date'].dt.year
    return df


df = load_data('Global_Superstore2.csv')

# --- Sidebar Filters ---
with st.sidebar:
    st.title("🔮 ODYSSEY")
    st.header("Control Center")

    selected_market = st.selectbox('Select Market', ['All Markets'] + sorted(df['market'].unique().tolist()))
    selected_category = st.selectbox('Select Product Category',
                                     ['All Categories'] + sorted(df['category'].unique().tolist()))

    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# --- Filtered DataFrame Logic ---
# 2. Improve the date range validation
if len(date_range) == 2:
    start_date, end_date = date_range
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
        st.stop()

    df_filtered = df[
        (df['order_date'] >= pd.to_datetime(start_date)) &
        (df['order_date'] <= pd.to_datetime(end_date))
        ]
    if selected_market != 'All Markets':
        df_filtered = df_filtered[df_filtered['market'] == selected_market]
    if selected_category != 'All Categories':
        df_filtered = df_filtered[df_filtered['category'] == selected_category]

    # Check if filtered data is empty
    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.stop()
else:
    st.error("Error: Please select a valid date range.")
    st.stop()


# --- AI Executive Summary ---
def generate_summary(df_in):
    if df_in.empty:
        return "No data for the selected filters. Please expand your selection."

    total_sales = df_in['sales'].sum()
    total_profit = df_in['profit'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    top_country = df_in.groupby('country')['sales'].sum().idxmax()

    summary = (
        f"Based on the current selection, Total Sales stand at ${total_sales:,.0f} with a "
        f"Net Profit of ${total_profit:,.0f}, yielding a {profit_margin:.1f}% Profit Margin. "
        f"The primary revenue driver is {top_country}."
    )
    return summary

# --- Main Dashboard ---
st.markdown('<h1 class="main-title">Odyssey | Strategic Command Center</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Navigating Business Intelligence with AI-Powered Analytics</p>',
            unsafe_allow_html=True)

# --- AI Summary & KPIs ---
st.markdown('<div class="ai-summary">', unsafe_allow_html=True)
st.subheader("🧠 AI Executive Summary")
st.markdown(generate_summary(df_filtered), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
kpi_cols = st.columns(4)
kpi_data = {
    "Total Sales": f"${df_filtered['sales'].sum():,.0f}",
    "Net Profit": f"${df_filtered['profit'].sum():,.0f}",
    "Profit Margin": f"{df_filtered['profit_margin'].mean():.1f}%",
    "Avg. Order Value": f"${df_filtered.groupby('order_id')['sales'].sum().mean():,.0f}"
}
for label, value in kpi_data.items():
    with kpi_cols[list(kpi_data.keys()).index(label)]:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Strategic Overview",
    "🔬 Product Intelligence",
    "🌍 Geographic Command",
    "🔮 AI Sales Forecast",
    "💡 What-If Scenarios"
])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Revenue Flow Analysis")
        # Sankey Diagram
        if not df_filtered.empty:
            sankey_data = df_filtered.groupby(['market', 'category'])['sales'].sum().reset_index()
            if not sankey_data.empty:
                all_nodes = list(pd.concat([sankey_data['market'], sankey_data['category']]).unique())
                node_map = {node: i for i, node in enumerate(all_nodes)}

                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes),
                    link=dict(
                        source=[node_map[x] for x in sankey_data['market']],
                        target=[node_map[x] for x in sankey_data['category']],
                        value=sankey_data['sales']
                    )
                )])
                fig_sankey.update_layout(title_text="Revenue Flow: Market to Category", font_size=10,
                                         paper_bgcolor='#121a3a', plot_bgcolor='#121a3a', font_color='#e0e0e0')
                st.plotly_chart(fig_sankey, use_container_width=True)

    with col2:
        st.subheader("Performance by Segment")
        if not df_filtered.empty:
            segment_perf = df_filtered.groupby('segment')['sales'].sum().reset_index()
            fig_pie = px.pie(segment_perf, values='sales', names='segment', hole=0.6,
                             color_discrete_sequence=['#00d4ff', '#764ba2', '#f093fb'])
            fig_pie.update_layout(paper_bgcolor='#121a3a', plot_bgcolor='#121a3a', font_color='#e0e0e0',
                                  showlegend=False)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("3D Product Portfolio Analysis")
    if not df_filtered.empty:
        scatter_data = df_filtered.groupby('sub_category').agg({
            'sales': 'sum',
            'profit': 'sum',
            'discount': 'mean'
        }).reset_index()

        fig_3d = px.scatter_3d(
            scatter_data,
            x='sales', y='profit', z='discount',
            color='profit',
            size='sales',
            hover_name='sub_category',
            title="Sales vs Profit vs Discount by Sub-Category"
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Total Sales',
                yaxis_title='Total Profit',
                zaxis_title='Avg. Discount',
                bgcolor='#0c1228'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='#121a3a',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("Global Sales Heatmap")
    if not df_filtered.empty:
        country_data = df_filtered.groupby('country').agg(
            total_sales=('sales', 'sum'),
            total_profit=('profit', 'sum')
        ).reset_index()
        fig_map = px.choropleth(
            country_data,
            locations="country",
            locationmode='country names',
            color="total_sales",
            hover_name="country",
            hover_data=['total_profit'],
            color_continuous_scale='Cividis',
            projection="natural earth"
        )
        fig_map.update_layout(paper_bgcolor='#121a3a', font_color='#e0e0e0', geo=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_map, use_container_width=True)

# 3. Enhance the forecast tab with better error handling
with tab4:
    st.subheader("AI-Powered Sales Forecast")
    if not PROPHET_AVAILABLE:
        st.error("Prophet library not available. Forecasting features are disabled. Install with: pip install prophet")
    else:
        # Prepare data for Prophet
        df_ts = df_filtered[['order_date', 'sales']].rename(columns={'order_date': 'ds', 'sales': 'y'})
        df_ts = df_ts.groupby('ds').sum().reset_index()

        if len(df_ts) < 10:
            st.warning("Need at least 10 data points for forecasting. Please select a larger date range.")
        elif len(df_ts) < 30:
            st.info("Limited data available. Forecast may be less accurate with fewer than 30 data points.")

        if len(df_ts) >= 10:
            try:
                # Forecast period selector
                forecast_days = st.slider("Forecast Period (days)", 30, 180, 90)
                with st.spinner("Generating forecast... This may take a moment."):
                    m = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.05  # Add regularization
                    )
                    m.fit(df_ts)
                    future = m.make_future_dataframe(periods=forecast_days)
                    forecast = m.predict(future)

                    # Plot forecast
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(
                        go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#00d4ff')))
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty',
                                                      fillcolor='rgba(0, 212, 255, 0.2)',
                                                      line=dict(color='rgba(255,255,255,0)'), name='Confidence'))
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty',
                                                      fillcolor='rgba(0, 212, 255, 0.2)',
                                                      line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                    fig_forecast.add_trace(go.Scatter(x=df_ts['ds'], y=df_ts['y'], mode='markers', name='Actual Sales',
                                                      marker=dict(color='#f093fb', size=5)))

                    # Anomaly Detection
                    forecast_actuals = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(
                        df_ts.set_index('ds'))
                    forecast_actuals['anomaly'] = (forecast_actuals['y'] < forecast_actuals['yhat_lower']) | (
                                forecast_actuals['y'] > forecast_actuals['yhat_upper'])
                    anomalies = forecast_actuals[forecast_actuals['anomaly']]
                    fig_forecast.add_trace(
                        go.Scatter(x=anomalies.index, y=anomalies['y'], mode='markers', name='Anomaly',
                                   marker=dict(color='red', size=10, symbol='x')))

                    fig_forecast.update_layout(title_text=f"{forecast_days}-Day Sales Forecast with Anomaly Detection",
                                               plot_bgcolor='#121a3a', paper_bgcolor='#121a3a', font_color='#e0e0e0',
                                               xaxis=dict(gridcolor='#2a3a6b'), yaxis=dict(gridcolor='#2a3a6b'))
                    st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")

# 7. Improve the What-If scenario with better calculations
with tab5:
    st.subheader("💡 What-If Scenario Planner")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Simulation Controls")
        discount_change = st.slider("Adjust Overall Discount (%)", -10.0, 10.0, 0.0, 0.5, key="discount_slider")
        sales_uplift = st.slider("Simulate Sales Uplift (%)", -20.0, 50.0, 0.0, 1.0, key="sales_slider")
        cost_change = st.slider("Simulate Cost Change (%)", -15.0, 15.0, 0.0, 0.5, key="cost_slider")

        # Simulate scenario
        if st.button("🎯 Run Simulation"):
            if not df_filtered.empty:
                sim_df = df_filtered.copy()

                # More sophisticated simulation
                sim_df['sim_discount'] = np.clip(sim_df['discount'] + discount_change / 100, 0, 1)
                sim_df['sim_sales'] = sim_df['sales'] * (1 + sales_uplift / 100)
                sim_df['sim_cost'] = sim_df['cost'] * (1 + cost_change / 100)
                sim_df['sim_profit'] = sim_df['sim_sales'] - sim_df['sim_cost']

                # Store results in session state
                st.session_state.simulation_results = {
                    'original_sales': df_filtered['sales'].sum(),
                    'simulated_sales': sim_df['sim_sales'].sum(),
                    'original_profit': df_filtered['profit'].sum(),
                    'simulated_profit': sim_df['sim_profit'].sum()
                }
                st.success("Simulation complete! View results on the right.")
            else:
                st.warning("Cannot run simulation on empty data.")

    with col2:
        st.markdown("#### Simulated Impact")
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results

            original_sales = results.get('original_sales', 0)
            original_profit = results.get('original_profit', 0)

            sales_impact = results.get('simulated_sales', 0) - original_sales
            profit_impact = results.get('simulated_profit', 0) - original_profit

            sales_impact_pct = (sales_impact / original_sales * 100) if original_sales > 0 else 0
            profit_impact_pct = (profit_impact / original_profit * 100) if original_profit > 0 else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Sales Impact",
                    f"${sales_impact:,.0f}",
                    f"{sales_impact_pct:.1f}%"
                )
            with col_b:
                st.metric(
                    "Profit Impact",
                    f"${profit_impact:,.0f}",
                    f"{profit_impact_pct:.1f}%"
                )
        else:
            st.info("Adjust controls and click 'Run Simulation' to see projected results.")

# --- Sidebar additions ---
# 4. Add data export functionality
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Export Data")
if st.sidebar.button("Export Filtered Data"):
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"odyssey_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# 5. Add performance metrics in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Performance Metrics")
if not df_filtered.empty:
    st.sidebar.metric("Data Quality", f"{(1 - df_filtered.isnull().sum().sum() / df_filtered.size) * 100:.1f}%")
    st.sidebar.metric("Records Loaded", f"{len(df_filtered):,}")
    st.sidebar.metric("Date Range", f"{(end_date - start_date).days} days")

# 6. Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# 8. Add footer with additional info
st.markdown("---")
st.markdown("""<div style="text-align: center; padding: 1rem; color: #a0a0a0;">
    <p>🔮 <strong>Odyssey Strategic Command Center</strong> | Built by Malik Saad</p>
    <p style="font-size: 0.8rem;">Powered by AI & Advanced Analytics • Streamlit • Plotly • Prophet</p>
    <p style="font-size: 0.8rem;">Contact: <a href="mailto:ms347135@gmail.com" style="color: #00d4ff; text-decoration: none;">ms347135@gmail.com</a></p>
</div>""", unsafe_allow_html=True)
