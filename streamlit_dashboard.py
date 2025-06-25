import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
import os # Import os for file operations

# IMPORTANT: This script assumes 'definitive_solar_analysis.py' is in the same directory
# so it can import the DefinitiveSolarFix class to run the analysis if a raw CSV is uploaded.
# If you are only uploading the already processed 'definitive_solar_analysis.csv',
# the import can be removed, and you can directly load that CSV.
from definitive_solar_analysis import DefinitiveSolarFix

def create_streamlit_dashboard():
    """
    Creates an interactive Streamlit dashboard for solar loss analysis.
    Users can upload their raw dataset (Dataset 1.csv) or an already processed
    definitive_solar_analysis.csv.
    """
    
    st.set_page_config(
        page_title="Solar Energy Loss Analysis Dashboard",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌞 Solar Energy Loss Analysis Dashboard")
    st.markdown("**Deconstructing Solar Energy Losses - Zelestra Phase 2**")
    
    # Sidebar for file upload and controls
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (Dataset 1.csv or definitive_solar_analysis.csv)", type=['csv'])
    
    df = None
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        try:
            if file_name == "Dataset 1.csv":
                st.info("Processing raw 'Dataset 1.csv'. This may take a moment.")
                # If raw data is uploaded, run the full analysis using DefinitiveSolarFix
                temp_path = f"temp_uploaded_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                analyzer = DefinitiveSolarFix(temp_path)
                df = analyzer.run_definitive_analysis()
                os.remove(temp_path) # Clean up temp file
                st.success("Analysis complete! Displaying results for raw data.")

            elif file_name == "definitive_solar_analysis.csv":
                st.info("Loading pre-processed 'definitive_solar_analysis.csv'.")
                df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                # Ensure 'datetime' index name is set if it was lost
                if df.index.name is None:
                    df.index.name = 'datetime'
                st.success("Pre-processed data loaded successfully.")
            else:
                st.warning("Please upload 'Dataset 1.csv' or 'definitive_solar_analysis.csv'.")
        except Exception as e:
            st.error(f"Error processing file: {e}. Please ensure the CSV format is correct.")
            df = None # Reset df to None on error
    
    if df is not None:
        # Filter data by date range if selected
        st.sidebar.header("🔧 Filters")
        # Ensure date_range is within the actual data's min/max dates
        min_date_df = df.index.min().date()
        max_date_df = df.index.max().date()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[min_date_df, max_date_df],
            min_value=min_date_df,
            max_value=max_date_df
        )

        # Ensure date_range is a tuple/list of two dates
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df.index.date >= start_date) & (df.index.date <= end_date)].copy() # Use .copy() to avoid SettingWithCopyWarning
        else:
            df_filtered = df.copy() # No date filter applied if only one date selected, use .copy()
        
        # Ensure 'avg_module_temp' is available for the dashboard even if not used in primary analysis
        if 'avg_module_temp' not in df_filtered.columns:
            # Attempt to create from raw data if possible, or fill with default
            temp_cols = [col for col in df_filtered.columns if 't_mod' in col or 't_amb' in col]
            if temp_cols:
                df_filtered['avg_module_temp'] = df_filtered[temp_cols].mean(axis=1).fillna(25)
            else:
                df_filtered['avg_module_temp'] = 25.0 # Default if no temp data found
                
        # Ensure 'actual_power_mw' is available for dashboard
        if 'actual_power_mw' not in df_filtered.columns:
            # Recreate from actual_energy_15min if possible
            if 'actual_energy_15min' in df_filtered.columns:
                df_filtered['actual_power_mw'] = df_filtered['actual_energy_15min'] / 0.25
            else:
                df_filtered['actual_power_mw'] = 0.0 # Default to 0 if no actual energy

        # Main dashboard content
        st.header("✨ Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        total_theoretical_energy = df_filtered['theoretical_energy_15min'].sum()
        total_actual_energy = df_filtered['actual_energy_15min'].sum()
        overall_pr = total_actual_energy / total_theoretical_energy if total_theoretical_energy > 0 else 0
        total_energy_gap = total_theoretical_energy - total_actual_energy
        system_availability = (df_filtered['actual_energy_15min'] > 0.01).mean() * 100 # Availability if producing > 0.01 MWh

        with col1:
            st.metric(
                "Total Theoretical Energy",
                f"{total_theoretical_energy:.1f} MWh"
            )
        
        with col2:
            st.metric(
                "Total Actual Energy",
                f"{total_actual_energy:.1f} MWh",
                delta=f"Lost: {total_energy_gap:.1f} MWh" if total_energy_gap > 0 else "No Loss",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Performance Ratio",
                f"{overall_pr:.3f}",
                delta=f"{((overall_pr - 0.85) * 100):.1f}% vs target (0.85)"
            )
        
        with col4:
            st.metric(
                "System Availability",
                f"{system_availability:.1f}%"
            )
        
        # Loss breakdown charts
        st.header("📊 Energy Loss Breakdown")
        
        col1, col2 = st.columns(2)
        
        loss_data = {
            'Cloud': df_filtered['cloud_energy_loss'].sum(),
            'Shading': df_filtered['shading_energy_loss'].sum(),
            'Temperature': df_filtered['temperature_energy_loss'].sum(),
            'Soiling': df_filtered['soiling_energy_loss'].sum(),
            'Inverter': df_filtered['inverter_energy_loss'].sum(),
            'Curtailment': df_filtered['curtailment_energy_loss'].sum(),
            'Tracker Malfunction': df_filtered['tracker_malfunction_energy_loss'].sum(),
            'Other': df_filtered['other_energy_loss'].sum()
        }
        # Filter out very small losses for cleaner visualization
        loss_data_filtered = {k: v for k, v in loss_data.items() if v > 0.01}
        
        with col1:
            if loss_data_filtered:
                fig_pie = px.pie(
                    values=list(loss_data_filtered.values()),
                    names=list(loss_data_filtered.keys()),
                    title="Energy Loss Distribution (MWh)",
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.RdBu # Diverging color scale
                )
                fig_pie.update_traces(textinfo='percent+label', pull=[0.1 if k == max(loss_data_filtered, key=loss_data_filtered.get) else 0 for k in loss_data_filtered.keys()])
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No significant energy losses to display in the pie chart for the selected period.")

        with col2:
            # Daily energy comparison
            daily_data = df_filtered.resample('D').agg({
                'theoretical_energy_15min': 'sum',
                'actual_energy_15min': 'sum'
            })
            
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['theoretical_energy_15min'],
                name='Theoretical',
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            fig_daily.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['actual_energy_15min'],
                name='Actual',
                mode='lines',
                line=dict(color='red', width=2)
            ))
            fig_daily.update_layout(
                title="Daily Energy: Theoretical vs Actual",
                xaxis_title="Date",
                yaxis_title="Energy (MWh)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Time-series analysis for flags and losses
        st.header("📈 Time Series Trends of Losses")
        
        analysis_type = st.selectbox(
            "Select Time Resolution for Trends",
            ["15-minute", "Hourly", "Daily", "Weekly", "Monthly"]
        )
        
        if analysis_type == "15-minute":
            resample_freq = '15T'
        elif analysis_type == "Hourly":
            resample_freq = 'H'
        elif analysis_type == "Daily":
            resample_freq = 'D'
        elif analysis_type == "Weekly":
            resample_freq = 'W'
        else:  # Monthly
            resample_freq = 'M'
        
        # Columns for summing flags and energy losses
        flag_cols = ['CloudCover', 'Shading', 'TemperatureEffect', 'Soiling', 'InverterLoss', 'Curtailment', 'TrackerMalfunction', 'OtherLosses']
        energy_loss_cols_plot = ['cloud_energy_loss', 'shading_energy_loss', 'temperature_energy_loss', 'soiling_energy_loss', 'inverter_energy_loss', 'curtailment_energy_loss', 'tracker_malfunction_energy_loss', 'other_energy_loss']

        # Resample data based on selection
        ts_data = df_filtered.resample(resample_freq).agg({
            **{col: 'sum' for col in flag_cols}, # Sum for event counts
            **{col: 'sum' for col in energy_loss_cols_plot}, # Sum for energy losses
            'performance_ratio': 'mean' # Mean for PR
        }).fillna(0) # Fill any NaNs after resampling with 0 for plotting
        
        fig_ts = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{analysis_type} Loss Event Counts', f'{analysis_type} Energy Loss Amount (MWh)'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Loss events count (stacked area)
        for i, col in enumerate(flag_cols):
            fig_ts.add_trace(go.Scatter(
                x=ts_data.index, 
                y=ts_data[col], 
                mode='lines', 
                name=col.replace("Loss", " Loss").replace("Effect", " Effect"),
                stackgroup='one', # Stacks the areas
                line={'width': 0.5},
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{col.replace("Loss", " Loss").replace("Effect", " Effect")} Events</b>: %{{y}}<extra></extra>',
                # Show legend for all traces for clarity in stacked charts
                showlegend=True, 
                legendgroup='event_counts' # Group for legend
            ), row=1, col=1)
        fig_ts.update_yaxes(title_text="Number of Events", row=1, col=1)
        
        # Energy losses (stacked area)
        for i, col in enumerate(energy_loss_cols_plot):
            fig_ts.add_trace(go.Scatter(
                x=ts_data.index, 
                y=ts_data[col], 
                mode='lines', 
                name=col.replace("_energy_loss", " Loss").replace("_", " ").title(),
                stackgroup='two', # Stacks the areas
                line={'width': 0.5},
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{col.replace("_energy_loss", " Loss").replace("_", " ").title()}</b>: %{{y:.2f}} MWh<extra></extra>',
                # Show legend for all traces for clarity in stacked charts
                showlegend=True,
                legendgroup='energy_losses' # Group for legend
            ), row=2, col=1)
        fig_ts.update_yaxes(title_text="Energy Loss (MWh)", row=2, col=1)

        fig_ts.update_layout(height=700, showlegend=True, hovermode="x unified") # Ensure overall legend is shown and hover is unified
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # New: Daily Total Losses (Stacked Bar Chart)
        st.header("Daily Losses by Category")
        daily_losses_data = df_filtered[energy_loss_cols_plot].resample('D').sum().fillna(0)
        
        fig_daily_losses = go.Figure()
        for col in energy_loss_cols_plot:
            fig_daily_losses.add_trace(go.Bar(
                x=daily_losses_data.index,
                y=daily_losses_data[col],
                name=col.replace("_energy_loss", "").replace("_", " ").title()
            ))
        
        fig_daily_losses.update_layout(
            barmode='stack',
            title='Daily Total Losses by Category',
            xaxis_title='Date',
            yaxis_title='Energy Loss (MWh)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_daily_losses, use_container_width=True)


        # Asset-level analysis (simplified, as detailed string data is limited)
        st.header("🏭 Asset-Level Insights")
        
        col_inv1, col_inv2 = st.columns(2)
        
        with col_inv1:
            st.subheader("Inverter Power Output Over Time")
            # Assuming 'inversores_ctin03_inv_03_03_p' and 'inversores_ctin08_inv_08_08_p' exist
            inverter_power_cols = [col for col in df_filtered.columns if 'inv_' in col and '_p' in col and 'p_dc' not in col]
            
            if inverter_power_cols:
                inv_power_data = df_filtered[inverter_power_cols].copy() # Get relevant power columns
                # Convert to MW if necessary (assuming they might be in kW)
                # Let's verify by checking max value. If > 1000, probably kW, so divide by 1000
                if inv_power_data.max().max() > 100: # Heuristic: if max power exceeds 100kW, assume it's in W or kW
                    inv_power_data = inv_power_data / 1000 # Convert to MW
                
                inv_power_data.columns = [col.replace('inversores_ctin', 'INV-').replace('_inv_', '-').replace('_p', ' Power (MW)') for col in inverter_power_cols]
                
                fig_inv = px.line(inv_power_data.dropna(), title="Inverter Power Output Over Time")
                fig_inv.update_yaxes(title_text="Power (MW)")
                fig_inv.update_xaxes(title_text="Date")
                st.plotly_chart(fig_inv, use_container_width=True)
            else:
                st.info("Inverter power data not available for detailed comparison.")
                
        with col_inv2:
            st.subheader("Aggregated Inverter Energy Output")
            # Using specific column names as provided in Dataset 1.csv for direct summation
            inv_03_power = df_filtered.get('inversores_ctin03_inv_03_03_p', pd.Series(0.0, index=df_filtered.index))
            inv_08_power = df_filtered.get('inversores_ctin08_inv_08_08_p', pd.Series(0.0, index=df_filtered.index))

            # Convert power (assumed in kW from previous analysis) to MWh
            inv_03_total_energy = inv_03_power.sum() * 0.25 / 1000
            inv_08_total_energy = inv_08_power.sum() * 0.25 / 1000

            inv_totals = pd.DataFrame({
                'Inverter': ['INV-03', 'INV-08'],
                'Total Energy (MWh)': [inv_03_total_energy, inv_08_total_energy]
            })
            
            fig_inv_bar = px.bar(inv_totals, x='Inverter', y='Total Energy (MWh)', 
                                 title='Total Energy Output by Inverter')
            st.plotly_chart(fig_inv_bar, use_container_width=True)

        # Environmental correlation analysis
        st.header("🌡️ Environmental & Operational Correlations")
        
        col_env1, col_env2 = st.columns(2)
        
        with col_env1:
            st.subheader("Module Temperature vs Performance Ratio")
            if 'avg_module_temp' in df_filtered.columns and 'performance_ratio' in df_filtered.columns:
                fig_temp_pr = px.scatter(
                    df_filtered.dropna(subset=['avg_module_temp', 'performance_ratio']),
                    x='avg_module_temp',
                    y='performance_ratio',
                    color='TemperatureEffect', # Color by temperature effect flag
                    title="Module Temperature vs Performance Ratio",
                    labels={'avg_module_temp': 'Module Temperature (°C)', 'performance_ratio': 'Performance Ratio'}
                )
                st.plotly_chart(fig_temp_pr, use_container_width=True)
            else:
                st.info("Module temperature or performance ratio data not available.")
        
        with col_env2:
            st.subheader("POA Irradiance vs Actual Power")
            if 'poa_irradiance' in df_filtered.columns and 'actual_power_mw' in df_filtered.columns:
                fig_irr_power = px.scatter(
                    df_filtered.dropna(subset=['poa_irradiance', 'actual_power_mw']),
                    x='poa_irradiance',
                    y='actual_power_mw',
                    color='CloudCover', # Color by cloud cover flag
                    title="POA Irradiance vs Actual Power",
                    labels={'poa_irradiance': 'POA Irradiance (W/m²)', 'actual_power_mw': 'Actual Power (MW)'}
                )
                st.plotly_chart(fig_irr_power, use_container_width=True)
            else:
                st.info("POA irradiance or actual power data not available.")
        
        # New: Hourly Average Performance Ratio Trend
        st.header("⏱️ Hourly Performance Trend")
        if 'performance_ratio' in df_filtered.columns and 'is_day' in df_filtered.columns:
            # Filter for daytime and relevant PR values (not zeroed out night values)
            hourly_pr_data = df_filtered[df_filtered['is_day'] & (df_filtered['theoretical_energy_15min'] > 0.1)].copy()
            hourly_pr_data['hour'] = hourly_pr_data.index.hour
            hourly_avg_pr = hourly_pr_data.groupby('hour')['performance_ratio'].mean().reset_index()
            
            fig_hourly_pr = px.line(
                hourly_avg_pr, 
                x='hour', 
                y='performance_ratio', 
                title='Average Performance Ratio by Hour of Day',
                labels={'hour': 'Hour of Day', 'performance_ratio': 'Average Performance Ratio'},
                markers=True
            )
            fig_hourly_pr.update_yaxes(range=[0, 1.0]) # PR is typically between 0 and 1
            st.plotly_chart(fig_hourly_pr, use_container_width=True)
        else:
            st.info("Performance ratio data not sufficient for hourly trend analysis.")

        # New: POA Irradiance Distribution
        st.header("☀️ Irradiance Distribution")
        if 'poa_irradiance' in df_filtered.columns:
            fig_irr_hist = px.histogram(
                df_filtered[df_filtered['poa_irradiance'] > 10].dropna(subset=['poa_irradiance']), # Filter out night zeros
                x='poa_irradiance',
                nbins=50,
                title='Distribution of POA Irradiance',
                labels={'poa_irradiance': 'POA Irradiance (W/m²)'}
            )
            st.plotly_chart(fig_irr_hist, use_container_width=True)
        else:
            st.info("POA Irradiance data not available for distribution analysis.")


        st.markdown("---")
        st.markdown("Disclaimer: Economic and maintenance recommendations are illustrative and depend on specific operational contexts and costs.")
    
    else:
        st.info("👆 Please upload your dataset to begin the analysis. You can upload either the raw 'Dataset 1.csv' or the pre-processed 'definitive_solar_analysis.csv'.")
        
        # Show sample data format expectations
        st.subheader("Expected Raw Data Format (Dataset 1.csv)")
        st.write("Your raw CSV should contain columns similar to:")
        expected_raw_cols = [
            'datetime', 'meteorolgicas_em_03_02_gii', 'meteorolgicas_em_08_01_gii',
            'inversores_ctin03_inv_03_03_p', 'inversores_ctin08_inv_08_08_p',
            'ppc_eact_imp', # Plant level energy
            'celulas_ctin03_cc_03_1_t_mod', 'celulas_ctin08_cc_08_1_t_mod', # Module temperatures
            'inversores_ctin03_inv_03_03_eact_tot', 'inversores_ctin08_inv_08_08_eact_tot', # Inverter cumulative energy
            'inversores_ctin03_inv_03_03_p_dc', 'inversores_ctin08_inv_08_08_p_dc', # DC Power
            'seguidores_ct03_gcu031_t0308035_pos_ang', 'seguidores_ct08_gcu081_t0808029_pos_ang', # Tracker angles
            # Example string currents (if available and used for shading)
            'cadenas_ct03_s1_pv_i', 'cadenas_ct03_s2_pv_i', # and so on for other strings
        ]
        st.code('\n'.join(expected_raw_cols))

        st.subheader("Expected Pre-processed Data Format (definitive_solar_analysis.csv)")
        st.write("If uploading pre-processed data, it should include columns like:")
        expected_processed_cols = [
            'theoretical_energy_15min', 'actual_energy_15min', 'performance_ratio',
            'CloudCover', 'Shading', 'TemperatureEffect', 'Soiling', 'InverterLoss',
            'Curtailment', 'TrackerMalfunction', 'OtherLosses',
            'cloud_energy_loss', 'shading_energy_loss', 'temperature_energy_loss',
            'soiling_energy_loss', 'inverter_energy_loss', 'curtailment_energy_loss',
            'tracker_malfunction_energy_loss', 'other_energy_loss',
            'energy_gap', 'availability', 'is_day', 'is_night', 'avg_module_temp',
            'system_efficiency', 'solar_elevation', 'poa_irradiance'
        ]
        st.code('\n'.join(expected_processed_cols))


if __name__ == "__main__":
    create_streamlit_dashboard()

