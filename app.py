import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import folium
from folium import plugins
import streamlit as st
from shiny import App, ui, render, reactive
from shiny.types import FileInfo
import asyncio
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from shinywidgets import render_widget, output_widget
from folium.plugins import Draw
from utils.functions import get_display_data # Add this import

# Main application
def create_app():
    # Load data once for UI definition and server
    try:
        df = pd.read_parquet("./data/data.parquet")
        
        # Process country list for UI
        country_list = (
            df[df['is_collab'] == False]
            .drop_duplicates(subset=['country', 'iso2c', 'lat', 'lng', 'cc', 'region'])
            .dropna(subset=['country', 'iso2c'])
            .query("country != '' and iso2c != ''")
            .fillna({'region': 'Other'})
            .sort_values('country')
            .reset_index(drop=True)
        )
        
        # Get data for UI elements
        chemical_categories = sorted(df['chemical'].dropna().unique())
        regions = sorted(country_list['region'].unique())
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())

        # Process article data
        article_data = pd.DataFrame() # Initialize as empty DataFrame
        # Define the columns expected for article_data based on your R script and utils/functions.py
        article_columns_map = {'source': 'source', 'year_x': 'year', 'country_x': 'country', 'percentage_x': 'value'}
        required_raw_cols = list(article_columns_map.keys())

        if all(col in df.columns for col in required_raw_cols):
            article_data_raw = df[required_raw_cols].copy()
            article_data = article_data_raw.rename(columns=article_columns_map)
            
            # Filter out rows with missing essential data, similar to R:
            # !is.na(value) & !is.na(source) & source != ""
            article_data = article_data.dropna(subset=['value', 'source'])
            article_data = article_data[article_data['source'] != ""]
        else:
            print(f"Warning: Not all required columns for article_data found in DataFrame. Missing: {[col for col in required_raw_cols if col not in df.columns]}")
        
        initial_data = {
            'chemical_categories': chemical_categories,
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year
        }
        
        data_objects = {
            'data': df,
            'country_list': country_list,
            'chemical_categories': chemical_categories,
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year,
            'article_data': article_data # Include the processed article_data
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        initial_data = {
            'chemical_categories': ["All"],
            'regions': [],
            'min_year': 1996,
            'max_year': 2022
        }
        data_objects = None

    # UI Definition - use initial_data instead of reactive load_data()
    app_ui = ui.page_navbar(
        ui.nav_panel(
            "Explore Chemical Space",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Filters & Options ⚙️"),
                    ui.input_select(
                        "region_filter",
                        "Filter by Region:",
                        choices=["All"] + initial_data['regions'],
                        selected="All"
                    ),
                    ui.input_slider(
                        "years",
                        "Year Range:",
                        min=initial_data['min_year'],
                        max=initial_data['max_year'],
                        value=[initial_data['min_year'], initial_data['max_year']],
                        step=1
                    ),
                    ui.input_radio_buttons(
                        "chemical_category",
                        "Chemical Space Category:",
                        choices=initial_data['chemical_categories'],
                        selected="All" if "All" in initial_data['chemical_categories'] else initial_data['chemical_categories'][0]
                    ),
                    ui.input_action_button(
                        "clear_selection",
                        "Clear Selection",
                        class_="btn-outline-danger"
                    ),
                    width=300
                ),
                ui.card(
                    ui.card_header("Interactive Chemical Space Map"),
                    # ui.output_ui("map_output"),
                    ui.output_ui("map_output"),
                ),
                ui.navset_card_tab(
                    ui.nav_panel(
                        "Trends",
                        output_widget("main_plot")
                    ),
                    ui.nav_panel(
                        "Contribution Map", 
                        output_widget("contribution_map")
                    ),
                    ui.nav_panel(
                        "Data Table",
                        ui.output_data_frame("summary_table")
                    )
                )
            )
        ),
        ui.nav_panel(
            "Article Highlights",
            ui.navset_card_tab(
                ui.nav_panel(
                    "Main Countries",
                    output_widget("country_cs_plot")
                ),
                ui.nav_panel(
                    "Top Collaborations",
                    ui.input_select(
                        "top_collabs_chem_filter",
                        "Chemical Category:",
                        choices=initial_data['chemical_categories'],
                        selected="All" if "All" in initial_data['chemical_categories'] else initial_data['chemical_categories'][0]
                    ),
                    ui.input_radio_buttons(
                        "top_data_type_filter",
                        "Show Top:",
                        choices={
                            "collabs": "Collaborations",
                            "individuals": "Individual Countries"
                        },
                        selected="collabs"
                    ),
                    output_widget("article_top_collabs_plot")
                ),
                ui.nav_panel("GDP", output_widget("article_gdp_plot")),
                ui.nav_panel("Researchers", output_widget("article_researchers_plot")),
                ui.nav_panel("CS Expansion", output_widget("article_cs_expansion_plot"))
            )
        ),
        title="Chemical Space Explorer",
        id="navbar"
    )

    def server(input, output, session):
        # Use the data loaded above
        if not data_objects:
            return
            
        df_main = data_objects['data'] 
        country_list_main = data_objects['country_list'] # This is the correct country_list for the server scope
        
        # Reactive values
        selected_countries = reactive.Value([])
        display_mode = reactive.Value("compare_individuals") # Keep this as is or add UI to change it

        @reactive.Effect
        @reactive.event(input.map_click_iso)
        def _handle_map_click():
            clicked_iso = input.map_click_iso()
            if clicked_iso:
                current_selection = list(selected_countries())
                # Toggle selection: if already selected, remove; otherwise, add.
                # If you want single selection, just do: selected_countries.set([clicked_iso])
                if clicked_iso in current_selection:
                    current_selection.remove(clicked_iso)
                else:
                    current_selection.append(clicked_iso)
                selected_countries.set(current_selection)

        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])
            # Optionally, reset other inputs if needed
            # ui.update_select("region_filter", selected="All") # Example

        @output
        @render.ui
        def map_output():
            """Render the interactive map"""
            # Ensure you are using country_list_main from the server function's scope
            m = create_folium_map(country_list_main, selected_countries.get())
            return ui.HTML(m._repr_html_())
            
        @output  
        @render_widget
        def main_plot():
            """Main trends plot"""
            # Pass the required arguments to get_display_data
            data = get_display_data(
                df=df_main,
                selected_isos=selected_countries.get(),
                year_range=input.years(),
                chemical_category=input.chemical_category(),
                display_mode=display_mode.get(), # or a specific mode if intended
                region_filter=input.region_filter(),
                country_list=country_list_main
            )
            if data.empty:
                return create_empty_plot("Select countries and filters to view trends")
                
            return create_trends_plot(data, selected_countries.get(), display_mode.get())
            
        @output
        @render_widget  
        def contribution_map():
            """Contribution choropleth map"""
            # Pass the required arguments to get_display_data
            data = get_display_data(
                df=df_main,
                selected_isos=selected_countries.get(),
                year_range=input.years(),
                chemical_category=input.chemical_category(),
                display_mode=display_mode.get(), # or a specific mode
                region_filter=input.region_filter(),
                country_list=country_list_main
            )
            if data.empty:
                return create_empty_plot("No data for contribution map with current selections")
                
            return create_contribution_choropleth(data)
            
        @output
        @render.data_frame
        def summary_table():
            """Summary data table"""
            # Pass the required arguments to get_display_data
            data = get_display_data(
                df=df_main,
                selected_isos=selected_countries.get(),
                year_range=input.years(),
                chemical_category=input.chemical_category(),
                display_mode=display_mode.get(), # or a specific mode
                region_filter=input.region_filter(),
                country_list=country_list_main
            )
            if data.empty:
                return pd.DataFrame({"Message": ["No data available for current selections"]})
                
            return create_summary_dataframe(data, display_mode.get())

        # Article plot outputs
        @output
        @render_widget
        def country_cs_plot():
            article_data = data_objects['article_data']
            if article_data.empty:
                return create_empty_plot("No article data available")
            
            df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
            if df_filtered.empty:
                return create_empty_plot("No country participation data available")
            
            return create_article_plot(df_filtered, "Country participation in the CS")

        @output
        @render_widget
        def article_top_collabs_plot():
            is_collab = input.top_data_type_filter() == "collabs"
            chem_filter = input.top_collabs_chem_filter()
            
            filtered_data = df_main[
                (df_main['is_collab'] == is_collab) & 
                (df_main['chemical'] == chem_filter)
            ]
            
            if filtered_data.empty:
                return create_empty_plot("No data available")
                
            # Get top 10
            top_data = (
                filtered_data.groupby('country')['percentage']
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            
            return create_top_trends_plot(
                filtered_data[filtered_data['country'].isin(top_data.index)],
                f"Top 10 {'Collaborations' if is_collab else 'Countries'}: {chem_filter}"
            )

        @output
        @render_widget
        def article_gdp_plot():
            article_data = data_objects['article_data']
            if article_data.empty:
                # Create dummy data for testing
                dummy_data = pd.DataFrame({
                    'source': ['Annual growth rate of the GDP'] * 69,
                    'year': list(range(2000, 2023)) * 3,
                    'country': ['United States'] * 23 + ['China'] * 23 + ['Germany'] * 23,
                    'value': np.random.uniform(-5, 10, 69),
                })
                return create_gdp_plot(dummy_data)
            
            gdp_data = article_data[article_data['source'] == "Annual growth rate of the GDP"]
            if gdp_data.empty:
                return create_empty_plot("No GDP data available")
                
            return create_gdp_plot(gdp_data)

        @output
        @render_widget
        def article_researchers_plot():
            article_data = data_objects['article_data']
            if article_data.empty:
                # Create dummy data for testing
                dummy_data = pd.DataFrame({
                    'source': ['Number of Researchers'] * 69,
                    'year': list(range(2000, 2023)) * 3,
                    'country': ['United States'] * 23 + ['China'] * 23 + ['Germany'] * 23,
                    'value': np.random.uniform(500000, 2000000, 69),
                })
                return create_researchers_plot(dummy_data)
            
            researchers_data = article_data[article_data['source'] == "Number of Researchers"]
            if researchers_data.empty:
                return create_empty_plot("No researchers data available")
                
            return create_researchers_plot(researchers_data)
        

        @reactive.Effect
        def _validate_chemical_category():
            current_category = input.chemical_category()
            if not current_category or str(current_category).strip() == "":
                # Reset to "All" if invalid selection
                ui.update_select(
                    "chemical_category",
                    selected="All"
                )


        @output
        @render_widget
        def article_cs_expansion_plot():
            article_data = data_objects['article_data']
            if article_data.empty:
                # Create dummy data for testing
                dummy_data = pd.DataFrame({
                    'source': ['Expansion of the CS'] * 69,
                    'year': list(range(2000, 2023)) * 3,
                    'country': ['United States'] * 23 + ['China'] * 23 + ['Germany'] * 23,
                    'value': np.random.uniform(1, 50, 69),
                })
                return create_cs_expansion_plot(dummy_data)
            
            cs_data = article_data[article_data['source'] == "Expansion of the CS"]
            if cs_data.empty:
                return create_empty_plot("No CS expansion data available")
                
            return create_cs_expansion_plot(cs_data)
        
    return App(app_ui, server)


# Helper functions
def create_folium_map(country_list: pd.DataFrame, selected_countries: List[str]) -> folium.Map:
    """Create interactive Folium map"""
    m = folium.Map(location=[30, 10], zoom_start=2)
    
    # Load world geometries
    try:
        world_path = "./data/world_boundaries.geojson"
        world = gpd.read_file(world_path)
        
        iso_column = None
        for possible_col in ['iso_a2', 'ISO_A2', 'iso2c', 'ISO2C']:
            if possible_col in world.columns:
                iso_column = possible_col
                break
        
        if iso_column is None:
            print("Warning: No ISO column found in GeoJSON. Available columns:", world.columns.tolist())
            raise FileNotFoundError("No suitable ISO column found")
                
        for _, country_row in country_list.iterrows():
            iso = country_row['iso2c']
            country_name = country_row['country']
            color = country_row['cc'] if iso in selected_countries else 'lightgray'
            
            country_geo = world[world[iso_column] == iso]
            
            if not country_geo.empty:
                # Create a popup with working JavaScript
                popup_html = f"""
                <div>
                    <b>{country_name} ({iso})</b><br>
                    <button onclick="
                        if (window.parent && window.parent.Shiny) {{
                            window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }} else if (window.Shiny) {{
                            window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }} else {{
                            console.log('Shiny not found, trying alternative...');
                            try {{
                                document.dispatchEvent(new CustomEvent('shiny:inputchanged', {{
                                    detail: {{name: 'map_click_iso', value: '{iso}'}}
                                }}));
                            }} catch(e) {{
                                console.log('Alternative method failed:', e);
                            }}
                        }}
                    " style="padding: 5px 10px; margin: 5px 0; cursor: pointer;">
                        {'Deselect' if iso in selected_countries else 'Select'}
                    </button>
                </div>
                """
                
                folium.GeoJson(
                    country_geo.iloc[0].geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'white',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=f"{country_name} ({iso})",
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)
            else:
                # Fallback to marker
                popup_html = f"""
                <div>
                    <b>{country_name} ({iso})</b><br>
                    <button onclick="
                        if (window.parent && window.parent.Shiny) {{
                            window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }} else if (window.Shiny) {{
                            window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }}
                    " style="padding: 5px 10px; margin: 5px 0; cursor: pointer;">
                        {'Deselect' if iso in selected_countries else 'Select'}
                    </button>
                </div>
                """
                
                folium.CircleMarker(
                    location=[country_row['lat'], country_row['lng']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"{country_name} ({iso})"
                ).add_to(m)
                
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        # Fallback to markers
        for _, country in country_list.iterrows():
            color = country['cc'] if country['iso2c'] in selected_countries else 'lightgray'
            iso = country['iso2c']
            country_name = country['country']
            
            popup_html = f"""
            <div>
                <b>{country_name} ({iso})</b><br>
                <button onclick="
                    if (window.parent && window.parent.Shiny) {{
                        window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }} else if (window.Shiny) {{
                        window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }}
                " style="padding: 5px 10px; margin: 5px 0; cursor: pointer;">
                    {'Deselect' if iso in selected_countries else 'Select'}
                </button>
            </div>
            """
            
            folium.CircleMarker(
                location=[country['lat'], country['lng']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=country_name
            ).add_to(m)
    
    return m

def create_trends_plot(data: pd.DataFrame, selected_countries: List[str], mode: str):
    """Create trends plot using Plotly"""
    fig = go.Figure()
    
    # Check which column name is used for the percentage values
    # Possible names: 'percentage', 'value', 'value_raw', 'percentage_x'
    value_column = None
    for possible_col in ['percentage', 'value', 'value_raw', 'percentage_x']:
        if possible_col in data.columns:
            value_column = possible_col
            break
    
    if value_column is None:
        # No valid column found, create empty plot with error message
        fig.add_annotation(
            text="Error: Missing value column in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    if mode == "compare_individuals" or len(selected_countries) == 1:
        # Individual country trends
        for country in data['country'].unique():
            country_data = data[data['country'] == country]
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data[value_column],
                mode='lines+markers',
                name=country,
                hovertemplate=f"<b>{country}</b><br>" +
                             "Year: %{x}<br>" +
                             f"{value_column}: %{{y:.2f}}%<extra></extra>"
            ))
    else:
        # Collaboration trends
        for collab in data['iso2c'].unique():
            collab_data = data[data['iso2c'] == collab]
            fig.add_trace(go.Scatter(
                x=collab_data['year'],
                y=collab_data[value_column], 
                mode='lines+markers',
                name=collab,
                hovertemplate=f"<b>{collab}</b><br>" +
                             "Year: %{x}<br>" +
                             f"{value_column}: %{{y:.2f}}%<extra></extra>"
            ))
    
    fig.update_layout(
        title="Chemical Space Contribution Trends",
        xaxis_title="Year",
        yaxis_title="% of New Substances",
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def create_contribution_choropleth(data: pd.DataFrame):
    """Create world choropleth map"""
    # Check which column name is used for the percentage values
    value_column = None
    for possible_col in ['percentage', 'value', 'value_raw', 'percentage_x']:
        if possible_col in data.columns:
            value_column = possible_col
            break
    
    if value_column is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Error: Missing value column in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Calculate average contribution by country
    avg_data = data.groupby(['iso2c', 'country'])[value_column].mean().reset_index()
    
    fig = px.choropleth(
        avg_data,
        locations='iso2c',
        color=value_column,
        hover_name='country',
        color_continuous_scale='Viridis',
        title='Average Chemical Space Contribution'
    )
    
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        title_x=0.5
    )
    
    return fig

def create_summary_dataframe(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Create summary statistics table"""
    # Check which column name is used for the percentage values
    value_column = None
    for possible_col in ['percentage', 'value', 'value_raw', 'percentage_x']:
        if possible_col in data.columns:
            value_column = possible_col
            break
    
    if value_column is None:
        # Return empty DataFrame with message
        return pd.DataFrame({'Error': ['Missing value column in data']})
    
    if mode == "find_collaborations":
        summary = (
            data.groupby(['iso2c', 'chemical'])
            .agg({
                value_column: ['mean', 'max', 'count']
            })
            .round(2)
            .reset_index()
        )
        summary.columns = ['Collaboration', 'Chemical', 'Avg %', 'Max %', 'Years Present']
    else:
        summary = (
            data.groupby(['country', 'iso2c', 'chemical'])
            .agg({
                value_column: ['mean', 'max', 'count']
            })
            .round(2)
            .reset_index()
        )
        summary.columns = ['Country', 'ISO', 'Chemical', 'Avg %', 'Max %', 'Years Present']
    
    return summary

def create_article_plot(data: pd.DataFrame, title: str):
    """Create article plots"""
    fig = go.Figure()
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Figure: {title}",
        xaxis_title="Year",
        yaxis_title="Value",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_top_trends_plot(data: pd.DataFrame, title: str):
    """Create top contributors/collaborations plot"""
    fig = go.Figure()
    
    for entity in data['country'].unique():
        entity_data = data[data['country'] == entity]
        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data['percentage'],
            mode='lines+markers',
            name=entity,
            line=dict(width=1.5),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year", 
        yaxis_title="% of New Substances",
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    return fig

def create_empty_plot(message: str):
    """Create empty plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white'
    )
    return fig


def create_gdp_plot(data: pd.DataFrame):
    """Create GDP article plot with annotations for economic events"""
    fig = go.Figure()
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    # Add vertical lines and annotations for economic events
    fig.add_vline(x=2007.5, line_dash="dash", line_color="grey")
    fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")
    
    # Calculate y-position for annotations based on data
    if fig.data:
        max_val = max([max(trace['y']) for trace in fig.data])
        fig.add_annotation(
            x=2007.5, 
            y=max_val * 0.9,
            text="Financial Crisis", 
            showarrow=True,
            arrowhead=1
        )
        fig.add_annotation(
            x=2019.5, 
            y=max_val * 0.8,
            text="COVID-19", 
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        title="Figure: Annual growth rate of the GDP",
        xaxis_title="Year",
        yaxis_title="GDP Growth Rate (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_researchers_plot(data: pd.DataFrame):
    """Create researchers plot with values in millions"""
    fig = go.Figure()
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        scaled_values = country_data['value'] / 1e6  # Convert to millions
        
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=scaled_values,
            mode='lines+markers',
            name=country,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Figure: Number of Researchers",
        xaxis_title="Year",
        yaxis_title="Number of Researchers (Millions)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_cs_expansion_plot(data: pd.DataFrame):
    """Create chemical space expansion plot"""
    fig = go.Figure()
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Figure: Chemical Space Expansion",
        xaxis_title="Year",
        yaxis_title="Expansion Rate",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run()