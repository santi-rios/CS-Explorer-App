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
            df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
            return create_article_plot(df_filtered, "Country participation in the CS")
            
        @output
        @render_widget
        def article_top_collabs_plot():
            is_collab = input.top_data_type_filter() == "collabs"
            chem_filter = input.top_collabs_chem_filter()
            
            filtered_data = df[
                (df['is_collab'] == is_collab) & 
                (df['chemical'] == chem_filter)
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

        # Additional article plots would follow similar pattern...
        
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
        for possible_col in ['iso_a2']: # Assuming 'iso_a2' is the correct column in your GeoJSON
            if possible_col in world.columns:
                iso_column = possible_col
                break
        
        if iso_column is None:
            print("Warning: No ISO column found in GeoJSON. Available columns:", world.columns.tolist())
            raise FileNotFoundError("No suitable ISO column found")
                
        for _, country_row in country_list.iterrows(): # Renamed 'country' to 'country_row' to avoid conflict
            iso = country_row['iso2c']
            country_name = country_row['country'] # Use 'country_name' for clarity
            color = country_row['cc'] if iso in selected_countries else 'lightgray'
            
            country_geo = world[world[iso_column] == iso]
            
            if not country_geo.empty:
                # JavaScript to send ISO code to Shiny when popup button is clicked
                js_onclick_call = f"Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});"
                
                # Create a more informative popup with a selection button
                popup_html = f"""
                <b>{country_name} ({iso})</b><br>
                <button onclick="{js_onclick_call}">
                    {'Deselect' if iso in selected_countries else 'Select'}
                </button>
                """
                
                folium.GeoJson(
                    country_geo.iloc[0].geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'white', # Border color of the polygon
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=f"{country_name} ({iso})", # Tooltip on hover
                    popup=folium.Popup(popup_html) # Popup on click
                ).add_to(m)
            else:
                # Fallback to marker if country not found in GeoJSON
                folium.CircleMarker(
                    location=[country_row['lat'], country_row['lng']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"{country_name} ({iso}) - Data point", # Simpler popup for markers
                    tooltip=f"{country_name} ({iso})"
                ).add_to(m)
                
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        # Fallback to markers if any issue with GeoJSON
        for _, country in country_list.iterrows():
            color = country['cc'] if country['iso2c'] in selected_countries else 'lightgray'
            folium.CircleMarker(
                location=[country['lat'], country['lng']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=country['country'],
                tooltip=country['country']
            ).add_to(m)
    
    return m

def create_trends_plot(data: pd.DataFrame, selected_countries: List[str], mode: str):
    """Create trends plot using Plotly"""
    fig = go.Figure()
    
    if mode == "compare_individuals" or len(selected_countries) == 1:
        # Individual country trends
        for country in data['country'].unique():
            country_data = data[data['country'] == country]
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data['percentage'],
                mode='lines+markers',
                name=country,
                hovertemplate=f"<b>{country}</b><br>" +
                             "Year: %{x}<br>" +
                             "Percentage: %{y:.2f}%<extra></extra>"
            ))
    else:
        # Collaboration trends
        for collab in data['iso2c'].unique():
            collab_data = data[data['iso2c'] == collab]
            fig.add_trace(go.Scatter(
                x=collab_data['year'],
                y=collab_data['percentage'], 
                mode='lines+markers',
                name=collab,
                hovertemplate=f"<b>{collab}</b><br>" +
                             "Year: %{x}<br>" +
                             "Percentage: %{y:.2f}%<extra></extra>"
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
    # Calculate average contribution by country
    avg_data = data.groupby(['iso2c', 'country'])['percentage'].mean().reset_index()
    
    fig = px.choropleth(
        avg_data,
        locations='iso2c',
        color='percentage',
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
    if mode == "find_collaborations":
        summary = (
            data.groupby(['iso2c', 'chemical'])
            .agg({
                'percentage': ['mean', 'max', 'count']
            })
            .round(2)
            .reset_index()
        )
        summary.columns = ['Collaboration', 'Chemical', 'Avg %', 'Max %', 'Years Present']
    else:
        summary = (
            data.groupby(['country', 'iso2c', 'chemical'])
            .agg({
                'percentage': ['mean', 'max', 'count']
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

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run()