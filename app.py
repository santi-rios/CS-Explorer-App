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
from utils.functions import get_display_data
import functools
from functools import lru_cache

# Global cache for expensive operations
@lru_cache(maxsize=128)
def cached_get_display_data(
    selected_isos_tuple: tuple,
    year_range: tuple,
    chemical_category: str,
    display_mode: str,
    region_filter: str = "All"
):
    """Cached version of get_display_data for performance"""
    # Convert back to list for compatibility
    selected_isos = list(selected_isos_tuple) if selected_isos_tuple else []
    
    # Only load data when actually needed
    df = pd.read_parquet("./data/data.parquet")
    country_list = load_country_list()
    
    return get_display_data(
        df=df,
        selected_isos=selected_isos,
        year_range=year_range,
        chemical_category=chemical_category,
        display_mode=display_mode,
        region_filter=region_filter,
        country_list=country_list
    )

@lru_cache(maxsize=1)
def load_country_list():
    """Cached country list loading"""
    df = pd.read_parquet("./data/data.parquet")
    return (
        df[df['is_collab'] == False]
        .drop_duplicates(subset=['country', 'iso2c', 'lat', 'lng', 'cc', 'region'])
        .dropna(subset=['country', 'iso2c'])
        .query("country != '' and iso2c != ''")
        .fillna({'region': 'Other'})
        .sort_values('country')
        .reset_index(drop=True)
    )

# Main application
def create_app():
    # Load minimal data for UI initialization only
    try:
        # Only read metadata for UI, not full dataset
        df_sample = pd.read_parquet("./data/data.parquet", columns=['chemical', 'year', 'region'])
        
        unique_chemicals = df_sample['chemical'].dropna().unique().tolist()
        chemical_categories = ['All'] + sorted([chem for chem in unique_chemicals if chem and str(chem).strip()])
        chemical_categories = list(dict.fromkeys(chemical_categories))

        unique_regions = df_sample['region'].fillna('Other').unique().tolist()
        regions = ['All'] + sorted(list(set(region for region in unique_regions if region and str(region).strip() and region != 'All')))
        
        min_year = int(df_sample['year'].min())
        max_year = int(df_sample['year'].max())
        
        initial_data = {
            'chemical_categories': chemical_categories,
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year
        }
        
    except Exception as e:
        print(f"Error loading initial data for UI: {e}")
        initial_data = {
            'chemical_categories': ["All"],
            'regions': ["All"],
            'min_year': 1996,
            'max_year': 2022
        }

    app_ui = ui.page_navbar(
                ui.nav_panel(
                    "Dashboard", # This will be the main landing page
                    ui.div(
                        # --- Section 1: Article Highlights ---
                        ui.h3("Key Article Insights", style="text-align: center; margin-bottom: 20px; margin-top: 20px;"),
                        ui.row(
                            ui.column(12, 
                                ui.navset_card_tab(
                                    ui.nav_panel(
                                        "🏆 Main Countries",
                                        ui.card(
                                            output_widget("country_cs_plot")
                                        )
                                    ),
                                    ui.nav_panel(
                                        "💰 GDP",
                                        ui.card(
                                            output_widget("article_gdp_plot")
                                        )
                                    ),
                                    ui.nav_panel(
                                        "👥 Researchers",
                                        ui.card(
                                            output_widget("article_researchers_plot")
                                        )
                                    ),
                                    # You can add the CS Expansion plot here as another tab if desired
                                    # ui.nav_panel(
                                    #     "📊 CS Expansion",
                                    #     ui.card(
                                    #         output_widget("article_cs_expansion_plot")
                                    #     )
                                    # )
                                )
                            )
                        ),
                        ui.row(
                            ui.column(12, ui.card(
                                ui.card_header("🤝 Top Collaborations"),
                                ui.row( # Filters for this specific plot
                                    ui.column(6, ui.input_select("top_collabs_chem_filter", "Chemical Category:", choices=initial_data['chemical_categories'], selected="All")),
                                    ui.column(6, ui.input_radio_buttons("top_data_type_filter", "Show Top:", choices={"collabs": "Collaborations", "individuals": "Countries"}, selected="collabs"))
                                ),
                                output_widget("article_top_collabs_plot")
                            ))
                        ),
                        # The CS Expansion plot was here. If you want to keep it, 
                        # you could add it as another tab above or in a new row.
                        # For example, to add it as a new row:
                        # ui.row(
                        #     ui.column(12, ui.card(
                        #         ui.card_header("� CS Expansion"),
                        #         output_widget("article_cs_expansion_plot")
                        #     ))
                        # ),
        
                        ui.hr(style="margin-top: 30px; margin-bottom: 30px; border-top: 2px solid #007bff;"),

                ui.hr(style="margin-top: 30px; margin-bottom: 30px; border-top: 2px solid #007bff;"),

                # --- Section 2: Explore Chemical Space ---
                ui.h3("Explore Chemical Space Interactively", style="text-align: center; margin-bottom: 20px;"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h5("⚙️ Filters & Options"),
                        ui.input_select(
                            "region_filter", "🌍 Filter by Region:",
                            choices=initial_data['regions'], selected="All"
                        ),
                        ui.input_slider(
                            "years", "📅 Year Range:",
                            min=initial_data['min_year'], max=initial_data['max_year'],
                            value=[initial_data['min_year'], initial_data['max_year']],
                            step=1, sep=""
                        ),
                        ui.input_select(
                            "chemical_category", "🧪 Chemical Space:",
                            choices=initial_data['chemical_categories'], selected="All"
                        ),
                        ui.input_radio_buttons(
                            "display_mode_input", "📊 Display Mode:",
                            choices={
                                "compare_individuals": "Compare Countries",
                                "find_collaborations": "Find Collaborations"
                            },
                            selected="compare_individuals"
                        ),
                        ui.div(
                            ui.input_action_button(
                                "clear_selection", "🗑️ Clear Selection",
                                class_="btn-outline-danger w-100"
                            ),
                            ui.div(ui.output_text("selection_info"), class_="mt-2 text-muted small")
                        ),
                        ui.hr(style="margin-top: 20px; margin-bottom: 15px;"),
                        ui.h5("🌍 Global Contribution Snapshot"),
                        output_widget("contribution_map"), # Global Contribution Map in sidebar
                        width=370 # Adjust width for better map display in sidebar
                    ),
                    # Main content area for the "Explore" section
                    ui.card( 
                        ui.card_header("🗺️ Interactive Chemical Space Map"),
                        ui.output_ui("map_output") 
                    ),
                    ui.navset_card_tab( 
                        ui.nav_panel("📈 Trends", output_widget("main_plot")),
                        ui.nav_panel("📋 Data Table", ui.output_data_frame("summary_table"))
                    )
                )
            )
        ),
        # --- Other nav panels remain for auxiliary content ---
        ui.nav_panel(
            "📖 Original Article",
            ui.tags.iframe(
                src="original_article.pdf", # Ensure this is in the www/ directory
                style="width: 100%; height: 80vh; border: none;"
            )
        ),
        ui.nav_panel(
            "🔗 Useful Links",
            ui.div(
                ui.h4("Project & Resources"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.a("GitHub Repository", href="YOUR_GITHUB_REPO_LINK_HERE", target="_blank")),
                    ui.tags.li(ui.tags.a("Your Personal Webpage", href="YOUR_PERSONAL_WEBPAGE_LINK_HERE", target="_blank")),
                    ui.tags.li(ui.tags.a("Lab/Institution Page", href="YOUR_LAB_LINK_HERE", target="_blank")),
                ),
                ui.h4("Contact"),
                ui.p("For questions or collaborations, please reach out to [Your Name/Email]."),
                style="padding: 20px;"
            )
        ),
        title="Chemical Space Explorer 🧬",
        footer=ui.div(
            ui.hr(),
            ui.p(
                "Source: Bermúdez-Montaña, M., et al. (2025). China's rise in the chemical space and the decline of US influence. ",
                ui.tags.i("Placeholder citation details."), # Example of adding more details
                style="text-align: center; padding: 10px; font-size: 0.9em; color: #777;"
            )
        )
    )


    def server(input, output, session):
        # Reactive values
        selected_countries = reactive.Value([])
        
        # Cached reactive for country list
        @reactive.Calc
        def country_list():
            return load_country_list()
        
        # Optimized reactive for main data
        @reactive.Calc
        def filtered_data():
            # Only load full data when needed and cache results
            selected_tuple = tuple(sorted(selected_countries.get())) if selected_countries.get() else ()
            
            return cached_get_display_data(
                selected_isos_tuple=selected_tuple,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode=input.display_mode_input(),
                region_filter=input.region_filter()
            )

        @reactive.Effect
        @reactive.event(input.map_click_iso)
        def _handle_map_click():
            clicked_iso = input.map_click_iso()
            if clicked_iso:
                current_selection = list(selected_countries())
                if clicked_iso in current_selection:
                    current_selection.remove(clicked_iso)
                else:
                    current_selection.append(clicked_iso)
                selected_countries.set(current_selection)

        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])

        @reactive.Effect
        @reactive.event(input.region_filter)
        def _handle_region_change():
            """Optional: Clear selections when region filter changes"""
            # You can uncomment this if you want selections to clear when changing regions
            # selected_countries.set([])
            pass

        @output
        @render.text
        def selection_info():
            """Show enhanced selection info with region context"""
            count = len(selected_countries.get())
            current_region = input.region_filter()
            
            # Get available countries in current region
            all_countries = country_list()
            if current_region != "All":
                available_countries = all_countries[
                    all_countries['region'] == current_region
                ]
            else:
                available_countries = all_countries
            
            available_count = len(available_countries)
            
            if count == 0:
                return f"No countries selected (from {available_count} available in {current_region})"
            elif count == 1:
                return f"1 country selected (from {available_count} available in {current_region})"
            else:
                return f"{count} countries selected (from {available_count} available in {current_region})"

        @output
        @render.ui
        def map_output():
            """Render the interactive map with region filtering"""
            # Apply region filter to countries shown on map
            all_countries = country_list()
            current_region_filter = input.region_filter()
            
            if current_region_filter != "All":
                filtered_countries = all_countries[
                    all_countries['region'] == current_region_filter
                ]
            else:
                filtered_countries = all_countries
            
            m = create_folium_map(filtered_countries, selected_countries.get())
            return ui.HTML(m._repr_html_())
            
        @output  
        @render_widget
        def main_plot():
            """Main trends plot with validation"""
            current_mode = input.display_mode_input()
            selected = selected_countries.get()
            
            # Validate collaboration mode
            if current_mode == "find_collaborations" and len(selected) < 2:
                return create_empty_plot(
                    "🤝 Select at least 2 countries to find collaborations.\n"
                    "Click on countries in the map above."
                )
            
            data = filtered_data()
            if data.empty:
                if current_mode == "find_collaborations":
                    return create_empty_plot("No collaborations found for selected countries and filters.")
                else:
                    return create_empty_plot("Select countries from the map to view trends.")
                
            return create_trends_plot(data, selected, current_mode)
            
        @output
        @render_widget  
        def contribution_map():
            """Fixed contribution choropleth map"""
            try:
                # Load data for all countries in region
                df = pd.read_parquet("./data/data.parquet")
                countries_for_choropleth = country_list()
                
                current_region_filter = input.region_filter()
                if current_region_filter != "All":
                    countries_for_choropleth = countries_for_choropleth[
                        countries_for_choropleth['region'] == current_region_filter
                    ]
                
                isos_for_choropleth = countries_for_choropleth['iso2c'].unique().tolist()
                
                if not isos_for_choropleth:
                     return create_empty_plot(f"No countries found for region: {current_region_filter}")
        
                # Filter data for choropleth - use all available data, not just selected countries
                choropleth_data = get_display_data(
                    df=df,
                    selected_isos=isos_for_choropleth,  # Use all countries in region
                    year_range=input.years(),
                    chemical_category=input.chemical_category(),
                    display_mode="compare_individuals",
                    region_filter=current_region_filter,
                    country_list=countries_for_choropleth
                )
                
                if choropleth_data.empty:
                    return create_empty_plot("No data for global map with current selections")
                
                # Debug info - you can remove this later
                # print(f"Choropleth data shape: {choropleth_data.shape}")
                # print(f"Countries with data: {choropleth_data['iso2c'].nunique()}")
                # print(f"Value range: {choropleth_data['total_percentage'].min():.2f} - {choropleth_data['total_percentage'].max():.2f}")
                    
                return create_contribution_choropleth(choropleth_data)
                
            except Exception as e:
                print(f"Error in contribution_map: {str(e)}")
                return create_empty_plot(f"Error creating map: {str(e)}")
            
        @output
        @render.data_frame
        def summary_table():
            """Summary data table"""
            data = filtered_data()
            if data.empty:
                current_mode = input.display_mode_input()
                if current_mode == "find_collaborations" and len(selected_countries.get()) < 2:
                    message = "Select at least 2 countries to find collaborations."
                else:
                    message = "No data available for current selections."
                return pd.DataFrame({"Message": [message]})
                
            return create_summary_dataframe(data, input.display_mode_input())

        # Article plot outputs with lazy loading
        @output
        @render_widget
        def country_cs_plot():
            try:
                article_data = load_article_data()
                if article_data.empty:
                    return create_empty_plot("No article data available")
                
                df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
                if df_filtered.empty:
                    return create_empty_plot("No country participation data available")
                
                return create_article_plot(df_filtered, "Country participation in the growth of the CS")
            except Exception as e:
                return create_empty_plot(f"Error loading article data: {str(e)}")

        @output
        @render_widget
        def article_top_collabs_plot():
            try:
                df = pd.read_parquet("./data/data.parquet")
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
                    f"Top 10 {'Collaborations' if is_collab else 'Countries'}: {chem_filter} 'Chemicals'" 
                )
            except Exception as e:
                return create_empty_plot(f"Error: {str(e)}")

        @output
        @render_widget
        def article_gdp_plot():
            try:
                article_data = load_article_data()
                if article_data.empty:
                    return create_dummy_gdp_plot()
                
                gdp_data = article_data[article_data['source'] == "Annual growth rate of the GDP"]
                if gdp_data.empty:
                    return create_dummy_gdp_plot()
                    
                return create_gdp_plot(gdp_data)
            except Exception as e:
                return create_dummy_gdp_plot()

        @output
        @render_widget
        def article_researchers_plot():
            try:
                article_data = load_article_data()
                if article_data.empty:
                    return create_dummy_researchers_plot()
                
                researchers_data = article_data[article_data['source'] == "Number of Researchers"]
                if researchers_data.empty:
                    return create_dummy_researchers_plot()
                    
                return create_researchers_plot(researchers_data)
            except Exception as e:
                return create_dummy_researchers_plot()

        @output
        @render_widget
        def article_cs_expansion_plot():
            try:
                article_data = load_article_data()
                if article_data.empty:
                    return create_dummy_cs_expansion_plot()
                
                cs_data = article_data[article_data['source'] == "Expansion of the CS"]
                if cs_data.empty:
                    return create_dummy_cs_expansion_plot()
                    
                return create_cs_expansion_plot(cs_data)
            except Exception as e:
                return create_dummy_cs_expansion_plot()
        
    return App(app_ui, server)

# Cached helper functions
@lru_cache(maxsize=1)
def load_article_data():
    """Load and cache article data"""
    try:
        df = pd.read_parquet("./data/data.parquet")
        article_columns_map = {'source': 'source', 'year_x': 'year', 'country_x': 'country', 'percentage_x': 'value', 'cc': 'cc'}
        required_raw_cols = list(article_columns_map.keys())

        if all(col in df.columns for col in required_raw_cols):
            article_data_raw = df[required_raw_cols].copy()
            article_data = article_data_raw.rename(columns=article_columns_map)
            article_data = article_data.dropna(subset=['value', 'source'])
            article_data = article_data[article_data['source'] != ""]
            return article_data
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Helper functions
def create_folium_map(country_list: pd.DataFrame, selected_countries: List[str]) -> folium.Map:
    """Create interactive Folium map with improved region handling"""
    
    # Determine map center based on filtered countries
    if not country_list.empty:
        center_lat = country_list['lat'].mean()
        center_lng = country_list['lng'].mean()
        
        # Adjust zoom based on region spread
        lat_range = country_list['lat'].max() - country_list['lat'].min()
        lng_range = country_list['lng'].max() - country_list['lng'].min()
        
        # Determine appropriate zoom level
        if lat_range > 60 or lng_range > 120:  # Global view
            zoom_start = 2
        elif lat_range > 30 or lng_range > 60:  # Continental view
            zoom_start = 3
        elif lat_range > 15 or lng_range > 30:  # Regional view
            zoom_start = 4
        else:  # Local view
            zoom_start = 5
    else:
        center_lat, center_lng, zoom_start = 30, 10, 2
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start)
    
    # Add region info to map title
    if len(country_list) > 0:
        regions_in_map = country_list['region'].unique()
        if len(regions_in_map) == 1 and regions_in_map[0] != 'Other':
            map_title = f"Region: {regions_in_map[0]} ({len(country_list)} countries)"
        else:
            map_title = f"Showing {len(country_list)} countries"
        
        # Add a subtle title overlay
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 30px; 
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 2px solid rgba(0,0,0,0.2);
                    border-radius: 5px;
                    z-index:9999; 
                    font-size:12px;
                    font-weight: bold;
                    text-align: center;
                    padding: 5px;">
            {map_title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
    
    # Load world geometries for better country shapes
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
        
        # Add countries to map
        for _, country_row in country_list.iterrows():
            iso = country_row['iso2c']
            country_name = country_row['country']
            region = country_row.get('region', 'Unknown')
            
            # Enhanced color scheme
            if iso in selected_countries:
                color = country_row['cc']  # Use country color when selected
                fill_opacity = 0.8
                stroke_weight = 2
            else:
                color = "#a1b4af",  # Default color for unselected countries
                fill_opacity = 0.5
                stroke_weight = 1
            
            country_geo = world[world[iso_column] == iso]
            
            if not country_geo.empty:
                # Enhanced popup with region info
                popup_html = f"""
                <div style="min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">
                        {country_name} ({iso})
                    </h4>
                    <p style="margin: 5px 0; color: #7f8c8d;">
                        <strong>Region:</strong> {region}
                    </p>
                    <p style="margin: 5px 0; color: #7f8c8d;">
                        <strong>Status:</strong> 
                        {'Selected' if iso in selected_countries else 'Available'}
                    </p>
                    <button onclick="
                        if (window.parent && window.parent.Shiny) {{
                            window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }} else if (window.Shiny) {{
                            window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }}
                    " style="
                        padding: 8px 16px; 
                        margin: 10px 0 5px 0; 
                        cursor: pointer;
                        background-color: {'#e74c3c' if iso in selected_countries else '#3498db'};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                        width: 100%;
                    ">
                        {'🗑️ Deselect' if iso in selected_countries else '✅ Select'}
                    </button>
                </div>
                """
                
                folium.GeoJson(
                    country_geo.iloc[0].geometry,
                    style_function=lambda x, color=color, fill_opacity=fill_opacity, weight=stroke_weight: {
                        'fillColor': color,
                        'color': 'white',
                        'weight': weight,
                        'fillOpacity': fill_opacity,
                        'dashArray': '0' if iso in selected_countries else '5, 5'
                    },
                    tooltip=folium.Tooltip(
                        f"<b>{country_name}</b><br>Region: {region}<br>Click to {'deselect' if iso in selected_countries else 'select'}",
                        sticky=True
                    ),
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(m)
            else:
                # Enhanced fallback markers
                popup_html = f"""
                <div style="min-width: 180px;">
                    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">
                        {country_name} ({iso})
                    </h4>
                    <p style="margin: 5px 0; color: #7f8c8d;">
                        <strong>Region:</strong> {region}
                    </p>
                    <button onclick="
                        if (window.parent && window.parent.Shiny) {{
                            window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }} else if (window.Shiny) {{
                            window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                        }}
                    " style="
                        padding: 8px 16px; 
                        margin: 10px 0 5px 0; 
                        cursor: pointer;
                        background-color: {'#e74c3c' if iso in selected_countries else '#3498db'};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                        width: 100%;
                    ">
                        {'🗑️ Deselect' if iso in selected_countries else '✅ Select'}
                    </button>
                </div>
                """
                
                folium.CircleMarker(
                    location=[country_row['lat'], country_row['lng']],
                    radius=8 if iso in selected_countries else 5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    weight=stroke_weight,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=folium.Tooltip(
                        f"<b>{country_name}</b><br>Region: {region}",
                        sticky=True
                    )
                ).add_to(m)
                
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        # Enhanced fallback to markers with region info
        for _, country in country_list.iterrows():
            iso = country['iso2c']
            country_name = country['country']
            region = country.get('region', 'Unknown')
            
            if iso in selected_countries:
                color = country['cc']
                radius = 8
                fill_opacity = 0.8
            else:
                color = 'lightblue'
                radius = 5
                fill_opacity = 0.5
            
            popup_html = f"""
            <div style="min-width: 180px;">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">
                    {country_name} ({iso})
                </h4>
                <p style="margin: 5px 0; color: #7f8c8d;">
                    <strong>Region:</strong> {region}
                </p>
                <button onclick="
                    if (window.parent && window.parent.Shiny) {{
                        window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }} else if (window.Shiny) {{
                        window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }}
                " style="
                    padding: 8px 16px; 
                    margin: 10px 0 5px 0; 
                    cursor: pointer;
                    background-color: {'#e74c3c' if iso in selected_countries else '#3498db'};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    width: 100%;
                ">
                    {'🗑️ Deselect' if iso in selected_countries else '✅ Select'}
                </button>
            </div>
            """
            
            folium.CircleMarker(
                location=[country['lat'], country['lng']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=folium.Tooltip(
                    f"<b>{country_name}</b><br>Region: {region}",
                    sticky=True
                )
            ).add_to(m)
    
    return m

def create_trends_plot(data: pd.DataFrame, selected_countries: List[str], mode: str):
    """Create trends plot using Plotly"""
    fig = go.Figure()
    
    value_column = 'total_percentage' # Expect this from get_display_data
    
    if value_column not in data.columns or data.empty:
        # No valid column found, or data is empty
        fig.add_annotation(
            text=f"Error: Missing '{value_column}' column or no data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white'
        )
        return fig
    
    plot_title = "Chemical Space Contribution Trends"

    if mode == "compare_individuals":
        # Individual country trends
        for country_name in data['plot_group'].unique(): # plot_group is country name for individuals
            country_data = data[data['plot_group'] == country_name]
            color = country_data['plot_color'].iloc[0] if 'plot_color' in country_data.columns and not country_data.empty else None
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data[value_column],
                mode='lines+markers',
                name=str(country_name),
                line=dict(color=color if color else None),
                hovertemplate=(
                    f"<b>{country_name}</b><br>" +
                    "Year: %{x}<br>" +
                    f"{value_column}: %{{y:.2f}}%<extra></extra>"
                )
            ))
        plot_title = "Individual Country Contribution Trends"
    elif mode == "find_collaborations":
        # Collaboration trends
        collab_type_colors = {
            "Bilateral": "rgba(255, 127, 14, 0.9)",
            "Trilateral": "rgba(44, 160, 44, 0.9)",
            "4-country": "rgba(214, 39, 40, 0.9)",
            "5-country+": "rgba(148, 103, 189, 0.9)",
            "Unknown": "rgba(127, 127, 127, 0.9)"
        }

        for collab_id in data['plot_group'].unique(): # plot_group is collab ID string
            collab_data = data[data['plot_group'] == collab_id]
            if collab_data.empty: continue

            collab_type = collab_data['plot_color_group'].iloc[0] if 'plot_color_group' in collab_data.columns else "Unknown"
            
            fig.add_trace(go.Scatter(
                x=collab_data['year'],
                y=collab_data[value_column],
                mode='lines+markers',
                name=str(collab_id),
                line=dict(color=collab_type_colors.get(str(collab_type), collab_type_colors["Unknown"])),
                hovertemplate=(
                    f"<b>Collaboration: {collab_id}</b> ({collab_type})<br>" +
                    "Year: %{x}<br>" +
                    f"{value_column}: %{{y:.2f}}%<extra></extra>"
                )
            ))
        plot_title = "Collaboration Trends"
    
    fig.update_layout(
        title=plot_title,
        xaxis_title="Year",
        yaxis_title="% of New Substances",
        hovermode='closest',
        template='plotly_white'
    )


    
    return fig

def create_contribution_choropleth(data: pd.DataFrame):
    """Create world choropleth map with proper color scaling"""
    value_column = 'total_percentage'

    if value_column not in data.columns or data.empty:
        return create_empty_plot("No data available for choropleth")
    
    # Calculate average percentage per country
    avg_data = (
        data.groupby(['iso3c', 'country'], as_index=False)
        .agg({
            'total_percentage': 'mean',
            'region': 'first'
        })
        .round(2)
    )
    
    if avg_data.empty:
        return create_empty_plot("No aggregated data for choropleth")

    # Ensure we have valid data for plotting
    avg_data = avg_data.dropna(subset=['total_percentage'])
    
    if avg_data.empty:
        return create_empty_plot("No valid data after removing NaN values")

    # Create choropleth with explicit color range
    min_val = avg_data['total_percentage'].min()
    max_val = avg_data['total_percentage'].max()
    
    # Handle case where all values are the same
    if min_val == max_val:
        color_range = [max(0, min_val - 0.1), min_val + 0.1]
    else:
        color_range = [min_val, max_val]

    fig = go.Figure(data=go.Choropleth(
        locations=avg_data['iso3c'],
        z=avg_data['total_percentage'],
        locationmode='ISO-3',
        # colorscale='Viridis',
        colorscale=[[0, 'rgb(10, 49, 97)'], [1, 'rgb(238, 28, 37)']],
        reversescale=False,
        zmid=None,  # Let plotly handle the midpoint
        zmin=color_range[0],
        zmax=color_range[1],
        # legendwidth = 20,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{location})<br>" +
            "Avg Contribution: %{z:.2f}%<br>" +
            "Region: %{customdata[1]}<extra></extra>"
        ),
        customdata=avg_data[['country', 'region']].values,
        colorbar=dict(
            # title="Avg. Contribution (%)",
            orientation = "h",
            tickangle =0,
            tickformat = ".2f",
            ticklabelstep = 2,
            ticksuffix = "%",
            # titleside="right",
            tickmode="linear",
            # xanchor ='right',
            # yanchor ='bottom',
            tick0=color_range[0],
            dtick=max(0.5, (color_range[1] - color_range[0]) / 10)
        ),
        showscale=True
    ))
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            projection_type='robinson',
            bgcolor='rgba(0,0,0,0)',
            showlakes=True,
            lakecolor='rgba(127,205,255,0.1)'
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        modebar_remove=['zoom', 'pan', 'lasso', 'select']
    )
    
    return fig

def create_summary_dataframe(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Create summary statistics table"""
    value_column = 'total_percentage' # Expect this from get_display_data

    if value_column not in data.columns or data.empty:
        return pd.DataFrame({'Error': [f"Missing '{value_column}' column or no data for summary."]})
    
    if mode == "find_collaborations":
        required_cols = ['plot_group', 'chemical', 'collab_type', value_column]
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            return pd.DataFrame({'Error': [f"Summary data for collaborations missing: {missing}"]})
        
        summary = (
            data.groupby(['plot_group', 'chemical', 'collab_type'])
            .agg(
                avg_percentage=(value_column, 'mean'),
                max_percentage=(value_column, 'max'),
                years_present=('year', 'nunique') # Count distinct years
            )
            .round(2)
            .reset_index()
        )
        summary.columns = ['Collaboration', 'Chemical', 'Type', 'Avg %', 'Max %', 'Years Present']
    elif mode == "compare_individuals":
        required_cols = ['country', 'iso2c', 'chemical', value_column]
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            return pd.DataFrame({'Error': [f"Summary data for individuals missing: {missing}"]})

        summary = (
            data.groupby(['country', 'iso2c', 'chemical'])
            .agg(
                avg_percentage=(value_column, 'mean'),
                max_percentage=(value_column, 'max'),
                years_present=('year', 'nunique') # Count distinct years
            )
            .round(2)
            .reset_index()
        )
        summary.columns = ['Country', 'ISO', 'Chemical', 'Avg %', 'Max %', 'Years Present']
    else:
        return pd.DataFrame({'Message': [f"Summary not available for display mode: {mode}"]})
        
    return summary


def create_article_plot(data: pd.DataFrame, title: str):
    """Create article plots"""
    fig = go.Figure()
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        color = country_data['cc'].iloc[0] if 'cc' in country_data.columns and not country_data.empty else None
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['value'],
            mode='lines+markers',
            name=country,
            line=dict(color=color, width=1) if color else dict(width=1),
            marker=dict(size=country_data['value'].abs().clip(upper=15) + 2, color=color if color else 'red')
        ))
    
    fig.update_layout(
        title=f"{title}",
        # xaxis_title="Year",
        yaxis_title="% of New Substances",
        yaxis = dict(
            ticksuffix='%'
        ),
        template='plotly_white',
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
    )
    
    return fig

def create_top_trends_plot(data: pd.DataFrame, title: str):
    """Create top contributors/collaborations plot"""
    fig = go.Figure()
    
    # Calculate the average percentage for each entity to sort the legend
    avg_percentages = data.groupby('country')['percentage'].mean().sort_values(ascending=True)
    
    # Plot entities in order of their average percentage (highest first)
    for entity in avg_percentages.index:
        entity_data = data[data['country'] == entity]
        # Ensure entity data is sorted by year for proper line drawing
        entity_data = entity_data.sort_values('year')
        
        avg_value = avg_percentages[entity]
        
        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data['percentage'],
            mode='lines+markers',
            name=f"{entity} ({avg_value:.2f}%)",  # Include avg in legend
            line=dict(width=1.5),
            marker=dict(size=entity_data['percentage'].abs().clip(upper=15) + 2, color=entity_data['cc'].iloc[0] if 'cc' in entity_data.columns and not entity_data.empty else 'red'),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Year: %{x}<br>" +
                "Contribution: %{y:.2f}%<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year", 
        yaxis_title="% of New Substances",
        yaxis = dict(
            ticksuffix='%'
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h", 
            y=-0.2,
            traceorder="reversed"  # Display in same order as traces
        ),
        hovermode='closest',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
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
            marker=dict(size=country_data['value'].abs().clip(upper=15) + 2, color=country_data['cc'].iloc[0] if 'cc' in country_data.columns and not country_data.empty else 'red')
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
        yaxis_title="GDP Growth Rate (%)",
        template='plotly_white',
        yaxis = dict(
            ticksuffix='%'
        ),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
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
            marker=dict(size=scaled_values.abs().clip(lower=1, upper=15) + 2, color=country_data['cc'].iloc[0] if 'cc' in country_data.columns and not country_data.empty else 'red')
        ))
    
    fig.update_layout(
        yaxis_title="Number of Researchers (Millions)",
        yaxis = dict(
            ticksuffix= 'M'
            # tickformat = ',.0f'  # Format as whole numbers
        ),
        template='plotly_white',
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
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
        yaxis_title="Number of New Substances",
        template='plotly_white',
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
    )
    
    return fig

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run()