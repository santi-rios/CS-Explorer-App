import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd

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
from utils.functions import get_display_data, create_folium_map, create_trends_plot, create_contribution_choropleth, create_summary_dataframe, create_article_plot, create_top_trends_plot, create_empty_plot, create_gdp_plot, create_researchers_plot, create_cs_expansion_plot, create_china_us_dual_axis_plot
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
        chemical_categories = sorted([chem for chem in unique_chemicals if chem and str(chem).strip()])
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
                    "Dashboard", 
                    ui.div(
                        # --- Introductory Section ---
                        ui.h3("China's Rise in the Chemical Space (CS)", style="text-align: center; margin-bottom: 20px; margin-top: 20px;"),
                        ui.p(
                            "Explore the rise of China's influence in the chemical space and the decline of US dominance through interactive visualizations.",
                            style="text-align: center; margin-bottom: 15px; font-size: 1.1em; color: #555;"
                        ),
                        ui.p(
                            ui.tags.span("Source: "),
                            ui.tags.a("Bermúdez-Montaña, M., et al. (2025). China's rise in the chemical space and the decline of US influence.", 
                                  href="https://chemrxiv.org/engage/chemrxiv/article-details/67920ada6dde43c908f688f6",
                                  target="_blank"),
                            style="text-align: center; margin-bottom: 30px; font-size: 0.9em; color: #666; font-style: italic;"
                        ),
                        ui.div(
                            ui.popover(
                                ui.input_action_button(
                                    "btn", "📌 Quick Summary", 
                                    class_="btn-primary",
                                    style="margin: 0 auto; display: block; padding: 8px 16px; font-weight: bold;"
                                ),
                                "From 1996 to 2022, the landscape of chemical discovery shifted dramatically. China surged to dominate the report of new substances, primarily through domestic research.",
                                " Conversely, the United States' solo contributions declined, becoming more reliant on international collaborations, particularly with China.",
                                id="btn_popover",
                            ),
                            style="text-align: center; margin-bottom: 30px;"
                        ),
                    ), # End of Introductory Section
                    
                    # --- Tabbed Main Content ---
                    ui.navset_card_tab(
                        ui.nav_panel(
                            "📊 Key Article Figures",
                            # Content from former "Section 1: Article Highlights" (Key Article Figures)
                            ui.row(
                                ui.column(12, 
                                          ui.div(
                                              # The H4 title is now part of the tab name
                                              ui.navset_card_tab(
                                                  ui.nav_panel(
                                                      "🏆 Main Countries",
                                                      ui.card(
                                                          ui.p("Main countries contributing to the chemical space (CS) from 1996 to 2022.",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          output_widget("country_cs_plot"),
                                                          ui.p("Hide a Country by clicking on its name in the legend. Double-click to isolate it.",
                                                                style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "🥼China-US",
                                                      ui.card(
                                                          ui.p("Percentage of new substances with participation of China or the US resulting from China-US collaboration (right axis). Left axis show the percentage of new substances with participation of country (China or US) that are reported in papers with no international collaboration.",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          output_widget("china_us_plot"),
                                                          ui.p("Hide a Country by clicking on its name in the top legend. Double-click to isolate it.",
                                                                style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "💰 GDP and growth",
                                                      ui.card(
                                                          ui.p("Percentage of the annual growth rate of the gross domestic product (GDP) per capita",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          output_widget("article_gdp_plot")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "👥 Number of Researchers",
                                                      ui.card(
                                                          ui.p("Number of researchers in research and development activities",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          output_widget("article_researchers_plot")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "🧪 Chemical Space Expansion",
                                                      ui.card(
                                                          ui.p("Recent expansion of the CS and of three of its subspaces",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          output_widget("article_cs_expansion_plot")
                                                      )
                                                  ),
                                              ),
                                              style="border: 1px solid #dee2e6; border-radius: .25rem; padding: 15px; background-color: #f8f9fa; margin-top: 20px;" # Added margin-top
                                          )
                                )
                            )
                        ),
                        ui.nav_panel(
                            "🗺️ Interactive Explorer",
                            # Content from former "Section 2: Explore Chemical Space"
                            ui.h3("Explore Chemical Space Interactively", style="text-align: center; margin-bottom: 20px; margin-top: 20px;"),
                            ui.p(
                                "Dive into the interactive map and plots below. Use the filters to dynamically update the visualizations.",
                                style="text-align: center; margin-bottom: 30px; font-size: 1.1em; color: #555;"
                            ),
                            # --- New Filter Section ---
                            ui.panel_well(
                                ui.row(
                                    ui.column(4,
                                        ui.input_select(
                                            "region_filter", "🌍 Filter by Region:",
                                            choices=initial_data['regions'], selected="All"
                                        ),
                                    ),
                                class_="bg-light border rounded p-3 mb-4"
                            )
                            ),
                            
                            # --- Map and Selection Section ---
                            ui.card(
                                ui.card_header("Interactive Map & Selection"),
                                ui.row(
                                    ui.column(9,
                                        ui.output_ui("map_output")
                                    ),
                                    ui.column(3,
                                        ui.div(
                                            ui.h5("Selection Controls"),
                                            ui.input_action_button(
                                                "clear_selection", "🗑️ Clear Selection",
                                                class_="btn-outline-danger w-100 mb-3"
                                            ),
                                            ui.div(ui.output_text("selection_info"), class_="text-muted small")
                                        )
                                    )
                                )
                            ),

                            # --- New Filter Section ---
                            ui.panel_well(
                                ui.row(
                                    ui.column(4,
                                        ui.input_select(
                                            "chemical_category", "🧪 Chemical Space:",
                                            choices=initial_data['chemical_categories'], selected="All"
                                        ),
                                    ),
                                    ui.column(4,
                                        ui.input_radio_buttons(
                                            "display_mode_input", "📊 Display Mode:",
                                            choices={
                                                "compare_individuals": "Individual Countries",
                                                "find_collaborations": "Find Collaborations"
                                            },
                                            selected="compare_individuals"
                                        ),
                                    ),
                                ),
                                ui.row(
                                    ui.column(12,
                                        ui.input_slider(
                                            "years", "📅 Year Range:",
                                            min=initial_data['min_year'], max=initial_data['max_year'],
                                            value=[initial_data['min_year'], initial_data['max_year']],
                                            step=1, sep=""
                                        ),
                                    )
                                ),
                                class_="bg-light border rounded p-3 mb-4"
                            ),

                            # --- Plots Section (unchanged) ---
                            ui.navset_card_tab( 
                                ui.nav_panel("📈 Trends", output_widget("main_plot")),
                                ui.nav_panel(
                                    "🌎 Global Snapshot",
                                    ui.p(
                                        "This plot shows the global or regional snapshot of the chemical space contributions by countries, highlighting the average contributions based on the year range selected (defaults to 1996-2022).",
                                        style="font-size: 0.9em; color: #666; text-align: center;"
                                    ), 
                                    output_widget("contribution_map")
                                    ),
                                ui.nav_panel("📋 Data Table", ui.output_data_frame("summary_table"))
                            )
                        ),
                        ui.nav_panel(
                            "🤝 Top Trends",
                            # Content from former "Top Collaborations plot" section
                            ui.row(
                                ui.column(12, 
                                          ui.panel_well(
                                              ui.h4("Explore Top Collaborations and Countries in the CS", style="margin-top: 0; text-align: center;"),
                                              ui.row( 
                                                  ui.column(6, ui.input_select("top_collabs_chem_filter", "Chemical Space Category:", choices=initial_data['chemical_categories'], selected="All")),
                                                  ui.column(6, ui.input_radio_buttons("top_data_type_filter", "Show Top:", choices={"collabs": "Collaborations", "individuals": "Countries"}, selected="collabs")),
                                                  ui.p("When viewing collaborations, Blue Colors represent Countries Collaborations with the US",
                                                       style="font-size: 0.9em; color: #666; text-align: center;")
                                              ),
                                              output_widget("article_top_collabs_plot"),
                                              ui.p("Legend is sorted by the average contribution (shown in parenthesis) of the collaboration/country to the CS between 1996 to 2022.",
                                                  style="font-size: 0.9em; color: #666; text-align: center;"),
                                              class_="border rounded p-3 bg-light", 
                                              style="margin-top: 20px; margin-bottom: 20px;"
                                          )
                                )
                            )
                        ),
                    ) # End of Tabbed Main Content
                    # The ui.hr separator is removed as tabs provide separation.
                ),
        # --- Other nav panels remain for auxiliary content ---
        # ui.nav_panel(
        #     "📖 Original Article",
        #     ui.tags.iframe(
        #         src="original_article.pdf", # www/ directory
        #         style="width: 100%; height: 80vh; border: none;"
        #     )
        # ),
        ui.nav_panel(
            "🔗 Useful Links",
            ui.div(
                ui.h4("Project & Resources"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.a("GitHub Repository", href="https://github.com/santi-rios/CS-Explorer-App", target="_blank")),
                    # ui.tags.li(ui.tags.a("Your Personal Webpage", href="YOUR_PERSONAL_WEBPAGE_LINK_HERE", target="_blank")),
                    ui.tags.li(ui.tags.a("Site Builded by Santiago G-Rios", href="https://santi-rios.github.io/", target="_blank")),
                ),
                # ui.h4("Contact"),
                # ui.p("For questions or collaborations, please reach out to [Your Name/Email]."),
                style="padding: 20px;"
            )
        ),
        title="Chemical Space Explorer 🧪",
        footer=ui.div(
            ui.hr(),
            ui.p(
                "Source: Bermúdez-Montaña, M., et al. (2025). China's rise in the chemical space and the decline of US influence. ",
                ui.tags.i("ChemRxiv link: "),
                ui.tags.a("https://chemrxiv.org/engage/chemrxiv/article-details/67920ada6dde43c908f688f6", 
                          href="https://chemrxiv.org/engage/chemrxiv/article-details/67920ada6dde43c908f688f6", 
                          target="_blank"),
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
            current_mode = input.display_mode_input()
            
            # For collaboration mode, we need to handle selected countries differently
            # In collaboration mode, we want to find collaborations involving the selected countries
            # In individual mode, we want to filter to show only the selected countries
            if current_mode == "find_collaborations":
                # For collaborations, pass selected countries to find collaborations between them
                # The get_display_data function should handle this logic
                return cached_get_display_data(
                    selected_isos_tuple=selected_tuple,
                    year_range=tuple(input.years()),
                    chemical_category=input.chemical_category(),
                    display_mode=current_mode,
                    region_filter=input.region_filter()
                )
            else:
                # For individual countries, pass selected countries normally
                return cached_get_display_data(
                    selected_isos_tuple=selected_tuple,
                    year_range=tuple(input.years()),
                    chemical_category=input.chemical_category(),
                    display_mode=current_mode,
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

        @output
        @render_widget
        def china_us_plot():
            try:
                article_data = load_article_data()
                if article_data.empty:
                    return create_dummy_cs_expansion_plot() # Or create_empty_plot("No data for China-US plot")
                
                cs_data = article_data[article_data['source'] == "China-US in the CS"]
                if cs_data.empty:
                    return create_dummy_cs_expansion_plot() # Or create_empty_plot("No 'China-US in the CS' data found")
                    
                return create_china_us_dual_axis_plot(cs_data) # Use the new function
            except Exception as e:
                # Consider logging the error e
                return create_dummy_cs_expansion_plot() # Or create_empty_plot(f"Error: {str(e)}")
        
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

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run()