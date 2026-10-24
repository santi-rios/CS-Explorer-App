import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as plotly_offline
from plotly.subplots import make_subplots
import geopandas as gpd

from folium import plugins
from shiny import App, ui, render, reactive
from shiny.types import FileInfo
import asyncio
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from shinywidgets import render_widget, output_widget
from folium.plugins import Draw
from utils.functions import (
    get_display_data, create_folium_map, create_trends_plot,
    create_contribution_choropleth, 
    create_contribution_map_plot,
    create_summary_dataframe, create_article_plot, create_top_trends_plot,
    create_empty_plot, create_gdp_plot, create_researchers_plot,
    create_cs_expansion_plot, create_china_us_dual_axis_plot,
    get_raw_dataframe, get_world_geometries
)
import functools
from functools import lru_cache


# Added .resolve() to guarantee an absolute path (e.g., /home/santi/projects/...)
www_dir = Path(__file__).resolve().parent / "www"


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
    df = get_raw_dataframe()
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
    df = get_raw_dataframe()
    return (
        df[df['is_collab'] == False]
        .drop_duplicates(subset=['country', 'iso2c', 'lat', 'lng', 'cc', 'region'])
        .dropna(subset=['country', 'iso2c'])
        .query("country != '' and iso2c != ''")
        .fillna({'region': 'Other'})
        .sort_values('country')
        .reset_index(drop=True)
    )

def figure_to_html(fig):
    """Render Plotly figures as embedded HTML so trace data is preserved."""
    return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False, config={"responsive": True}))



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
        
        countries_df = load_country_list()
        country_choices = {
            row['iso2c']: f"{row['country']} ({row['iso2c']})"
            for _, row in countries_df.iterrows()
        }

        initial_data = {
            'chemical_categories': chemical_categories,
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year,
            'country_choices': country_choices
        }
        
    except Exception as e:
        print(f"Error loading initial data for UI: {e}")
        initial_data = {
            'chemical_categories': ["All"],
            'regions': ["All"],
            'min_year': 1996,
            'max_year': 2022,
            'country_choices': initial_data
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
                                  href="https://doi.org/10.1162/QSS.a.471",
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
                                                          ui.output_ui("country_cs_plot"),
                                                          ui.p("Hide a Country by clicking on its name in the legend. Double-click to isolate it.",
                                                                style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "🥼China-US",
                                                      ui.card(
                                                          ui.p("Percentage of new substances with participation of China or the US resulting from China-US collaboration (right axis). Left axis show the percentage of new substances with participation of country (China or US) that are reported in papers with no international collaboration.",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          ui.output_ui("china_us_plot"),
                                                          ui.p("Hide a Country by clicking on its name in the top legend. Double-click to isolate it.",
                                                                style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "💰 GDP and growth",
                                                      ui.card(
                                                          ui.p("Percentage of the annual growth rate of the gross domestic product (GDP) per capita",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          ui.output_ui("article_gdp_plot")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "👥 Number of Researchers",
                                                      ui.card(
                                                          ui.p("Number of researchers in research and development activities",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          ui.output_ui("article_researchers_plot")
                                                      )
                                                  ),
                                                  ui.nav_panel(
                                                      "🧪 Chemical Space Expansion",
                                                      ui.card(
                                                          ui.p("Recent expansion of the CS and of three of its subspaces",
                                                                style="margin-bottom: 10px; font-size: 0.9em; color: #666; text-align: center;"),
                                                          ui.output_ui("article_cs_expansion_plot")
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
                                "Dive into the interactive map and plots to explore the CS across 155 countries from 1996 to 2022. You can select more than one country to find the collaborations in the literature between the selected countries.",
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
                                ui.row(
                                    ui.column(4,
                                        ui.input_radio_buttons(
                                            "chemical_category", "🧪 Select Chemical Subspace:",
                                            choices={
                                                "All": "All Substances (Total CS)",
                                                "Organic": "Organic (Carbon-based)",
                                                "Organometallic": "Organometallic (Carbon + Metals)",
                                                "Rare-Earths": "Rare-earth Elements (REE)"
                                            },
                                            selected="All",
                                            inline=True
                                        ),
                                    ),
                                ),
                                class_="bg-light border border-1 rounded p-1 mb-1"
                            )
                            ),
                            
                            # --- Map and Selection Section ---
# --- Map and Selection Section ---
                    ui.card(
                        ui.card_header("Interactive Map & Selection"),
                        ui.row(
                            ui.column(8,
                                ui.output_ui("map_output")
                            ),
                            ui.column(4,
                                ui.div(
                                    ui.input_selectize(
                                        "country_picker",
                                        "🔍 Quick Search & Select Countries:",
                                        choices=initial_data['country_choices'],
                                        selected=[],
                                        multiple=True,
                                        remove_button=True,
                                        options={"placeholder": "Type country name or ISO..."}
                                    ),
                                    ui.input_action_button(
                                        "clear_selection", "🗑️ Reset Selection",
                                        class_="btn-outline-danger w-100 mt-2 mb-3"
                                    ),
                                    ui.div(ui.output_text("selection_info"), class_="text-muted small")
                                )
                            )
                        ),
                        class_="border rounded p-1 mb-1"
                    ),
                            # --- New Filter Section ---
                            ui.panel_well(
                                ui.row(
                                    ui.column(12,
                                        ui.input_slider(
                                            "years", "📅 Year Range:",
                                            min=initial_data['min_year'], max=initial_data['max_year'],
                                            value=[initial_data['min_year'], initial_data['max_year']],
                                            step=1, sep=""
                                        ),
                                    ui.br(),
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
                                    )
                                ),
                                class_="bg-light border border-1 rounded p-1 mb-1"
                            ),

# --- Plots Section ---
                    ui.navset_card_tab( 
                        ui.nav_panel("📈 Trends", ui.output_ui("main_plot")),
                        ui.nav_panel(
                            "🌎 Global Snapshot",
                            ui.div(
                                ui.input_radio_buttons(
                                    "snapshot_mode",
                                    "🗺️ Map View Mode:",
                                    choices={
                                        "static": "Period Average (Static Map)",
                                        "animated": "Yearly Evolution (Animated)"
                                    },
                                    selected="static",
                                    inline=True
                                ),
                                style="display: flex; justify-content: center; margin-bottom: 10px;"
                            ),
                            ui.p(
                                "This plot shows the global or regional snapshot of the chemical space contributions by countries, highlighting either the average contribution over the selected year range or the year-by-year animated evolution.",
                                style="font-size: 0.9em; color: #666; text-align: center;"
                            ), 
                            ui.output_ui("contribution_map")
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
                                              ui.output_ui("article_top_collabs_plot"),
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
        ui.nav_panel(
            "📖 Original Article",
            ui.tags.iframe(
                src="original_final.pdf", # www/ directory
                style="width: 100%; height: 80vh; border: none;"
            )
        ),
        ui.nav_panel(
            "🔗 Useful Links",
            ui.div(
                ui.h4("Project & Resources"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.a("GitHub Repository", href="https://github.com/santi-rios/CS-Explorer-App", target="_blank")),
                    # ui.tags.li(ui.tags.a("Your Personal Webpage", href="YOUR_PERSONAL_WEBPAGE_LINK_HERE", target="_blank")),
                    ui.tags.li(ui.tags.a("Site Built by Santiago G-Rios", href="https://santi-rios.github.io/", target="_blank")),
                ),
                # ui.h4("Contact"),
                # ui.p("For questions or collaborations, please reach out to [Your Name/Email]."),
                style="padding: 20px;"
            )
        ),
        header=ui.tags.script(plotly_offline.get_plotlyjs()),
        title="Chemical Space Explorer 🧪",
        footer=ui.div(
            ui.hr(),
            ui.p(
                "Source: Bermúdez-Montaña, M (Bermudez-Montana, Marisol) ; Garcia-Chung, A (Garcia-Chung, Angel) ; Stadler, PF (Stadler, Peter F.) ; Jost, J (Jost, Jurgen) ; Restrepo, G (Restrepo, Guillermo), (2025). China's rise in the chemical space and the decline of US influence. ",
                ui.tags.i("Published in Quantitative Science Studies (MIT Press): "),
                ui.tags.a("https://doi.org/10.1162/QSS.a.471", 
                          href="https://direct.mit.edu/qss/article/doi/10.1162/QSS.a.471/135818/Global-shifts-in-the-geopolitics-of-chemical", 
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
        @render.ui
        def main_plot():
            """Main trends plot with validation"""
            current_mode = input.display_mode_input()
            selected = selected_countries.get()
            
            # Validate collaboration mode
            if current_mode == "find_collaborations" and len(selected) < 2:
                return figure_to_html(create_empty_plot(
                    "🤝 Select at least 2 countries to find collaborations.\n"
                    "Click on countries in the map above."
                ))
            
            data = filtered_data()
            if data.empty:
                if current_mode == "find_collaborations":
                    return figure_to_html(create_empty_plot("No collaborations found for selected countries and filters."))
                else:
                    return figure_to_html(create_empty_plot("Select countries from the map to view trends."))

                return figure_to_html(create_trends_plot(data, selected, current_mode))
            
# 1. Sync dropdown changes -> reactive selected_countries
        @reactive.Effect
        @reactive.event(input.country_picker, ignore_none=False, ignore_init=True)
        def _sync_picker_to_selected():
            picker_isos = list(input.country_picker() or [])
            if picker_isos != selected_countries.get():
                selected_countries.set(picker_isos)

        # 2. Sync Folium map clicks -> reactive selected_countries AND dropdown
        @reactive.Effect
        @reactive.event(input.map_click_iso)
        def _handle_map_click():
            clicked_iso = input.map_click_iso()
            if clicked_iso:
                current_selection = list(selected_countries.get())
                if clicked_iso in current_selection:
                    current_selection.remove(clicked_iso)
                else:
                    current_selection.append(clicked_iso)
                selected_countries.set(current_selection)
                ui.update_selectize("country_picker", selected=current_selection)

        # 3. Sync Reset button -> clear both reactive value and dropdown
        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])
            ui.update_selectize("country_picker", selected=[])

        # 4. Filter dropdown choices when Region Filter changes
        @reactive.Effect
        @reactive.event(input.region_filter, ignore_init=True)
        def _handle_region_change():
            all_countries = country_list()
            current_region = input.region_filter()
            if current_region != "All":
                region_df = all_countries[all_countries['region'] == current_region]
            else:
                region_df = all_countries

            new_choices = {
                row['iso2c']: f"{row['country']} ({row['iso2c']})"
                for _, row in region_df.iterrows()
            }
            valid_selected = [iso for iso in selected_countries.get() if iso in new_choices]
            selected_countries.set(valid_selected)
            ui.update_selectize("country_picker", choices=new_choices, selected=valid_selected)

        # 5. Updated Global Snapshot output supporting Static vs. Animated modes
        @output
        @render.ui 
        def contribution_map():
            """Render either the static period-average map or animated yearly choropleth"""
            try:
                countries_for_choropleth = country_list()
                current_region_filter = input.region_filter()
                if current_region_filter != "All":
                    countries_for_choropleth = countries_for_choropleth[
                        countries_for_choropleth['region'] == current_region_filter
                    ]

                isos_for_choropleth = tuple(sorted(countries_for_choropleth['iso2c'].unique().tolist()))
                if not isos_for_choropleth:
                    return figure_to_html(create_empty_plot(f"No countries found for region: {current_region_filter}"))

                # Uses the cached function instead of reading data.parquet from disk every time
                choropleth_data = cached_get_display_data(
                    selected_isos_tuple=isos_for_choropleth,
                    year_range=tuple(input.years()),
                    chemical_category=input.chemical_category(),
                    display_mode="compare_individuals",
                    region_filter=current_region_filter
                )

                if choropleth_data.empty:
                    return figure_to_html(create_empty_plot("No data for global map with current selections"))

                if input.snapshot_mode() == "static":
                    y_start, y_end = input.years()
                    chem_cat = input.chemical_category()
                    return figure_to_html(
                        create_contribution_map_plot(
                            choropleth_data,
                            fill_label="Avg Share (%)",
                            main_title=f"Average CS Contribution ({y_start}–{y_end}) — {chem_cat}"
                        )
                    )
                else:
                    return figure_to_html(create_contribution_choropleth(choropleth_data))

            except Exception as e:
                print(f"Error in contribution_map: {str(e)}")
                return figure_to_html(create_empty_plot(f"Error creating map: {str(e)}"))
            
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

        @output
        @render.ui
        def country_cs_plot():
            try:
                article_data = get_raw_dataframe()
                if article_data.empty:
                    return figure_to_html(create_empty_plot("No article data available"))
                
                df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
                if df_filtered.empty:
                    return figure_to_html(create_empty_plot("No country participation data available"))
                
                return figure_to_html(create_article_plot(df_filtered, "Country participation in the growth of the CS"))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error loading article data: {str(e)}"))

        @output
        @render.ui
        def article_top_collabs_plot():
            try:
                df = get_raw_dataframe()
                is_collab = input.top_data_type_filter() == "collabs"
                chem_filter = input.top_collabs_chem_filter()
                
                filtered_data = df[
                    (df['is_collab'] == is_collab) & 
                    (df['chemical'] == chem_filter)
                ]
                
                if filtered_data.empty:
                    return figure_to_html(create_empty_plot("No data available"))
                    
                top_data = (
                    filtered_data.groupby('country')['percentage']
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                
                return figure_to_html(create_top_trends_plot(
                    filtered_data[filtered_data['country'].isin(top_data.index)],
                    f"Top 10 {'Collaborations' if is_collab else 'Countries'}: {chem_filter} Chemicals" 
                ))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error: {str(e)}"))

        @output
        @render.ui
        def article_gdp_plot():
            try:
                article_data = get_raw_dataframe()
                if article_data.empty:
                    return figure_to_html(create_empty_plot("No GDP data available"))
                
                gdp_data = article_data[article_data['source'] == "Annual growth rate of the GDP"]
                if gdp_data.empty:
                    return figure_to_html(create_empty_plot("No GDP data available"))
                    
                return figure_to_html(create_gdp_plot(gdp_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error: {str(e)}"))

        @output
        @render.ui
        def article_researchers_plot():
            try:
                article_data = get_raw_dataframe()
                if article_data.empty:
                    return figure_to_html(create_empty_plot("No researchers data available"))
                
                researchers_data = article_data[article_data['source'] == "Number of Researchers"]
                if researchers_data.empty:
                    return figure_to_html(create_empty_plot("No researchers data available"))
                    
                return figure_to_html(create_researchers_plot(researchers_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error: {str(e)}"))

        @output
        @render.ui
        def article_cs_expansion_plot():
            try:
                article_data = get_raw_dataframe()
                if article_data.empty:
                    return figure_to_html(create_empty_plot("No CS expansion data available"))
                
                cs_data = article_data[article_data['source'] == "Expansion of the CS"]
                if cs_data.empty:
                    return figure_to_html(create_empty_plot("No CS expansion data available"))
                    
                return figure_to_html(create_cs_expansion_plot(cs_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error: {str(e)}"))

        @output
        @render.ui
        def china_us_plot():
            try:
                article_data = get_raw_dataframe()
                if article_data.empty:
                    return figure_to_html(create_empty_plot("No data for China-US plot"))
                
                cs_data = article_data[article_data['source'] == "China-US in the CS"]
                if cs_data.empty:
                    return figure_to_html(create_empty_plot("No 'China-US in the CS' data found"))
                    
                return figure_to_html(create_china_us_dual_axis_plot(cs_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error: {str(e)}"))

    # ADD THIS LINE EXACTLY HERE:
    return App(app_ui, server, static_assets=www_dir)

# # --- PASTING THE MISSING FUNCTION BACK IN ---
# # Cached helper functions
# @lru_cache(maxsize=1)
# def load_article_data():
#     """Load and cache article data"""
#     try:
#         df = get_raw_dataframe()
#         article_columns_map = {'source': 'source', 'year_x': 'year', 'country_x': 'country', 'percentage_x': 'value', 'cc': 'cc'}
#         required_raw_cols = list(article_columns_map.keys())

#         if all(col in df.columns for col in required_raw_cols):
#             article_data_raw = df[required_raw_cols].copy()
#             article_data = article_data_raw.rename(columns=article_columns_map)
#             article_data = article_data.dropna(subset=['value', 'source'])
#             article_data = article_data[article_data['source'] != ""]
#             return article_data
#         return pd.DataFrame()
#     except Exception:
#         return pd.DataFrame()
# --------------------------------------------

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run(port=8888)