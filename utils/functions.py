"""
Helper functions for the Chemical Space Explorer Python Shiny App
Translated from R functions.R
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple, Union
import geopandas as gpd
from pathlib import Path
from functools import lru_cache
import folium

@lru_cache(maxsize=1)
def get_raw_dataframe(data_path: str = "./data/data.parquet") -> pd.DataFrame:
    """Read data.parquet into memory once and cache it."""
    return pd.read_parquet(data_path)


@lru_cache(maxsize=1)
def get_world_geometries(world_path: str = "./data/world_boundaries.geojson") -> dict:
    """Read world_boundaries.geojson once and cache the ISO-2 -> geometry lookup."""
    world = gpd.read_file(world_path)
    iso_to_geometry = {}
    for _, row in world.iterrows():
        for col in ['iso_a2_eh', 'wb_a2', 'iso_a2']:
            if col in world.columns:
                candidate = row.get(col)
                if candidate is not None and isinstance(candidate, str) and len(candidate) == 2 and candidate != "-99":
                    iso_to_geometry[candidate.upper()] = row.geometry
                    break
    return iso_to_geometry

def get_display_data(
    df: pd.DataFrame,
    selected_isos: List[str],
    year_range: Tuple[int, int],
    chemical_category: str,
    display_mode: str,
    region_filter: str = "All",
    country_list: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Optimized data fetching with early filtering and lazy evaluation
    """
    if df.empty:
        return pd.DataFrame()
    
    # Early exit for collaboration mode without sufficient selection
    if display_mode == "find_collaborations" and len(selected_isos) < 2:
        return pd.DataFrame()
    
    # Apply filters progressively for efficiency
    # Start with most selective filters first
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Chemical filter
    filtered_df = filtered_df[filtered_df['chemical'] == chemical_category]
    
    # Early exit if no data after initial filtering
    if filtered_df.empty:
        return pd.DataFrame()

    if display_mode in ["individual", "compare_individuals"]:
        if not selected_isos:
            return pd.DataFrame()

        # Filter for individual countries only
        result = filtered_df[
            (filtered_df['is_collab'] == False) & 
            (filtered_df['iso2c'].isin(selected_isos))
        ].copy()
        
        # Apply region filter efficiently
        if region_filter != "All" and 'region' in result.columns:
            result = result[result['region'] == region_filter]
            
        if result.empty:
            return pd.DataFrame()

        # Add metadata efficiently
        if country_list is not None:
            country_meta = country_list[['iso2c', 'country', 'cc', 'region']].drop_duplicates(subset=['iso2c'])
            result = result.merge(country_meta, on='iso2c', how='left', suffixes=('_orig', '_meta'))
            
            # Use metadata preferentially
            for col in ['country', 'cc', 'region']:
                meta_col = f'{col}_meta'
                if meta_col in result.columns:
                    result[col] = result[meta_col].fillna(result.get(f'{col}_orig', ''))
                    
        result['plot_group'] = result.get('country', result['iso2c'])
        result['plot_color'] = result.get('cc', '#808080')
        
    elif display_mode == "find_collaborations":
        # Optimized collaboration filtering
        collab_df = filtered_df[filtered_df['is_collab'] == True].copy()
        
        if collab_df.empty:
            return pd.DataFrame()

        # Efficient collaboration filtering using vectorized operations
        selected_set = set(selected_isos)
        
        def has_all_partners(iso_string):
            if pd.isna(iso_string):
                return False
            partners = set(str(iso_string).split('-'))
            return selected_set.issubset(partners)
        
        mask = collab_df['iso2c'].apply(has_all_partners)
        result = collab_df[mask].copy()

        if result.empty:
            return pd.DataFrame()
            
        # Add collaboration metadata
        result['partners'] = result['iso2c'].str.split('-')
        result['collab_size'] = result['partners'].str.len()
        
        # Vectorized collaboration type assignment
        result['collab_type'] = result['collab_size'].map({
            2: "Bilateral",
            3: "Trilateral", 
            4: "4-country"
        }).fillna("5-country+")
        
        result['plot_group'] = result['country']
        result['plot_color_group'] = result['collab_type']
    else:
        return pd.DataFrame()
        
    # Standardize columns efficiently
    if not result.empty:
        # Handle numeric conversions
        for col in ['year']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # Standardize percentage column
        percentage_cols = ['percentage', 'value', 'value_raw', 'percentage_x']
        for col in percentage_cols:
            if col in result.columns:
                result['total_percentage'] = pd.to_numeric(result[col], errors='coerce')
                break
        
        # Drop rows with invalid data
        result = result.dropna(subset=['year', 'total_percentage'])

    return result




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

def create_trends_plot(data: pd.DataFrame, selected_countries: List[str], mode: str):
    """Create trends plot using Plotly with an improved, responsive layout."""
    fig = go.Figure()
    
    value_column = 'total_percentage'
    
    if value_column not in data.columns or data.empty:
        fig.add_annotation(
            text=f"Error: Missing '{value_column}' column or no data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
    
    data_sorted = data.sort_values(['plot_group', 'year'])

    # --- Trace Generation ---
    if mode == "compare_individuals":
        for country_name_str in data_sorted['plot_group'].unique(): 
            country_name = str(country_name_str)
            country_data_for_trace = data_sorted[data_sorted['plot_group'] == country_name]
            if country_data_for_trace.empty:
                continue
            
            color = None
            if 'plot_color' in country_data_for_trace.columns:
                first_valid_color = country_data_for_trace['plot_color'].dropna()
                if not first_valid_color.empty:
                    color = first_valid_color.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=country_data_for_trace['year'],
                y=country_data_for_trace[value_column],
                mode='lines+markers',
                name=country_name,
                line=dict(color=color),
                marker=dict(symbol='circle', size=8, color=color),
                hovertemplate=(
                    f"<b>{country_name}</b><br>" +
                    "Year: %{x}<br>" +
                    f"Contribution: %{{y:.2f}}%<extra></extra>"
                )
            ))
            
    elif mode == "find_collaborations":
        collab_type_colors = {
            "Bilateral": "rgba(255, 127, 14, 0.9)", "Trilateral": "rgba(44, 160, 44, 0.9)",
            "4-country": "rgba(214, 39, 40, 0.9)", "5-country+": "rgba(148, 103, 189, 0.9)",
            "Unknown": "rgba(127, 127, 127, 0.9)"
        }
        collab_type_symbols = {
            "Bilateral": "circle", "Trilateral": "square",
            "4-country": "diamond", "5-country+": "triangle-up", "Unknown": "cross"
        }

        for collab_id_str in data_sorted['plot_group'].unique(): 
            collab_id = str(collab_id_str)
            collab_data_for_trace = data_sorted[data_sorted['plot_group'] == collab_id]
            if collab_data_for_trace.empty:
                continue
            
            collab_type = "Unknown"
            if 'plot_color_group' in collab_data_for_trace.columns:
                first_valid_type = collab_data_for_trace['plot_color_group'].dropna()
                if not first_valid_type.empty:
                    collab_type = str(first_valid_type.iloc[0])

            display_name = collab_id
            if 'iso2c' in collab_data_for_trace.columns:
                iso_codes_value = collab_data_for_trace['iso2c'].iloc[0]
                if isinstance(iso_codes_value, str) and '-' in iso_codes_value:
                    iso_codes = iso_codes_value.split('-')
                    highlighted_codes = [f"<b>{code}</b>" if code in selected_countries else code for code in iso_codes]
                    display_name = ' + '.join(highlighted_codes)

            fig.add_trace(go.Scatter(
                x=collab_data_for_trace['year'],
                y=collab_data_for_trace[value_column],
                mode='lines+markers',
                name=display_name,
                meta=collab_type,
                showlegend=True,
                line=dict(color=collab_type_colors.get(collab_type, collab_type_colors["Unknown"]), width=1.5),
                marker=dict(
                    symbol=collab_type_symbols.get(collab_type, "circle"), size=8,
                    color=collab_type_colors.get(collab_type, collab_type_colors["Unknown"])
                ),
                hovertemplate=(
                    f"<b>Collaboration Network:</b><br>" +
                    f"Partners: {display_name.replace(' + ', ' 🤝 ')}<br>" +
                    f"Complexity: {collab_type}<br>" +
                    "Year: %{x}<br>" +
                    f"Share of New Substances: %{{y:.2f}}%<extra></extra>"
                ),
            ))
    
    # --- Layout and Interactivity Improvements ---
    legend_config = {
        "orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
        "bgcolor": "rgba(255,255,255,0.8)", "bordercolor": "rgba(0,0,0,0.1)", "borderwidth": 1,
        "title": {"text": "Countries" if mode == "compare_individuals" else "Collaborations", "font": {"size": 12}}
    }

    updatemenus = []
    if mode == "find_collaborations":
        collab_types = sorted(data_sorted['plot_color_group'].dropna().unique().tolist()) if 'plot_color_group' in data_sorted.columns else []
        if collab_types:
            filter_buttons = [dict(label="All Types", method="restyle", args=[{"visible": [True] * len(fig.data)}])]
            for collab_type in collab_types:
                visibility = [trace.meta == collab_type for trace in fig.data]
                filter_buttons.append(dict(label=collab_type, method="restyle", args=[{"visible": visibility}]))
            
            updatemenus.append(dict(
                type="buttons", direction="right", active=0, x=0.5, y=1.15,
                xanchor="center", yanchor="top", buttons=filter_buttons,
                pad={"r": 5, "t": 10}, bgcolor="rgba(240,240,240,0.95)",
                bordercolor="#ccc", borderwidth=1
            ))

# Place this right before fig.update_layout(...) in create_trends_plot():
    if fig.data:
        all_y = [val for trace in fig.data for val in trace['y'] if val is not None]
        max_val = max(all_y) if all_y else 10
        for event_year, label in [(2008, "Financial Crisis"), (2020, "COVID-19")]:
            fig.add_vline(x=event_year, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")
            fig.add_annotation(
                x=event_year, y=max_val * 0.95, text=label,
                showarrow=False, textangle=-90, font=dict(size=10, color="gray"), xshift=-10
            )

    fig.update_layout(
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="Percentage of New Substances", standoff=5, font=dict(size=12, color='black'))
        ),
        xaxis=dict(
            fixedrange=True,
            title=dict(text="Year", standoff=5, font=dict(size=12, color='black'))
        ),
        hovermode='x unified',
        template='plotly_white',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale'],
        showlegend=True,
        legend=legend_config,
        updatemenus=updatemenus,
        margin=dict(
            l=60, 
            r=200,  # Increased right margin for the vertical legend
            b=60,
            t=80 if mode == "find_collaborations" and updatemenus else 40
        )
    )
    
    return fig


def create_article_plot(data: pd.DataFrame, title: str):
    """Create article plots"""
    fig = go.Figure()
    
    # Find min and max years for dynamic range buttons
    min_year = int(data['year'].min()) if not data.empty else 1996
    max_year = int(data['year'].max()) if not data.empty else 2022
    recent_years = max(max_year - 5, min_year)
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        color = country_data['cc'].iloc[0] if 'cc' in country_data.columns and not country_data.empty else None
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['value'],
            mode='lines+markers',
            name=country,
            line=dict(color=color, width=1) if color else dict(width=1),
            marker=dict(size=country_data['value'].abs().clip(lower = 1, upper=10), opacity=0.3, color=color if color else 'red')
        ))
    
    fig.update_layout(
        height=550,
        yaxis = dict(
            ticksuffix='%',
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="New Substances",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            autorange=False,
            range=[min_year-1, max_year+1],
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1,
                font=dict(size=12, color='black')
            ),
            # rangeslider=dict(
            #     visible=True,
            #     thickness=0.05,
            #     bgcolor="rgba(99, 110, 250, 0.2)"
            # ),
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h", 
            y=-0.25,
            traceorder="reversed"
        ),
        hovermode='x',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale'],
        # Add custom year range selector buttons
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=2,  # Set the default active button (All Years)
                x=1,
                y=1.35,
                buttons=list([
                    dict(
                        label="1996-2012",
                        method="relayout",
                        args=[{"xaxis.range": [1996, 2013]}]
                    ),
                    dict(
                        label="2012-2022",
                        method="relayout", 
                        args=[{"xaxis.range": [max_year - 10, max_year+1]}]
                    ),
                    dict(
                        label="All",
                        method="relayout",
                        args=[{"xaxis.range": [min_year-1, max_year+1]}]
                    )
                ]),
            )
        ]
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
            marker=dict(color=entity_data['cc'].iloc[0] if 'cc' in entity_data.columns and not entity_data.empty else 'red'),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Year: %{x}<br>" +
                "Contribution: %{y:.2f}%<extra></extra>"
            )
        ))
    
    fig.update_layout(
        # title=title,
        height=550,
        yaxis = dict(
            ticksuffix='%',
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="New Substances",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            fixedrange = True,
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            # title="Top Contributors",
            # text = "Sorted by Avg Contribution",
            orientation="h", 
            y=-0.2,
            traceorder="reversed"  # Display in same order as traces
        ),
        hovermode='closest',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )

    return fig

def create_folium_map(country_list: pd.DataFrame, selected_countries: List[str]) -> folium.Map:
    """Create interactive Folium map with improved region handling and better country visualization"""
    
    # Determine map center based on filtered countries
    if not country_list.empty:
        center_lat = country_list['lat'].mean()
        center_lng = country_list['lng'].mean()
        
        # Adjust zoom based on region spread
        lat_range = country_list['lat'].max() - country_list['lat'].min()
        lng_range = country_list['lng'].max() - country_list['lng'].min()
        
        # Determine appropriate zoom level
        if lat_range > 60 or lng_range > 120:  # Global view
            zoom_start = 1
        elif lat_range > 30 or lng_range > 60:  # Continental view
            zoom_start = 3
        elif lat_range > 15 or lng_range > 30:  # Regional view
            zoom_start = 3
        else:  # Local view
            zoom_start = 4
    else:
        center_lat, center_lng, zoom_start = 30, 10, 2

    # Determine circle radius in meters based on initial zoom level for fallback markers
    if zoom_start > 3:
        circle_radius_meters = 80000  # Smaller radius for zoomed-in views
    else:
        circle_radius_meters = 85000 # Larger radius for zoomed-out views
    
    m = folium.Map(
        tiles="OpenStreetMap",
        location=[center_lat, center_lng], 
        zoom_start=zoom_start
        )
    
    # Define a simplified, universal legend
    legend_html = '''
    <div style=" 
                bottom: 50px; right: 10px; 
                background-color: rgba(255, 255, 255, 0.85);
                border: 1px solid #bbb;
                border-radius: 5px;
                z-index:9999; 
                padding: 10px;
                font-size: 12px;
                font-family: sans-serif;">
        <div style="display: flex; align-items: center;">
            <span style="background: #83928e; border: 1px solid #333; width: 14px; height: 14px; display: inline-block; margin-right: 6px; opacity: 0.7;"></span>
            <span>Available Countries</span>
        </div>
    </div>
    '''
    
    # Add region info to map title
    if len(country_list) > 0:
        regions_in_map = country_list['region'].unique()
        if len(regions_in_map) == 1 and regions_in_map[0] != 'Other':
            map_title = f"Region: {regions_in_map[0]} ({len(country_list)} countries)"
        else:
            map_title = f"Showing {len(country_list)} countries"
        
        # Add a subtle title overlay
        title_html = f'''
        <div style="
                    top: 10px; left: 50px; width: 250px; height: 20px; 
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 2px solid rgba(0,0,0,0.2);
                    border-radius: 5px;
                    z-index:9999; 
                    font-size:10px;
                    text-align: center;
                    padding: 3px;">
            {map_title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
    
    # Load world geometries for better country shapes
    
    try:
        iso_to_geometry = get_world_geometries()
        
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
                color = "#83928e"  # Default color for unselected countries
                fill_opacity = 0.5
                stroke_weight = 1
            
            # Try different case variations of the ISO code
            geometry = None
            for iso_variant in [iso, iso.upper(), iso.lower()]:
                if iso_variant in iso_to_geometry:
                    geometry = iso_to_geometry[iso_variant]
                    break
                    
            # Simplified popup with just action and country name
            popup_html = f"""
            <div style="min-width: 120px; text-align: center;">
                <button onclick="
                    if (window.parent && window.parent.Shiny) {{
                        window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }} else if (window.Shiny) {{
                        window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                    }}
                " style="
                    padding: 8px 12px; 
                    margin: 5px 0; 
                    cursor: pointer;
                    background-color: {'#e74c3c' if iso in selected_countries else '#3498db'};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    width: 100%;
                ">
                    {'🗑️ Deselect ' if iso in selected_countries else '✅ Select '}{country_name} ({iso})
                </button>
            </div>
            """
                
            if geometry is not None:
                # Use the found geometry
                folium.GeoJson(
                    geometry,
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
                # Use folium.Circle for scalable markers for countries without shapes
                folium.Circle(
                    location=[country_row['lat'], country_row['lng']],
                    radius=circle_radius_meters,
                    color= "#525756",
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=folium.Tooltip(
                        f"<b>{country_name}</b><br>Region: {region}<br>⚠️ Using approximate location",
                        sticky=True
                    )
                ).add_to(m)
                
        # Add the simplified legend to the map
        # m.get_root().html.add_child(folium.Element(legend_html))
                
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        # Fallback to markers if GeoJSON fails
        for _, country in country_list.iterrows():
            iso = country['iso2c']
            country_name = country['country']
            region = country.get('region', 'Unknown')
            
            if iso in selected_countries:
                color = country['cc']
                fill_opacity = 0.8
            else:
                color = 'lightblue'
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
            
            folium.Circle(
                location=[country['lat'], country['lng']],
                radius=circle_radius_meters,
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=folium.Tooltip(
                    f"<b>{country_name}</b><br>Region: {region}<br>⚠️ Using approximate location",
                    sticky=True
                )
            ).add_to(m)
            
        # Add warning about missing shapes
        warning_html = '''
        <div style="position: fixed; 
                    bottom: 10px; left: 50px; 
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 2px solid rgba(255, 87, 34, 0.5);
                    border-radius: 5px;
                    z-index:9998; 
                    padding: 10px;
                    font-size: 12px;
                    max-width: 300px;">
            <div style="color: #d32f2f;"><b>⚠️ Note:</b></div>
            <div>Using approximate location markers since country shapes could not be loaded.</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(warning_html))
        # Add the simplified legend in the except block as well
        m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_contribution_choropleth(data: pd.DataFrame):
    """Create animated world choropleth map"""
    if data.empty or 'total_percentage' not in data.columns:
        return create_empty_plot("No data available for choropleth")
    
    # Sort by year to ensure animation frames play sequentially
    anim_data = data.sort_values('year').dropna(subset=['total_percentage'])
    
    fig = px.choropleth(
        anim_data,
        locations="iso3c",
        color="total_percentage",
        hover_name="country",
        animation_frame="year",
        color_continuous_scale="Viridis",
        range_color=[0, anim_data['total_percentage'].max()],
        projection="robinson",
        labels={'total_percentage': 'Contribution (%)'}
    )
    
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="lightgray"),
        template='plotly_white',
        height=550,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Improve animation speed and transition
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
    
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
        height=500,
        yaxis = dict(
            ticksuffix='%',
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="GDP Growth Rate",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            fixedrange = True,
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1,
                font=dict(size=12, color='black')
            ),
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h", 
            y=-0.2,
            traceorder="reversed"  # Display in same order as traces
        ),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
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
        yaxis = dict(
            ticksuffix='M',
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="Number of Researchers (Millions)",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            fixedrange = True,
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h", 
            y=-0.2,
            traceorder="reversed"  # Display in same order as traces
        ),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
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
            mode='lines',
            name=country,
            line=dict(width=2),
            # marker=dict(size=6)
        ))
    # Add historical context markers for CS perturbations
    fig.add_vline(x=2008, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")
    fig.add_vline(x=2020, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")
    
    if fig.data:
        max_val = max([max(trace['y']) for trace in fig.data if len(trace['y']) > 0])
        fig.add_annotation(
            x=2008, y=max_val * 0.95, text="Financial Crisis", 
            showarrow=False, textangle=-90, font=dict(size=10, color="gray"), xshift=-10
        )
        fig.add_annotation(
            x=2020, y=max_val * 0.95, text="COVID-19", 
            showarrow=False, textangle=-90, font=dict(size=10, color="gray"), xshift=-10
        )
    fig.update_layout(
        yaxis = dict(
            fixedrange = True,
            type="log",
            title=go.layout.yaxis.Title(
                text="Number of New Substances (Log Scale)",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            fixedrange = True,
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h", 
            y=-0.2,
            traceorder="reversed"  # Display in same order as traces
        ),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    
    return fig

def create_china_us_dual_axis_plot(data: pd.DataFrame):
    """
    Create a dual-axis plot for China-US comparison in the Chemical Space.
    Plots "United States" and "China" on the primary y-axis,
    and "CN-US collab/US" and "CN-US collab/CN" on the secondary y-axis.
    """
    fig = go.Figure()

    primary_yaxis_categories = ["United States", "China"]
    secondary_yaxis_categories = ["CN-US collab/US", "CN-US collab/CN"]

    # Define colors for clarity
    # You can customize these colors
    colors = {
        "United States": "blue",
        "China": "red",
        "CN-US collab/US": "lightblue",
        "CN-US collab/CN": "salmon"
    }

    for country_name in data['country'].unique():
        country_data = data[data['country'] == country_name]
        
        trace_params = {
            'x': country_data['year'],
            'y': country_data['value'],
            'mode': 'lines+markers', # Added markers for better visibility of points
            'name': country_name,
            'line': dict(width=2, color=colors.get(country_name)),
            'marker': dict(size=6, color=colors.get(country_name)) # Added markers
        }

        if country_name in secondary_yaxis_categories:
            trace_params['yaxis'] = 'y2'
        
        fig.add_trace(go.Scatter(**trace_params))

    fig.update_layout(
        # title_text="China-US Contributions & Collaboration Impact",
        height=550,
        yaxis = dict(
            ticksuffix='%',
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="National Contribution to the National share of the CS",
                standoff=1,
                font=dict(size=12, color='black')
            )
        ),
        xaxis = dict(
            fixedrange = True,
            title=go.layout.xaxis.Title(
                text="Year",
                standoff=1.3,
                font=dict(size=11, color='black')
            )
            # rangeslider=dict(
            #     visible=True,
            #     thickness=0.05,
            #     bgcolor="rgba(99, 110, 250, 0.2)"
            # ),
        ),
        yaxis2=dict(
            overlaying='y',
            side='right',
            ticksuffix= '%',
            fixedrange=True,
            title=go.layout.yaxis.Title(
                text="China-US Contribution to the National share of the CS",
                standoff=1,
                font=dict(size=10, color='black')
            )
            # titlefont=dict(color=colors.get("CN-US collab/US")) # Optional: color secondary axis title
            # rangemode='tozero' # Or 'normal', depending on data
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    
    return fig

def create_contribution_map_plot(
    processed_data_df: pd.DataFrame,
    fill_label: str = "Avg Contribution (%)",
    main_title: str = "Average Contribution Over Selected Period"
) -> go.Figure:
    """Static choropleth map showing average country contribution over the selected period."""
    if processed_data_df.empty or 'total_percentage' not in processed_data_df.columns:
        return create_empty_plot("No data for map")

    group_cols = ['iso3c', 'iso2c', 'country'] if 'iso3c' in processed_data_df.columns else ['iso2c', 'country']
    map_data = (
        processed_data_df.groupby(group_cols, as_index=False)
        .agg(
            avg_percentage=('total_percentage', 'mean'),
            max_percentage=('total_percentage', 'max'),
            min_percentage=('total_percentage', 'min'),
            first_year=('year', 'min'),
            last_year=('year', 'max'),
            region=('region', 'first')
        )
        .round(2)
    )

    # Use square-root scaling for color visibility across 0.01% - 41% while displaying real % on ticks
    max_val = max(float(map_data['avg_percentage'].max()), 1.0)
    tick_vals_real = [0, 0.5, 2, 5, 10, 20, 30, 40]
    tick_vals_real = [v for v in tick_vals_real if v <= max_val * 1.1]
    tick_vals_scaled = [np.sqrt(v) for v in tick_vals_real]

    loc_col = 'iso3c' if 'iso3c' in map_data.columns else 'country'
    loc_mode = 'ISO-3' if 'iso3c' in map_data.columns else 'country names'

    fig = go.Figure(data=go.Choropleth(
        locations=map_data[loc_col],
        z=np.sqrt(map_data['avg_percentage'].clip(lower=0)),
        locationmode=loc_mode,
        colorscale='Viridis',
        zmin=0,
        zmax=np.sqrt(max_val),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>" +
            "Avg Contribution: <b>%{customdata[2]:.2f}%</b><br>" +
            "Region: %{customdata[3]}<br>" +
            "Range: %{customdata[4]:.2f}% – %{customdata[5]:.2f}%<br>" +
            "Years: %{customdata[6]:.0f}–%{customdata[7]:.0f}<extra></extra>"
        ),
        customdata=map_data[[
            'country', 'iso2c', 'avg_percentage', 'region',
            'min_percentage', 'max_percentage', 'first_year', 'last_year'
        ]].values,
        colorbar=dict(
            title=dict(text=fill_label, side="right"),
            tickvals=tick_vals_scaled,
            ticktext=[f"{v}%" for v in tick_vals_real]
        )
    ))

    fig.update_layout(
        title={'text': main_title, 'x': 0.5, 'xanchor': 'center'},
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            projection_type='robinson',
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        height=550,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


def create_contribution_choropleth(data: pd.DataFrame) -> go.Figure:
    """Create animated yearly world choropleth map."""
    if data.empty or 'total_percentage' not in data.columns:
        return create_empty_plot("No data available for choropleth")

    anim_data = data.sort_values('year').dropna(subset=['total_percentage']).copy()
    anim_data['sqrt_pct'] = np.sqrt(anim_data['total_percentage'].clip(lower=0))
    max_val = max(float(anim_data['total_percentage'].max()), 1.0)
    tick_vals_real = [v for v in [0, 0.5, 2, 5, 10, 20, 30, 40] if v <= max_val * 1.1]

    fig = px.choropleth(
        anim_data,
        locations="iso3c",
        color="sqrt_pct",
        hover_name="country",
        custom_data=["country", "iso2c", "year", "total_percentage", "region"],
        animation_frame="year",
        color_continuous_scale="Viridis",
        range_color=[0, np.sqrt(max_val)],
        projection="robinson"
    )

    hover_tmpl = (
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
        "Year: %{customdata[2]}<br>"
        "Contribution: <b>%{customdata[3]:.2f}%</b><br>"
        "Region: %{customdata[4]}<extra></extra>"
    )
    fig.update_traces(hovertemplate=hover_tmpl)
    for frame in fig.frames:
        for trace in frame.data:
            trace.hovertemplate = hover_tmpl

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="Contribution (%)", side="right"),
            tickvals=[np.sqrt(v) for v in tick_vals_real],
            ticktext=[f"{v}%" for v in tick_vals_real]
        ),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="lightgray"),
        template='plotly_white',
        height=550,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250

    return fig