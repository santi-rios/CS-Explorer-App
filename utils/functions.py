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


def load_country_data(data_path: str = "./data/data.parquet") -> Dict:
    """
    Load and prepare initial data
    
    Args:
        data_path: Path to the parquet file
        
    Returns:
        Dictionary containing processed data objects
    """
    try:
        # Read parquet file
        df = pd.read_parquet(data_path)
        
        # Process country list  
        country_list = (
            df[df['is_collab'] == False]
            .drop_duplicates(subset=['country', 'iso2c', 'lat', 'lng', 'cc', 'region'])
            .dropna(subset=['country', 'iso2c'])
            .query("country != '' and iso2c != ''")
            .copy()
        )
        
        # Handle missing regions
        country_list['region'] = country_list['region']
        country_list = country_list.sort_values('country').reset_index(drop=True)
        # Get chemical categories - Modified to remove empty values
        chemical_categories = (
            sorted(df['chemical']
                  .dropna()
                  .unique()
                  .tolist())
        )
        # Remove any empty strings
        chemical_categories = [c for c in chemical_categories if c and str(c).strip()]
        # Remove duplicates while preserving order
        chemical_categories = list(dict.fromkeys(chemical_categories))
         
        # Get regions
        regions = sorted(country_list['region'].unique().tolist())
        
        # Get year range
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        
        # Process article data (adjust column names as needed)
        article_columns = ['source', 'year_x', 'country_x', 'percentage_x']
        if all(col in df.columns for col in article_columns):
            article_data = df[article_columns].dropna().copy()
            article_data.columns = ['source', 'year', 'country', 'value']
        else:
            article_data = pd.DataFrame()
            
        return {
            'data': df,
            'country_list': country_list,
            'chemical_categories': chemical_categories,  # Updated list
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year,
            'article_data': article_data
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

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


def create_main_plot(
    data: pd.DataFrame, 
    display_mode: str, 
    selected_isos: List[str],
    country_list: pd.DataFrame = None
) -> go.Figure:
    """
    Create the main plot based on processed data and display mode
    
    Args:
        data: Processed data DataFrame
        display_mode: Current display mode
        selected_isos: List of selected ISO codes
        country_list: Country metadata DataFrame
        
    Returns:
        Plotly Figure object
    """
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display for current selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(template='plotly_white')
        return fig
        
    fig = go.Figure()
    
    if display_mode in ["individual", "compare_individuals"]:
        # Individual country plots
        for group in data['plot_group'].unique():
            group_data = data[data['plot_group'] == group]
            color = group_data['plot_color'].iloc[0] if 'plot_color' in group_data.columns else None
            
            fig.add_trace(go.Scatter(
                x=group_data['year'],
                y=group_data['total_percentage'],
                mode='lines+markers',
                name=group,
                line=dict(color=color, width=1) if color else dict(width=1),
                marker=dict(
                    color=color if color else None,
                    size=group_data['total_percentage'] / 5,  # Size based on value
                    sizemin=4,
                    sizemax=12
                ),
                hovertemplate=(
                    f"<b>{group}</b><br>" +
                    "Year: %{x}<br>" +
                    "Percentage: %{y:.2f}%<extra></extra>"
                )
            ))
            
    elif display_mode == "find_collaborations":
        # Collaboration plots
        colors = px.colors.qualitative.Set1
        color_map = {}
        
        for i, group in enumerate(data['plot_group'].unique()):
            group_data = data[data['plot_group'] == group]
            color = colors[i % len(colors)]
            color_map[group] = color
            
            fig.add_trace(go.Scatter(
                x=group_data['year'],
                y=group_data['total_percentage'],
                mode='lines+markers',
                name=group,
                line=dict(color=color, width=2),
                marker=dict(
                    color=color,
                    size=group_data['total_percentage'] / 5,
                    sizemin=4,
                    sizemax=12
                ),
                hovertemplate=(
                    f"<b>{group}</b><br>" +
                    "Year: %{x}<br>" +
                    "Percentage: %{y:.2f}%<extra></extra>"
                )
            ))
    
    # Update layout
    fig.update_layout(
        title="Chemical Contribution Trends",
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
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02
        )
    )
    
    return fig

def create_contribution_map_plot(
    processed_data_df: pd.DataFrame,
    fill_label: str = "Average Contribution (%)",
    main_title: str = "Average Contribution Over Selected Period"
) -> go.Figure:
    """
    Optimized choropleth map creation
    """
    if processed_data_df.empty or 'total_percentage' not in processed_data_df.columns:
        return create_empty_plot("No data for map")
        
    # Efficient aggregation
    map_data = (
        processed_data_df.groupby(['iso2c', 'country'], as_index=False)
        .agg({
            'total_percentage': ['mean', 'max', 'min'],
            'year': ['min', 'max'],
            'region': 'first'
        })
        .round(2)
    )
    
    # Flatten column names
    map_data.columns = [
        'iso2c', 'country', 
        'avg_percentage', 'max_percentage', 'min_percentage',
        'first_year', 'last_year', 'region'
    ]
    
    # Create choropleth with better color scale
    fig = go.Figure(data=go.Choropleth(
        locations=map_data['iso2c'],
        z=map_data['avg_percentage'],
        locationmode='ISO-3166-1-alpha-2',
        colorscale='Viridis',
        reversescale=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{location})<br>" +
            "Avg Contribution: %{z:.2f}%<br>" +
            "Region: %{customdata[1]}<br>" +
            "Range: %{customdata[2]:.2f}% - %{customdata[3]:.2f}%<br>" +
            "Years: %{customdata[4]:.0f}-%{customdata[5]:.0f}<extra></extra>"
        ),
        customdata=map_data[['country', 'region', 'min_percentage', 'max_percentage', 'first_year', 'last_year']].values,
        colorbar=dict(
            title=fill_label,
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=1
        )
    ))
    
    fig.update_layout(
        title={
            'text': main_title,
            'x': 0.5,
            'xanchor': 'center'
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        height=500
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

def create_trends_plot(data: pd.DataFrame, selected_countries: List[str], mode: str):
    """Create trends plot using Plotly"""
    fig = go.Figure()
    
    value_column = 'total_percentage' # Expect this from get_display_data
    
    if value_column not in data.columns or data.empty:
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
    
    # Sort the entire dataset by plot_group and then by year.
    # This ensures that when we iterate through unique plot_groups,
    # the data for each group is already sorted by year.
    data_sorted = data.sort_values(['plot_group', 'year'])

    plot_title = "Chemical Space Contribution Trends"
    show_legend_for_plot = True # Default to showing legend

    if mode == "compare_individuals":
        plot_title = "Individual Country Contribution Trends"
        for country_name_str in data_sorted['plot_group'].unique(): 
            country_name = str(country_name_str) # Ensure string for name
            # country_data_for_trace will be sorted by year due to the initial sort of data_sorted
            country_data_for_trace = data_sorted[data_sorted['plot_group'] == country_name]

            if country_data_for_trace.empty:
                continue
            
            # Assuming plot_color is consistent for the entire group (country)
            color = None
            if 'plot_color' in country_data_for_trace.columns and not country_data_for_trace.empty:
                # Take the first available color for this country
                first_valid_color = country_data_for_trace['plot_color'].dropna()
                if not first_valid_color.empty:
                    color = first_valid_color.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=country_data_for_trace['year'],
                y=country_data_for_trace[value_column],
                mode='lines+markers',
                name=country_name,
                line=dict(color=color if color else None), # Plotly will assign a color if None
                hovertemplate=(
                    f"<b>{country_name}</b><br>" +
                    "Year: %{x}<br>" +
                    f"Contribution: %{{y:.2f}}%<extra></extra>"
                )
            ))
    elif mode == "find_collaborations":
        plot_title = "Collaboration Trends"
        show_legend_for_plot = False # Hide legend for collaboration mode
        
        collab_type_colors = {
            "Bilateral": "rgba(255, 127, 14, 0.9)",
            "Trilateral": "rgba(44, 160, 44, 0.9)",
            "4-country": "rgba(214, 39, 40, 0.9)",
            "5-country+": "rgba(148, 103, 189, 0.9)",
            "Unknown": "rgba(127, 127, 127, 0.9)"
        }

        for collab_id_str in data_sorted['plot_group'].unique(): 
            collab_id = str(collab_id_str) # Ensure string for name
            # collab_data_for_trace will be sorted by year
            collab_data_for_trace = data_sorted[data_sorted['plot_group'] == collab_id]

            if collab_data_for_trace.empty:
                continue
            
            # Assuming plot_color_group is consistent for the entire group (collaboration)
            collab_type = "Unknown"
            if 'plot_color_group' in collab_data_for_trace.columns and not collab_data_for_trace.empty:
                first_valid_type = collab_data_for_trace['plot_color_group'].dropna()
                if not first_valid_type.empty:
                    collab_type = str(first_valid_type.iloc[0])

            fig.add_trace(go.Scatter(
                x=collab_data_for_trace['year'],
                y=collab_data_for_trace[value_column],
                mode='lines+markers',
                name=collab_id, # Name is kept for hover data, even if legend is hidden
                showlegend=False, 
                line=dict(color=collab_type_colors.get(collab_type, collab_type_colors["Unknown"])),
                hovertemplate=(
                    f"<b>Collaboration: {collab_id}</b> ({collab_type})<br>" +
                    "Year: %{x}<br>" +
                    f"Contribution: %{{y:.2f}}%<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title=plot_title,
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
        hovermode='closest', # Changed from 'x unified' to 'closest' for better individual point hovering
        template='plotly_white',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale'],
        showlegend=show_legend_for_plot # Control overall legend visibility
    )
    
    return fig


def create_summary_table(data: pd.DataFrame, display_mode: str) -> pd.DataFrame:
    """
    Create summary data table based on processed data
    
    Args:
        data: Processed data DataFrame
        display_mode: Current display mode
        
    Returns:
        Summary DataFrame for display
    """
    if data.empty:
        return pd.DataFrame({"Message": ["No data available"]})
        
    if display_mode in ["individual", "compare_individuals"]:
        summary = (
            data.groupby(['iso2c', 'country', 'region', 'chemical'])
            .agg({
                'total_percentage': ['mean', 'max', 'count']
            })
            .round(2)
            .reset_index()
        )
        
        # Flatten column names
        summary.columns = [
            'ISO', 'Country', 'Region', 'Chemical Category',
            'Avg %', 'Max %', 'Years Present'
        ]
        
    elif display_mode == "find_collaborations": 
        summary = (
            data.groupby(['iso2c', 'collab_type', 'chemical'])
            .agg({
                'total_percentage': ['mean', 'max', 'count']
            })
            .round(2)
            .reset_index()
        )
        
        # Flatten column names
        summary.columns = [
            'Collaboration', 'Type', 'Chemical Category',
            'Avg %', 'Max %', 'Years Present'
        ]
    else:
        return pd.DataFrame({"Message": ["Invalid display mode"]})
        
    return summary

def calculate_top_contributors(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    chemical_category: str,
    region_filter: str = "All", 
    country_list: pd.DataFrame = None,
    top_n: int = 10,
    ignore_year_filter: bool = False
) -> pd.DataFrame:
    """
    Calculate top contributing countries based on filters
    
    Args:
        df: Main data DataFrame
        year_range: Year range tuple
        chemical_category: Chemical category filter
        region_filter: Region filter
        country_list: Country metadata
        top_n: Number of top contributors to return
        ignore_year_filter: Whether to ignore year filter
        
    Returns:
        DataFrame with top contributors
    """
    # Base query for solo contributions
    query_df = df[df['is_collab'] == False].copy()
    
    # Apply filters
    if not ignore_year_filter:
        query_df = query_df[
            (query_df['year'] >= year_range[0]) & 
            (query_df['year'] <= year_range[1])
        ]
        
    if chemical_category != "All":
        query_df = query_df[query_df['chemical'] == chemical_category]
        
    if region_filter != "All":
        query_df = query_df[query_df['region'] == region_filter]
        
    # Calculate top contributors
    top_data = (
        query_df.groupby('iso2c')['percentage']
        .mean()
        .reset_index()
        .sort_values('percentage', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    
    top_data['rank'] = range(1, len(top_data) + 1)
    
    # Add country names if available
    if country_list is not None:
        top_data = top_data.merge(
            country_list[['iso2c', 'country']], 
            on='iso2c', 
            how='left'
        )
    
    return top_data[['rank', 'iso2c', 'country', 'percentage']].rename(
        columns={'percentage': 'avg_percentage'}
    )

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
            y=-0.2,
            traceorder="reversed"
        ),
        hovermode='x',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale'],
        # Add custom year range selector buttons
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=3,  # Set the default active button (All Years)
                x=1,
                y=1.15,
                buttons=list([
                    dict(
                        label="1996-2012 (US Dominance)",
                        method="relayout",
                        args=[{"xaxis.range": [1996, 2013]}]
                    ),
                    dict(
                        label="2012-2022 (China's Rise)",
                        method="relayout", 
                        args=[{"xaxis.range": [max_year - 10, max_year+1]}]
                    ),
                    dict(
                        label="All Years",
                        method="relayout",
                        args=[{"xaxis.range": [min_year-1, max_year+1]}]
                    )
                ]),
            )
        ]
    )

    return fig

def create_article_plot_simple(
    article_df: pd.DataFrame,
    source_title: str, 
    y_title: str,
    animate: bool = False
) -> go.Figure:
    """
    Create simplified plot for static article data
    
    Args:
        article_df: Article data DataFrame
        source_title: Source identifier
        y_title: Y-axis label
        animate: Whether to animate (not implemented)
        
    Returns:
        Plotly Figure object
    """
    if article_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for this figure",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
        
    # Define consistent colors (subset of R version)
    country_colors = {
        "China": "#c5051b", "China alone": "#c56a75", "China w/o US": "#9b2610",
        "France": "#0a3161", "Germany": "#000000", "India": "#ff671f", 
        "Japan": "#000091", "Russia": "#d51e9b", "USA alone": "#3b5091",
        "USA w/o China": "#006341", "United Kingdom": "#74acdf", 
        "United States": "#002852", "US": "#002853"
    }
    
    fig = go.Figure()
    
    # Process data based on source
    plot_data = article_df.copy()
    
    if source_title == "Number of Researchers":
        plot_data['plot_value'] = plot_data['value'] / 1e6
        y_title = f"{y_title} (Millions)"
    elif source_title == "Annual growth rate of the GDP":
        plot_data['plot_value'] = plot_data['value']
        y_title = f"{y_title} (%)"
    else:
        plot_data['plot_value'] = plot_data['value']
        
    # Add traces for each country
    for country in plot_data['country'].unique():
        country_data = plot_data[plot_data['country'] == country]
        color = country_colors.get(country, '#1f77b4')  # Default blue
        
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['plot_value'],
            mode='lines+markers',
            name=country,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=6),
            hovertemplate=(
                f"<b>{country}</b><br>" +
                "Year: %{x}<br>" +
                f"Value: %{{y:.2f}}<extra></extra>"
            )
        ))
    
    # Add annotations for GDP plot
    if source_title == "Annual growth rate of the GDP":
        fig.add_vline(x=2007.5, line_dash="dash", line_color="grey")
        fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")
        fig.add_annotation(x=2007.5, y=fig.data[0].y.max() * 0.9 if fig.data else 10,
                          text="Financial Crisis", showarrow=True)
        fig.add_annotation(x=2019.5, y=fig.data[0].y.max() * 0.8 if fig.data else 8,
                          text="COVID-19", showarrow=True)
    
    fig.update_layout(
        title=f"Figure: {source_title}",
        xaxis_title="Year",
        yaxis_title=y_title,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    return fig

def create_top_collabs_plot(top_collab_data: pd.DataFrame, title: str = "Collaboration Trends") -> go.Figure:
    """
    Create plot for top collaboration trends
    
    Args:
        top_collab_data: Data with top collaborations
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if top_collab_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No collaboration data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
        
    fig = go.Figure()
    
    # Use qualitative colors
    colors = px.colors.qualitative.Set1
    
    for i, entity in enumerate(top_collab_data['country'].unique()):
        entity_data = top_collab_data[top_collab_data['country'] == entity]
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data['percentage'],
            mode='lines+markers',
            name=entity,
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=4),
            hovertemplate=(
                f"<b>{entity}</b><br>" +
                "Year: %{x}<br>" +
                "Percentage: %{y:.2f}%<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=title,
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
        legend=dict(orientation="h", y=-0.3)
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
        colorscale='Viridis',
        # colorscale=[[0, 'rgb(10, 49, 97)'], [1, 'rgb(238, 28, 37)']],
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
            title="Avg. Contribution",
            orientation = "h",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.1,
            # tickangle =0,
            # tickformat = ".2f",
            ticklabelstep = 4,
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

    # Add dropdowns
    button_layer_1_height = 1.02
    fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["colorscale", "Viridis"],
                    label="Viridis",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Cividis"],
                    label="Cividis",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Reds"],
                    label="Reds",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Blues"],
                    label="Blues",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Greens"],
                    label="Greens",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Jet"],
                    label="Rainbow",
                    method="restyle"
                ),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"
        ),
        dict(
            buttons=list([
                dict(
                    args=["reversescale", False],
                    label="False",
                    method="restyle"
                ),
                dict(
                    args=["reversescale", True],
                    label="True",
                    method="restyle"
                )
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.37,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"
        ),
    ]
    )

    fig.update_layout(
    annotations=[
        dict(text="Colorscale", x=0, xref="paper", y=1.04, yref="paper",
                             align="left", showarrow=False),
        dict(text="Reverse Colorscale", x=0.25, xref="paper", y=1.04,
                             yref="paper", showarrow=False)
    ])

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
            rangeslider=dict(
                visible=True
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
    
    fig.update_layout(
        yaxis = dict(
            fixedrange = True,
            title=go.layout.yaxis.Title(
                text="Number of New Substances",
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
                standoff=1,
                font=dict(size=10, color='black')
            )
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
