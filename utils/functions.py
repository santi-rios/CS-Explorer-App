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
        country_list['region'] = country_list['region'].fillna('Other')
        country_list = country_list.sort_values('country').reset_index(drop=True)
        # Get chemical categories - Modified to remove empty values
        chemical_categories = (
            ['All'] + 
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
    Fetch and process data based on user selections
    
    Args:
        df: Main data DataFrame
        selected_isos: List of selected ISO codes
        year_range: Tuple of (min_year, max_year)
        chemical_category: Selected chemical category
        display_mode: Current display mode
        region_filter: Selected region filter
        country_list: Country metadata DataFrame
        
    Returns:
        Processed DataFrame ready for plotting
    """
    if not selected_isos:
        return pd.DataFrame()
        
    # Base filtering
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ].copy()
    
    # Apply chemical filter - Modified to handle "All" correctly
    if chemical_category == "All":
        filtered_df = filtered_df[filtered_df['chemical'] == "All"]
    else:
        filtered_df = filtered_df[filtered_df['chemical'] == chemical_category]
    
        
    # Logic based on display mode
    if display_mode in ["individual", "compare_individuals"]:
        # Individual country data
        result = filtered_df[
            (filtered_df['is_collab'] == False) & 
            (filtered_df['iso2c'].isin(selected_isos))
        ].copy()
        
        # Apply region filter
        if region_filter != "All":
            result = result[result['region'] == region_filter]
            
        # Add plotting metadata
        if country_list is not None:
            result = result.merge(
                country_list[['iso2c', 'country', 'cc', 'region']], 
                on='iso2c', 
                how='left',
                suffixes=('', '_meta')
            )
            
        result['plot_group'] = result['country']
        result['plot_color'] = result.get('cc', '#808080')
        
    elif display_mode == "find_collaborations":
        # Collaboration data
        collab_df = filtered_df[filtered_df['is_collab'] == True].copy()
        
        # Filter for collaborations involving ALL selected countries
        mask = pd.Series([True] * len(collab_df))
        for iso in selected_isos:
            mask &= collab_df['iso2c'].str.contains(iso)
            
        result = collab_df[mask].copy()
        
        if not result.empty:
            # Add collaboration metadata
            result['partners'] = result['iso2c'].str.split('-')
            result['collab_size'] = result['partners'].apply(len)
            result['collab_type'] = result['collab_size'].map({
                2: "Bilateral",
                3: "Trilateral", 
                4: "4-country",
            }).fillna("5-country+")
            
            result['plot_group'] = result['country']
            result['plot_color'] = result['collab_type']
    else:
        result = pd.DataFrame()
        
    # Ensure numeric types
    if not result.empty:
        result['year'] = pd.to_numeric(result['year'])
        result['percentage'] = pd.to_numeric(result['percentage'])
        result = result.rename(columns={'percentage': 'total_percentage'})
        
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
                line=dict(color=color, width=2) if color else dict(width=2),
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
        xaxis_title="Year",
        yaxis_title="% of New Substances",
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
    Create choropleth map showing average contribution percentages
    
    Args:
        processed_data_df: Processed data with country contributions
        fill_label: Legend title for fill variable
        main_title: Main plot title
        
    Returns:
        Plotly Figure object
    """
    if processed_data_df.empty or 'total_percentage' not in processed_data_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for map",
            xref="paper", yref="paper", 
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
        
    # Calculate average percentage per country
    map_data = (
        processed_data_df.groupby(['iso2c', 'country', 'region'])
        .agg({
            'total_percentage': ['mean', 'max', 'min'],
            'year': ['min', 'max']
        })
        .round(2)
        .reset_index()
    )
    
    # Flatten column names
    map_data.columns = [
        'iso2c', 'country', 'region', 
        'avg_percentage', 'max_percentage', 'min_percentage',
        'first_year', 'last_year'
    ]
    
    # Create choropleth
    fig = go.Figure(data=go.Choropleth(
        locations=map_data['iso2c'],
        z=map_data['avg_percentage'],
        locationmode='ISO-3166-1-alpha-2',
        colorscale='Viridis',
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{location})<br>" +
            "Avg Contribution: %{z:.2f}%<br>" +
            "Region: %{customdata[1]}<br>" +
            "Max: %{customdata[2]:.2f}%<br>" +
            "Years: %{customdata[3]:.0f}-%{customdata[4]:.0f}<extra></extra>"
        ),
        customdata=map_data[['country', 'region', 'max_percentage', 'first_year', 'last_year']].values,
        colorbar=dict(title=fill_label)
    ))
    
    fig.update_layout(
        title=main_title,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
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
        xaxis_title="Year",
        yaxis_title="% of New Substances",
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.3)
    )
    
    return fig


