"""
Helper functions for the Chemical Space Explorer Python Shiny App
Optimized with Local Flags, Value-Proportional Flag Badges, High-Contrast Palette, and Persistent Map Zoom
"""

import base64
import json
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
import branca.colormap as cm

WWW_FLAGS_DIR = Path(__file__).resolve().parent.parent / "www" / "flags"
FLAG_CDN = "https://flagcdn.com/w80"


def iso_to_flag_emoji(iso2c: str) -> str:
    """Convert a 2-letter ISO code (e.g., 'CN', 'US', 'MX') to a Unicode flag emoji."""
    if not iso2c or not isinstance(iso2c, str) or len(iso2c.strip()) != 2:
        return "🏳️"
    iso = iso2c.strip().upper()
    if not iso.isalpha():
        return "🏳️"
    return "".join(chr(127397 + ord(c)) for c in iso)


@lru_cache(maxsize=256)
def get_flag_url(iso2c: str, for_plotly: bool = False) -> str:
    """
    Return the local flag asset path or base64 data URI if downloaded in ./www/flags/,
    falling back to FlagCDN only if the local file does not exist.
    """
    iso = str(iso2c).strip().lower()
    local_file = WWW_FLAGS_DIR / f"{iso}.png"
    if local_file.exists():
        if for_plotly:
            b64 = base64.b64encode(local_file.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        return f"flags/{iso}.png"
    return f"{FLAG_CDN}/{iso}.png"


@lru_cache(maxsize=1)
def get_raw_dataframe(data_path: str = "./data/data.parquet") -> pd.DataFrame:
    """Read data.parquet into memory once and cache it."""
    return pd.read_parquet(data_path)


@lru_cache(maxsize=1)
def load_article_data() -> pd.DataFrame:
    """Load, rename, and cache the article-specific series from data.parquet."""
    try:
        df = get_raw_dataframe()
        article_columns_map = {
            'source': 'source',
            'year_x': 'year',
            'country_x': 'country',
            'percentage_x': 'value',
            'cc': 'cc'
        }
        required_raw_cols = list(article_columns_map.keys())
        if all(col in df.columns for col in required_raw_cols):
            article_data = (
                df[required_raw_cols]
                .copy()
                .rename(columns=article_columns_map)
                .dropna(subset=['value', 'source'])
            )
            article_data = article_data[article_data['source'] != ""]
            article_data['year'] = pd.to_numeric(article_data['year'], errors='coerce')
            article_data['value'] = pd.to_numeric(article_data['value'], errors='coerce')
            return article_data.dropna(subset=['year', 'value']).sort_values(['country', 'year'])
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in load_article_data: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def get_world_geometries(world_path: str = "./data/world_boundaries.geojson") -> dict:
    """Read world_boundaries.geojson using built-in json (no geopandas/fiona needed)."""
    iso_to_geometry = {}
    try:
        with open(world_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        for feature in geojson_data.get("features", []):
            props = feature.get("properties") or {}
            geom = feature.get("geometry")
            if not geom:
                continue

            for col in ['iso_a2_eh', 'wb_a2', 'iso_a2', 'ISO_A2', 'iso2c', 'ISO2C']:
                candidate = props.get(col)
                if isinstance(candidate, str) and len(candidate.strip()) == 2 and candidate.strip() != "-99":
                    iso_to_geometry[candidate.strip().upper()] = geom
                    break
    except Exception as e:
        print(f"Warning: Could not load {world_path}: {e}")

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
    """Optimized data fetching with early filtering."""
    if df.empty:
        return pd.DataFrame()

    if display_mode == "find_collaborations" and len(selected_isos) < 2:
        return pd.DataFrame()

    filtered_df = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1]) &
        (df['chemical'] == chemical_category)
    ]

    if filtered_df.empty:
        return pd.DataFrame()

    if display_mode in ["individual", "compare_individuals"]:
        if not selected_isos:
            return pd.DataFrame()

        result = filtered_df[
            (filtered_df['is_collab'] == False) &
            (filtered_df['iso2c'].isin(selected_isos))
        ].copy()

        if region_filter != "All" and 'region' in result.columns:
            result = result[result['region'] == region_filter]

        if result.empty:
            return pd.DataFrame()

        if country_list is not None:
            meta_cols = [c for c in ['iso2c', 'iso3c', 'country', 'cc', 'region'] if c in country_list.columns]
            country_meta = country_list[meta_cols].drop_duplicates(subset=['iso2c'])
            result = result.merge(country_meta, on='iso2c', how='left', suffixes=('_orig', '_meta'))
            for col in ['country', 'cc', 'region', 'iso3c']:
                meta_col = f'{col}_meta'
                if meta_col in result.columns:
                    result[col] = result[meta_col].fillna(result.get(f'{col}_orig', ''))

        result['plot_group'] = result.get('country', result['iso2c'])
        result['plot_color'] = result.get('cc', '#808080')

    elif display_mode == "find_collaborations":
        collab_df = filtered_df[filtered_df['is_collab'] == True].copy()
        if collab_df.empty:
            return pd.DataFrame()

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

        result['partners'] = result['iso2c'].str.split('-')
        result['collab_size'] = result['partners'].str.len()
        result['collab_type'] = result['collab_size'].map({
            2: "Bilateral",
            3: "Trilateral",
            4: "4-country"
        }).fillna("5-country+")

        result['plot_group'] = result['country']
        result['plot_color_group'] = result['collab_type']
    else:
        return pd.DataFrame()

    if not result.empty:
        result['year'] = pd.to_numeric(result['year'], errors='coerce')
        for col in ['percentage', 'value', 'value_raw', 'percentage_x']:
            if col in result.columns:
                result['total_percentage'] = pd.to_numeric(result[col], errors='coerce')
                break
        result = result.dropna(subset=['year', 'total_percentage'])

    return result


def create_empty_plot(message: str) -> go.Figure:
    """Create a clean empty placeholder plot with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=15, color="#555")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white',
        height=450
    )
    return fig


def add_flags_to_trend_plot(fig: go.Figure, data: pd.DataFrame, value_col: str = "total_percentage") -> go.Figure:
    """
    Attach larger national flag images at the latest year of each country's line trace,
    using vertical de-overlapping and column staggering when multiple countries cluster together.
    """
    if data.empty or 'year' not in data.columns or 'iso2c' not in data.columns:
        return fig

    last_year = int(data['year'].max())
    min_year = int(data['year'].min())
    year_span = max(last_year - min_year, 5)
    y_max = float(data[value_col].max()) if not data[value_col].dropna().empty else 10.0

    latest_points = (
        data.sort_values('year')
        .dropna(subset=['iso2c', value_col])
        .groupby('iso2c', as_index=False)
        .last()
        .sort_values(value_col, ascending=True)
        .reset_index(drop=True)
    )

    n_flags = len(latest_points)
    if n_flags == 0:
        return fig

    scale_factor = 1.0 if n_flags <= 6 else (0.85 if n_flags <= 12 else 0.72)
    sizex = max(1.15, year_span * 0.058 * scale_factor)
    sizey = max(0.9, y_max * 0.085 * scale_factor)

    use_two_cols = n_flags > 6
    min_vertical_gap = sizey * (0.58 if use_two_cols else 0.88)

    adjusted_y = []
    for _, row in latest_points.iterrows():
        raw_y = float(row[value_col])
        if not adjusted_y:
            adjusted_y.append(raw_y)
        else:
            prev_y = adjusted_y[-1]
            if raw_y - prev_y < min_vertical_gap:
                adjusted_y.append(prev_y + min_vertical_gap)
            else:
                adjusted_y.append(raw_y)

    images_to_add = []
    for idx, (_, row) in enumerate(latest_points.iterrows()):
        iso = str(row['iso2c']).strip().lower()
        if len(iso) != 2:
            continue
        col_offset = (idx % 2) * (sizex * 0.92) if use_two_cols else 0.0
        x_pos = float(row['year']) + 0.32 + col_offset
        y_pos = adjusted_y[idx]

        images_to_add.append(
            dict(
                source=get_flag_url(iso, for_plotly=True),
                xref="x",
                yref="y",
                x=x_pos,
                y=y_pos,
                sizex=sizex,
                sizey=sizey,
                xanchor="left",
                yanchor="middle",
                layer="above"
            )
        )

    right_pad = 3.4 if use_two_cols else 2.5
    fig.update_layout(
        images=images_to_add,
        xaxis=dict(range=[min_year - 0.5, last_year + right_pad])
    )
    return fig


def create_trends_plot(
    data: pd.DataFrame,
    selected_countries: List[str],
    mode: str,
    show_flags: bool = True
) -> go.Figure:
    """Create interactive trends plot with distinct per-line colors, dynamic legends, and zoom."""
    value_column = 'total_percentage'
    if value_column not in data.columns or data.empty:
        return create_empty_plot("No data available for the current selection.")

    fig = go.Figure()
    data_sorted = data.sort_values(['plot_group', 'year'])

    if mode == "compare_individuals":
        distinct_fallback = px.colors.qualitative.Bold + px.colors.qualitative.D3 + px.colors.qualitative.Dark24
        used_colors = set()

        country_order = (
            data_sorted.groupby('plot_group')[value_column]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        for idx, country_name_str in enumerate(country_order):
            country_name = str(country_name_str)
            c_df = data_sorted[data_sorted['plot_group'] == country_name]
            if c_df.empty:
                continue

            iso_code = c_df['iso2c'].iloc[0] if 'iso2c' in c_df.columns else ""
            flag_emoji = iso_to_flag_emoji(iso_code)
            color = None
            if 'plot_color' in c_df.columns:
                valid_colors = c_df['plot_color'].dropna()
                if not valid_colors.empty:
                    candidate_color = str(valid_colors.iloc[0])
                    if candidate_color not in ("#3b82f6", "#808080") or candidate_color not in used_colors:
                        color = candidate_color
            if not color or color in used_colors:
                color = distinct_fallback[idx % len(distinct_fallback)]
            used_colors.add(color)

            avg_val = float(c_df[value_column].mean())
            fig.add_trace(go.Scatter(
                x=c_df['year'],
                y=c_df[value_column],
                mode='lines+markers',
                name=f"{flag_emoji} {country_name} ({avg_val:.1f}%)",
                line=dict(color=color, width=2.4),
                marker=dict(symbol='circle', size=7, color=color, line=dict(width=1, color="white")),
                showlegend=True,
                hovertemplate=(
                    f"<b>{flag_emoji} {country_name} ({iso_code})</b><br>"
                    "Year: <b>%{x}</b><br>"
                    "CS Contribution: <b>%{y:.2f}%</b><extra></extra>"
                )
            ))

        if show_flags:
            fig = add_flags_to_trend_plot(fig, data_sorted, value_col=value_column)

    elif mode == "find_collaborations":
        distinct_palette = (
            px.colors.qualitative.Bold
            + px.colors.qualitative.D3
            + px.colors.qualitative.Dark24
        )
        collab_type_symbols = {
            "Bilateral": "circle",
            "Trilateral": "square",
            "4-country": "diamond",
            "5-country+": "triangle-up",
            "Unknown": "cross"
        }
        tier_symbol_variants = {
            "Bilateral": ["circle", "circle-open", "disc", "circle-dot"],
            "Trilateral": ["square", "square-open", "square-dot", "square-cross"],
            "4-country": ["diamond", "diamond-open", "diamond-dot", "diamond-tall"],
            "5-country+": ["triangle-up", "triangle-up-open", "star", "hexagram"],
            "Unknown": ["cross", "x"]
        }
        line_dashes = ["solid", "dash", "dot", "dashdot", "longdash"]

        collab_order = (
            data_sorted.groupby('plot_group')[value_column]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        tier_counters: Dict[str, int] = {}
        trace_tier_ranks: List[int] = []

        for overall_rank, collab_id_str in enumerate(collab_order):
            collab_id = str(collab_id_str)
            c_df = data_sorted[data_sorted['plot_group'] == collab_id]
            if c_df.empty:
                continue

            collab_type = "Unknown"
            if 'plot_color_group' in c_df.columns:
                valid_type = c_df['plot_color_group'].dropna()
                if not valid_type.empty:
                    collab_type = str(valid_type.iloc[0])

            tier_rank = tier_counters.get(collab_type, 0)
            tier_counters[collab_type] = tier_rank + 1
            trace_tier_ranks.append(tier_rank)

            trace_color = distinct_palette[tier_rank % len(distinct_palette)]
            dash_style = "solid" if tier_rank < 6 else line_dashes[(tier_rank // 6) % len(line_dashes)]
            sym_list = tier_symbol_variants.get(collab_type, [collab_type_symbols.get(collab_type, "circle")])
            marker_sym = sym_list[(tier_rank // 8) % len(sym_list)]

            display_name = collab_id
            short_iso_label = collab_id
            if 'iso2c' in c_df.columns:
                iso_codes_val = c_df['iso2c'].iloc[0]
                if isinstance(iso_codes_val, str) and '-' in iso_codes_val:
                    iso_codes = iso_codes_val.split('-')
                    highlighted = [
                        f"{iso_to_flag_emoji(code)} <b>{code}</b>" if code in selected_countries
                        else f"{iso_to_flag_emoji(code)} {code}"
                        for code in iso_codes
                    ]
                    display_name = " + ".join(highlighted)
                    short_iso_label = "-".join(iso_codes)

            avg_share = float(c_df[value_column].mean())
            peak_share = float(c_df[value_column].max())

            is_top_tier_trace = tier_rank < 8
            line_width = 2.6 if tier_rank < 3 else (1.9 if is_top_tier_trace else 1.1)
            marker_size = 8 if tier_rank < 3 else (6.5 if is_top_tier_trace else 5)
            trace_opacity = 0.95 if is_top_tier_trace else 0.55

            text_labels = [""] * len(c_df)
            if tier_rank < 3 and len(c_df) > 0:
                text_labels[-1] = f"  {short_iso_label}"

            fig.add_trace(go.Scatter(
                x=c_df['year'],
                y=c_df[value_column],
                mode='lines+markers+text' if tier_rank < 3 else 'lines+markers',
                name=f"{display_name} ({avg_share:.2f}%)",
                meta=collab_type,
                text=text_labels,
                textposition="middle right",
                textfont=dict(size=10, color=trace_color),
                opacity=trace_opacity,
                showlegend=(overall_rank < 12),
                line=dict(color=trace_color, width=line_width, dash=dash_style),
                marker=dict(
                    symbol=marker_sym,
                    size=marker_size,
                    color=trace_color,
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    f"<span style='color:#555;'>Tier: <b>{collab_type}</b> (Rank #{tier_rank + 1} in tier)</span><br>"
                    f"<span style='color:#666;'>Countries: {collab_id}</span><br>"
                    "Year: <b>%{x}</b><br>"
                    "Annual Share: <b>%{y:.3f}%</b><br>"
                    f"<span style='color:#666;'>Period Avg: {avg_share:.3f}% | Peak: {peak_share:.3f}%</span>"
                    "<extra></extra>"
                )
            ))

    if fig.data:
        for event_year, label in [(2008, "Financial Crisis (2008)"), (2020, "COVID-19 (2020)")]:
            fig.add_vline(x=event_year, line_dash="dot", line_color="rgba(120, 120, 120, 0.55)")
            fig.add_annotation(
                x=event_year,
                y=0.98,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                textangle=-90,
                yanchor="top",
                font=dict(size=10, color="#666"),
                xshift=-10
            )

    updatemenus = []
    if mode == "find_collaborations":
        type_order = ["Bilateral", "Trilateral", "4-country", "5-country+"]
        present_types = (
            data_sorted['plot_color_group'].dropna().unique().tolist()
            if 'plot_color_group' in data_sorted.columns else []
        )
        collab_types = [t for t in type_order if t in present_types] + [
            t for t in present_types if t not in type_order
        ]

        if collab_types:
            all_max = float(data_sorted[value_column].max()) if not data_sorted.empty else 1.0
            all_upper = max(all_max * 1.12, 0.01)

            all_legend_flags = [i < 12 for i in range(len(fig.data))]
            filter_buttons = [
                dict(
                    label=f"All Types ({len(fig.data)})",
                    method="update",
                    args=[
                        {"visible": [True] * len(fig.data), "showlegend": all_legend_flags},
                        {"yaxis.range": [0, all_upper], "yaxis.autorange": False}
                    ]
                )
            ]

            for c_type in collab_types:
                vis = [trace.meta == c_type for trace in fig.data]
                tier_legend_flags = [
                    (trace.meta == c_type and trace_tier_ranks[idx] < 12)
                    for idx, trace in enumerate(fig.data)
                ]
                tier_count = sum(vis)

                sub_vals = data_sorted[data_sorted['plot_color_group'] == c_type][value_column]
                type_max = float(sub_vals.max()) if not sub_vals.empty else 0.05
                type_upper = max(type_max * 1.15, 0.005)

                filter_buttons.append(
                    dict(
                        label=f"{c_type} ({tier_count})",
                        method="update",
                        args=[
                            {"visible": vis, "showlegend": tier_legend_flags},
                            {"yaxis.range": [0, type_upper], "yaxis.autorange": False}
                        ]
                    )
                )

            updatemenus.append(dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.5,
                y=1.13,
                xanchor="center",
                yanchor="top",
                buttons=filter_buttons,
                pad={"r": 6, "t": 6},
                bgcolor="rgba(245,247,250,0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1
            ))

    max_yr = int(data_sorted['year'].max()) if not data_sorted.empty else 2022
    min_yr = int(data_sorted['year'].min()) if not data_sorted.empty else 1996

    if mode != "compare_individuals" or not show_flags:
        fig.update_layout(
            xaxis=dict(range=[min_yr - 0.5, max_yr + (1.8 if mode == "find_collaborations" else 1.0)])
        )

    fig.update_layout(
        height=540 if mode == "compare_individuals" else 600,
        dragmode='zoom',
        yaxis=dict(
            ticksuffix='%',
            fixedrange=False,
            rangemode='tozero',
            title=dict(text="Percentage of New Substances", standoff=5, font=dict(size=12, color='black'))
        ),
        xaxis=dict(
            fixedrange=False,
            title=dict(text="Year (Drag on chart to zoom, double-click to reset)", standoff=5, font=dict(size=12, color='black'))
        ),
        hovermode='closest' if mode == "find_collaborations" else 'x unified',
        hoverdistance=25,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#cbd5e1",
            font_size=12,
            font_family="Segoe UI, Arial, sans-serif",
            align="left"
        ),
        template='plotly_white',
        modebar_remove=['lasso2d', 'select2d'],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11),
            title=dict(
                text="Selected Countries (avg share):" if mode == "compare_individuals" else "Top 12 in Active View (click to hide, double-click to isolate):",
                font=dict(size=11)
            )
        ),
        updatemenus=updatemenus,
        margin=dict(
            l=65,
            r=35,
            b=110,
            t=80 if (mode == "find_collaborations" and updatemenus) else 30
        )
    )
    return fig


def create_enhanced_folium_map(
    country_list: pd.DataFrame,
    selected_countries: List[str],
    contributions_df: Optional[pd.DataFrame] = None,
    map_style: str = "polygons"
) -> folium.Map:
    """
    Unified full-width Folium map supporting 3 visual modes with persistent zoom/pan memory:
    - 'bubbles': Superzip-style proportional bubbles where BOTH radius and color intensity encode CS share (%)
    - 'flags': Value-proportional circular national flag badges (major contributors stand out clearly)
    - 'polygons': Shaded GeoJSON country boundaries with light pastel low values and bold high values
    """
    if not country_list.empty:
        center_lat = float(country_list['lat'].mean())
        center_lng = float(country_list['lng'].mean())
        lat_range = float(country_list['lat'].max() - country_list['lat'].min())
        lng_range = float(country_list['lng'].max() - country_list['lng'].min())
        if lat_range > 60 or lng_range > 120:
            zoom_start = 2
        elif lat_range > 25 or lng_range > 50:
            zoom_start = 3
        else:
            zoom_start = 4
    else:
        center_lat, center_lng, zoom_start = 25.0, 10.0, 2

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True
    )

    # Preserved Humanitarian Map (HOT) tile layer as primary base map
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://hot.openstreetmap.org/">HOT</a>',
        name="Humanitarian Map (HOT)",
        max_zoom=19,
        control=True,
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="Standard OpenStreetMap",
        max_zoom=19,
        control=True,
        show=False
    ).add_to(m)

    avg_shares = {}
    max_shares = {}
    if contributions_df is not None and not contributions_df.empty:
        val_col = 'total_percentage' if 'total_percentage' in contributions_df.columns else 'percentage'
        if val_col in contributions_df.columns:
            grouped = contributions_df.groupby('iso2c')[val_col]
            avg_shares = grouped.mean().to_dict()
            max_shares = grouped.max().to_dict()

    max_avg_share = max(list(avg_shares.values()) + [1.0])

    # High-contrast colormap: Very Light Ice-Blue for low values -> Royal Blue -> Amber -> Deep Crimson Red
    colormap = cm.LinearColormap(
        colors=['#f0f9ff', '#bae6fd', '#60a5fa', '#2563eb', '#f59e0b', '#dc2626', '#991b1b'],
        vmin=0,
        vmax=np.sqrt(max_avg_share),
        caption="Avg Chemical Space Share (%) [√-scaled: Light = Low, Red = High]"
    )

    iso_to_geometry = get_world_geometries() if map_style == "polygons" else {}

    # Sort rows so smaller contributors render first and larger/selected contributors sit on top!
    country_rows = country_list.copy()
    country_rows['_avg_sort'] = country_rows['iso2c'].str.strip().str.upper().map(lambda x: avg_shares.get(x, 0.0))
    country_rows = country_rows.sort_values('_avg_sort', ascending=True)

    for _, row in country_rows.iterrows():
        iso = str(row['iso2c']).strip().upper()
        name = str(row['country'])
        lat, lng = float(row['lat']), float(row['lng'])
        region = str(row.get('region', 'Unknown'))
        avg_val = float(avg_shares.get(iso, 0.0))
        peak_val = float(max_shares.get(iso, 0.0))
        is_selected = iso in selected_countries
        flag_url = get_flag_url(iso, for_plotly=False)

        btn_bg = '#e74c3c' if is_selected else '#1e88e5'
        btn_label = '🗑️ Deselect Country' if is_selected else '✅ Select Country'

        popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; text-align: center; min-width: 175px; padding: 4px;">
            <img src="{flag_url}" width="52" style="border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); margin-bottom: 6px;"/><br/>
            <div style="font-size: 14px; font-weight: 700; color: #1f2937;">{name} ({iso})</div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 6px;">Region: <b>{region}</b></div>
            <div style="background: #f3f4f6; border-radius: 6px; padding: 6px; font-size: 11.5px; color: #111827; text-align: left;">
                <div>📊 <b>Period Avg:</b> {avg_val:.3f}%</div>
                <div>🚀 <b>Peak Share:</b> {peak_val:.3f}%</div>
            </div>
            <button onclick="
                if (window.parent && window.parent.Shiny) {{
                    window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                }} else if (window.Shiny) {{
                    window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                }}
            " style="
                margin-top: 8px; padding: 7px 12px; cursor: pointer;
                background-color: {btn_bg}; color: white; border: none;
                border-radius: 5px; font-weight: 600; font-size: 12px; width: 100%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            ">
                {btn_label}
            </button>
        </div>
        """

        tooltip_html = (
            f"<div style='font-family:sans-serif; font-size:12px;'>"
            f"<b>{iso_to_flag_emoji(iso)} {name} ({iso})</b><br/>"
            f"Avg CS Share: <b>{avg_val:.3f}%</b> | Peak: {peak_val:.3f}%<br/>"
            f"<i>Click marker to {'deselect' if is_selected else 'select'}</i></div>"
        )

        if map_style == "flags":
            # Value-Proportional Circular Flag Badges:
            # Tiny unselected contributors (< 0.15%) render as subtle 6px pins so they don't clutter Europe/Africa,
            # while contributors >= 0.15% (or selected countries) scale proportionally from 12px up to 54px!
            if avg_val < 0.15 and not is_selected:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=3.5,
                    color="#64748b",
                    weight=1,
                    fill=True,
                    fill_color="#bae6fd",
                    fill_opacity=0.55,
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(m)
            else:
                size_px = int(np.clip(11 + np.sqrt(max(avg_val, 0.15)) * 7.2, 12, 54))
                if is_selected and size_px < 22:
                    size_px = 22
                border_css = "3px solid #dc2626" if is_selected else "2px solid #ffffff"
                shadow_css = (
                    "0 0 10px rgba(220, 38, 38, 0.9)" if is_selected
                    else "0 2px 6px rgba(0,0,0,0.4)"
                )
                z_boost = 1000 if is_selected else int(avg_val * 20)

                badge_html = f"""
                <div style="
                    width: {size_px}px;
                    height: {size_px}px;
                    border-radius: 50%;
                    background-image: url('{flag_url}');
                    background-size: cover;
                    background-position: center;
                    border: {border_css};
                    box-shadow: {shadow_css};
                "></div>
                """
                folium.Marker(
                    location=[lat, lng],
                    z_index_offset=z_boost,
                    icon=folium.DivIcon(
                        icon_size=(size_px, size_px),
                        icon_anchor=(size_px // 2, size_px // 2),
                        html=badge_html
                    ),
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(m)

        elif map_style == "polygons":
            geometry = iso_to_geometry.get(iso)
            fill_col = colormap(np.sqrt(max(avg_val, 0.0))) if avg_val > 0 else "#f1f5f9"
            if geometry is not None:
                folium.GeoJson(
                    geometry,
                    style_function=lambda x, fc=fill_col, sel=is_selected: {
                        'fillColor': fc,
                        'color': '#dc2626' if sel else '#64748b',
                        'weight': 2.5 if sel else 0.7,
                        'fillOpacity': 0.85 if sel else 0.75
                    },
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=7 if is_selected else 4,
                    color='#dc2626' if is_selected else '#64748b',
                    weight=2 if is_selected else 1,
                    fill=True,
                    fill_color=fill_col,
                    fill_opacity=0.85,
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=240)
                ).add_to(m)

        else:
            # Value-Proportional Bubbles ('bubbles' mode)
            radius_px = (
                3.5 if (avg_val < 0.10 and not is_selected)
                else float(np.clip(5.5 + np.sqrt(max(avg_val, 0.05)) * 5.8, 6, 40))
            )
            fill_col = colormap(np.sqrt(max(avg_val, 0.0)))
            stroke_col = "#dc2626" if is_selected else "#334155"
            stroke_weight = 2.5 if is_selected else 1

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius_px,
                color=stroke_col,
                weight=stroke_weight,
                fill=True,
                fill_color=fill_col,
                fill_opacity=0.90 if is_selected else (0.50 if avg_val < 0.10 else 0.80),
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(m)

    if map_style in ["bubbles", "polygons"]:
        colormap.add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Preserve zoom & pan coordinates across country clicks and animation updates
    map_var = m.get_name()
    preserve_view_js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var checkMap = setInterval(function() {{
            if (typeof {map_var} !== "undefined") {{
                clearInterval(checkMap);
                var leafletMap = {map_var};
                var storeTarget = window.parent || window;
                if (storeTarget._cs_saved_map_view) {{
                    leafletMap.setView(
                        storeTarget._cs_saved_map_view.center,
                        storeTarget._cs_saved_map_view.zoom,
                        {{animate: false}}
                    );
                }}
                leafletMap.on("moveend zoomend", function() {{
                    var c = leafletMap.getCenter();
                    storeTarget._cs_saved_map_view = {{
                        center: [c.lat, c.lng],
                        zoom: leafletMap.getZoom()
                    }};
                }});
            }}
        }}, 40);
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(preserve_view_js))
    return m


# High-contrast Choropleth Color Scale:
# Very light ice-blue for low values (<1%) so high values (US Navy Blue / China Crimson Red) stand out!
GEOPOLITICAL_CS_COLORSCALE = [
    [0.00, "#f8fafc"],  # 0.00% - Very light neutral snow
    [0.04, "#e0f2fe"],  # ~0.03% - Pale ice blue
    [0.18, "#bae6fd"],  # ~0.20% - Soft sky blue
    [0.38, "#60a5fa"],  # ~1.0%  - Medium sky blue
    [0.62, "#2563eb"],  # ~4.5%  - Vibrant royal blue
    [0.76, "#1d4ed8"],  # ~10.0% - Strong blue
    [0.86, "#1e3a8a"],  # ~18–22% - Deep US Navy Blue (US 1996–2007 dominance)
    [0.92, "#f97316"],  # ~28.0% - Warm orange crossover (2013 transition)
    [0.96, "#dc2626"],  # ~35.0% - Vibrant China Red
    [1.00, "#991b1b"],  # ~41.0% - Deep China Crimson Red (China 2022 dominance)
]


def create_contribution_choropleth(data: pd.DataFrame) -> go.Figure:
    """Create smooth animated yearly world choropleth map with high-contrast light-to-dark palette."""
    if data.empty or 'total_percentage' not in data.columns:
        return create_empty_plot("No data available for choropleth")

    df_clean = data.dropna(subset=['total_percentage', 'year', 'iso3c']).copy()
    if df_clean.empty:
        return create_empty_plot("No valid country data available for choropleth")

    df_clean['year_int'] = df_clean['year'].round().astype(int)
    all_years = sorted(df_clean['year_int'].unique())
    country_meta = (
        df_clean[['iso3c', 'iso2c', 'country', 'region']]
        .drop_duplicates(subset=['iso3c'])
        .set_index('iso3c')
    )

    grid_index = pd.MultiIndex.from_product(
        [country_meta.index, all_years],
        names=['iso3c', 'year_int']
    ).to_frame().reset_index(drop=True)

    merged = (
        grid_index
        .merge(country_meta.reset_index(), on='iso3c', how='left')
        .merge(
            df_clean.groupby(['iso3c', 'year_int'], as_index=False)['total_percentage'].mean(),
            on=['iso3c', 'year_int'],
            how='left'
        )
    )
    merged['total_percentage'] = merged['total_percentage'].fillna(0.0)
    merged = merged.sort_values(['year_int', 'iso3c']).reset_index(drop=True)

    merged['Year'] = merged['year_int'].astype(str)
    merged['color_val'] = np.log10(1.0 + merged['total_percentage'].clip(lower=0) * 10.0)

    max_val = max(float(merged['total_percentage'].max()), 40.0)
    max_color_val = np.log10(1.0 + max_val * 10.0)
    tick_vals_real = [0, 0.1, 1, 5, 10, 20, 40]

    fig = px.choropleth(
        merged,
        locations="iso3c",
        color="color_val",
        hover_name="country",
        custom_data=["country", "iso2c", "Year", "total_percentage", "region"],
        animation_frame="Year",
        animation_group="iso3c",
        color_continuous_scale=GEOPOLITICAL_CS_COLORSCALE,
        range_color=[0, max_color_val],
        projection="robinson"
    )

    hover_tmpl = (
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
        "Year: <b>%{customdata[2]}</b><br>"
        "CS Share: <b>%{customdata[3]:.3f}%</b><br>"
        "Region: %{customdata[4]}<extra></extra>"
    )
    fig.update_traces(
        hovertemplate=hover_tmpl,
        marker_line_color="rgba(100,116,139,0.45)",
        marker_line_width=0.5
    )
    for frame in fig.frames:
        for trace in frame.data:
            trace.hovertemplate = hover_tmpl
            trace.marker.line.color = "rgba(100,116,139,0.45)"
            trace.marker.line.width = 0.5

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="CS Share", side="right", font=dict(size=10)),
            thickness=8,
            len=0.68,
            x=0.99,
            xpad=2,
            ypad=0,
            outlinewidth=0,
            tickfont=dict(size=9),
            tickvals=[np.log10(1.0 + v * 10.0) for v in tick_vals_real],
            ticktext=[f"{v}%" for v in tick_vals_real]
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.6)",
            showland=True,
            landcolor="#f1f5f9",
            lataxis_range=[-55, 85],
            projection_scale=1.08
        ),
        template='plotly_white',
        height=520,
        margin=dict(l=0, r=10, t=10, b=0)
    )

    if fig.layout.sliders:
        fig.layout.sliders[0].currentvalue = dict(
            prefix="Year: ",
            font=dict(size=14, color="#1e293b"),
            visible=True,
            xanchor="left"
        )
        fig.layout.sliders[0].pad = dict(t=5, b=5)
        fig.layout.sliders[0].len = 0.85
        fig.layout.sliders[0].x = 0.12
        fig.layout.sliders[0].transition = dict(duration=300, easing="cubic-in-out")

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].pad = dict(r=10, t=12)
        fig.layout.updatemenus[0].x = 0.02
        fig.layout.updatemenus[0].y = 0.05
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 320
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["easing"] = "cubic-in-out"

    return fig


def create_contribution_map_plot(
    processed_data_df: pd.DataFrame,
    fill_label: str = "Avg Share",
    main_title: str = "Average Contribution Over Selected Period"
) -> go.Figure:
    """Static choropleth map with matching high-contrast light-to-dark palette and slim colorbar."""
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
    )

    max_val = max(float(map_data['avg_percentage'].max()), 40.0)
    max_color_val = np.log10(1.0 + max_val * 10.0)
    tick_vals_real = [0, 0.1, 1, 5, 10, 20, 40]
    tick_vals_scaled = [np.log10(1.0 + v * 10.0) for v in tick_vals_real]

    loc_col = 'iso3c' if 'iso3c' in map_data.columns else 'country'
    loc_mode = 'ISO-3' if 'iso3c' in map_data.columns else 'country names'

    fig = go.Figure(data=go.Choropleth(
        locations=map_data[loc_col],
        z=np.log10(1.0 + map_data['avg_percentage'].clip(lower=0) * 10.0),
        locationmode=loc_mode,
        colorscale=GEOPOLITICAL_CS_COLORSCALE,
        zmin=0,
        zmax=max_color_val,
        marker_line_color="rgba(100,116,139,0.45)",
        marker_line_width=0.5,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Avg Contribution: <b>%{customdata[2]:.3f}%</b><br>"
            "Region: %{customdata[3]}<br>"
            "Range: %{customdata[4]:.3f}% – %{customdata[5]:.3f}%<br>"
            "Years: %{customdata[6]:.0f}–%{customdata[7]:.0f}<extra></extra>"
        ),
        customdata=map_data[[
            'country', 'iso2c', 'avg_percentage', 'region',
            'min_percentage', 'max_percentage', 'first_year', 'last_year'
        ]].values,
        colorbar=dict(
            title=dict(text=fill_label, side="right", font=dict(size=10)),
            thickness=8,
            len=0.68,
            x=0.99,
            xpad=2,
            outlinewidth=0,
            tickfont=dict(size=9),
            tickvals=tick_vals_scaled,
            ticktext=[f"{v}%" for v in tick_vals_real]
        )
    ))

    fig.update_layout(
        title={'text': f"<b>{main_title}</b>", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 13}},
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.6)",
            showland=True,
            landcolor="#f1f5f9",
            lataxis_range=[-55, 85],
            projection_type='robinson',
            projection_scale=1.08,
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        height=520,
        margin=dict(l=0, r=10, t=35, b=0)
    )
    return fig


def create_animated_bubble_figure(
    df: pd.DataFrame,
    chemical_category: str = "All",
    top_n: int = 20
) -> go.Figure:
    """Client-side 60fps animated Bubble & Flag Explorer across 1996-2022."""
    sub = df[(df['is_collab'] == False) & (df['chemical'] == chemical_category)].copy()
    if sub.empty:
        return create_empty_plot("No data available for animated bubble chart.")

    val_col = 'percentage' if 'percentage' in sub.columns else 'value'
    sub['year'] = pd.to_numeric(sub['year'], errors='coerce')
    sub[val_col] = pd.to_numeric(sub[val_col], errors='coerce')
    sub = sub.dropna(subset=['year', val_col, 'country'])
    sub['region'] = sub['region'].fillna('Other')

    top_countries = (
        sub.groupby('country')[val_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    anim_df = sub[sub['country'].isin(top_countries)].copy()
    anim_df['year'] = anim_df['year'].astype(int)
    anim_df = anim_df.sort_values(['year', val_col], ascending=[True, False])

    anim_df['yearly_rank'] = anim_df.groupby('year')[val_col].rank(ascending=False, method='first')
    anim_df['label_with_flag'] = anim_df.apply(
        lambda r: f"{iso_to_flag_emoji(str(r.get('iso2c', '')))} {r['country']}", axis=1
    )
    anim_df['bubble_size'] = np.sqrt(anim_df[val_col].clip(lower=0.2)) * 8 + 8

    max_y = max(float(anim_df[val_col].max()) * 1.15, 10.0)

    fig = px.scatter(
        anim_df,
        x="yearly_rank",
        y=val_col,
        animation_frame="year",
        animation_group="country",
        size="bubble_size",
        color="region",
        hover_name="label_with_flag",
        text="label_with_flag",
        custom_data=["country", "iso2c", "year", val_col, "yearly_rank", "region"],
        size_max=48,
        range_x=[0.3, top_n + 0.8],
        range_y=[0, max_y],
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    hover_tmpl = (
        "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
        "Year: %{customdata[2]}<br>"
        "Global Rank: <b>#%{customdata[4]:.0f}</b><br>"
        "CS Contribution: <b>%{customdata[3]:.2f}%</b><br>"
        "Region: %{customdata[5]}<extra></extra>"
    )
    fig.update_traces(textposition="top center", textfont=dict(size=10), hovertemplate=hover_tmpl)
    for frame in fig.frames:
        for trace in frame.data:
            trace.hovertemplate = hover_tmpl
            trace.textposition = "top center"

    fig.update_layout(
        title=dict(
            text=f"<b>Animated Global Ranking & Bubble Share (1996–2022) — {chemical_category}</b>",
            x=0.5, xanchor="center"
        ),
        xaxis=dict(title="Global Contributor Rank (1 = Leader)", tickmode="linear", tick0=1, dtick=1, autorange=False),
        yaxis=dict(title="Share of New Substances (%)", ticksuffix="%", autorange=False),
        template="plotly_white",
        height=560,
        legend=dict(title=dict(text="Region"), orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 550
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

    return fig


def create_article_plot(data: pd.DataFrame, title: str, show_exp_fit: bool = True) -> go.Figure:
    """Create main article plot (Figure 1a) with China's exponential growth fit and 2013 crossover marker."""
    if data.empty:
        return create_empty_plot("No country participation data available")

    fig = go.Figure()
    min_year = int(data['year'].min())
    max_year = int(data['year'].max())

    for country in data['country'].unique():
        c_df = data[data['country'] == country].sort_values('year')
        color = c_df['cc'].iloc[0] if 'cc' in c_df.columns and not c_df['cc'].dropna().empty else None
        fig.add_trace(go.Scatter(
            x=c_df['year'],
            y=c_df['value'],
            mode='lines+markers',
            name=country,
            line=dict(color=color, width=2) if color else dict(width=2),
            marker=dict(
                size=c_df['value'].abs().clip(lower=4, upper=11),
                opacity=0.8,
                color=color if color else '#1f77b4'
            ),
            hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Share: <b>%{{y:.2f}}%</b><extra></extra>"
        ))

    if show_exp_fit:
        years_arr = np.arange(1996, max_year + 1)
        exp_fit = 1.33 * np.exp(0.1372 * (years_arr - 1996))
        fig.add_trace(go.Scatter(
            x=years_arr,
            y=exp_fit,
            mode='lines',
            name='China Exp. Fit (R² = 0.98)',
            line=dict(color='#c5051b', width=1.5, dash='dot'),
            hovertemplate="<b>China Exponential Fit</b><br>1.33·e^{0.1372(t-1996)}<br>Year: %{x}<br>Fit: %{y:.2f}%<extra></extra>"
        ))

    fig.add_vline(x=2013, line_dash="dash", line_color="rgba(197, 5, 27, 0.45)")
    fig.add_annotation(
        x=2013, y=38,
        text="<b>2013:</b> China becomes #1 contributor",
        showarrow=True, arrowhead=2, ax=-60, ay=-25,
        font=dict(size=11, color="#c5051b"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#c5051b", borderwidth=1
    )

    fig.update_layout(
        height=520,
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="Share of New Substances (%)", standoff=1, font=dict(size=12, color='black'))
        ),
        xaxis=dict(
            autorange=False,
            range=[min_year - 1, max_year + 1],
            title=dict(text="Year", standoff=1, font=dict(size=12, color='black'))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, traceorder="reversed"),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale'],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=2,
                x=1, y=1.15,
                buttons=[
                    dict(label="1996–2012", method="relayout", args=[{"xaxis.range": [1996, 2013]}]),
                    dict(label="2012–2022", method="relayout", args=[{"xaxis.range": [max_year - 10, max_year + 1]}]),
                    dict(label="All Years", method="relayout", args=[{"xaxis.range": [min_year - 1, max_year + 1]}])
                ]
            )
        ]
    )
    return fig


def create_top_trends_plot(data: pd.DataFrame, title: str) -> go.Figure:
    """Create top 10 contributors/collaborations plot."""
    fig = go.Figure()
    avg_percentages = data.groupby('country')['percentage'].mean().sort_values(ascending=True)

    for entity in avg_percentages.index:
        entity_data = data[data['country'] == entity].sort_values('year')
        avg_value = avg_percentages[entity]
        color = entity_data['cc'].iloc[0] if 'cc' in entity_data.columns and not entity_data['cc'].dropna().empty else None

        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data['percentage'],
            mode='lines+markers',
            name=f"{entity} ({avg_value:.2f}%)",
            line=dict(width=2, color=color) if color else dict(width=2),
            marker=dict(color=color if color else '#1f77b4', size=6),
            hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Contribution: <b>%{y:.2f}%</b><extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        height=520,
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="New Substances (%)", standoff=1, font=dict(size=12, color='black'))
        ),
        xaxis=dict(
            fixedrange=True,
            title=dict(text="Year", standoff=1, font=dict(size=12, color='black'))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, traceorder="reversed"),
        hovermode='closest',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    return fig


def create_summary_dataframe(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Create summary statistics table with flag emojis."""
    value_column = 'total_percentage'
    if value_column not in data.columns or data.empty:
        return pd.DataFrame({'Message': ["No data available for summary."]})

    if mode == "find_collaborations":
        required_cols = ['plot_group', 'chemical', 'collab_type', value_column]
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame({'Error': ["Summary data missing required columns."]})

        summary = (
            data.groupby(['plot_group', 'chemical', 'collab_type'], as_index=False)
            .agg(
                avg_percentage=(value_column, 'mean'),
                max_percentage=(value_column, 'max'),
                years_present=('year', 'nunique')
            )
            .round(3)
            .sort_values('avg_percentage', ascending=False)
        )
        summary.columns = ['Collaboration', 'Chemical Subspace', 'Type', 'Avg Share (%)', 'Peak Share (%)', 'Active Years']

    elif mode == "compare_individuals":
        required_cols = ['country', 'iso2c', 'chemical', value_column]
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame({'Error': ["Summary data missing required columns."]})

        summary = (
            data.groupby(['country', 'iso2c', 'chemical'], as_index=False)
            .agg(
                avg_percentage=(value_column, 'mean'),
                max_percentage=(value_column, 'max'),
                years_present=('year', 'nunique')
            )
            .round(3)
            .sort_values('avg_percentage', ascending=False)
        )
        summary['country'] = summary.apply(
            lambda r: f"{iso_to_flag_emoji(r['iso2c'])} {r['country']}", axis=1
        )
        summary.columns = ['Country', 'ISO', 'Chemical Subspace', 'Avg Share (%)', 'Peak Share (%)', 'Active Years']
    else:
        return pd.DataFrame({'Message': [f"Summary not available for mode: {mode}"]})

    return summary


def create_gdp_plot(data: pd.DataFrame) -> go.Figure:
    """Create GDP per capita annual growth plot (Figure 1d)."""
    fig = go.Figure()
    for country in data['country'].unique():
        c_df = data[data['country'] == country].sort_values('year')
        color = c_df['cc'].iloc[0] if 'cc' in c_df.columns and not c_df['cc'].dropna().empty else 'red'
        fig.add_trace(go.Scatter(
            x=c_df['year'],
            y=c_df['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2.5, color=color),
            marker=dict(size=c_df['value'].abs().clip(upper=14) + 3, color=color)
        ))

    fig.add_vline(x=2007.5, line_dash="dash", line_color="grey")
    fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")

    if fig.data:
        max_val = max([max(trace['y']) for trace in fig.data if len(trace['y']) > 0])
        fig.add_annotation(x=2007.5, y=max_val * 0.9, text="Global Financial Crisis", showarrow=True, arrowhead=1)
        fig.add_annotation(x=2019.5, y=max_val * 0.8, text="COVID-19", showarrow=True, arrowhead=1)

    fig.update_layout(
        height=480,
        yaxis=dict(ticksuffix='%', fixedrange=True, title=dict(text="GDP per Capita Growth (Annual %)", font=dict(size=12))),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, traceorder="reversed"),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    return fig


def create_researchers_plot(data: pd.DataFrame) -> go.Figure:
    """Create researchers headcount plot (Figure 1e) with 2005 China-US crossover marker."""
    fig = go.Figure()
    for country in data['country'].unique():
        c_df = data[data['country'] == country].sort_values('year')
        scaled_values = c_df['value'] / 1e6
        color = c_df['cc'].iloc[0] if 'cc' in c_df.columns and not c_df['cc'].dropna().empty else 'red'

        fig.add_trace(go.Scatter(
            x=c_df['year'],
            y=scaled_values,
            mode='lines+markers',
            name=country,
            line=dict(width=2.5, color=color),
            marker=dict(size=scaled_values.abs().clip(lower=4, upper=12) + 2, color=color)
        ))

    fig.add_vline(x=2005, line_dash="dot", line_color="rgba(100,100,100,0.6)")
    fig.add_annotation(
        x=2005, y=2.1, text="2005: China surpasses US in R&D researchers",
        showarrow=True, arrowhead=2, ax=70, ay=-20, font=dict(size=10)
    )

    fig.update_layout(
        height=480,
        yaxis=dict(ticksuffix='M', fixedrange=True, title=dict(text="Number of Researchers (Millions)", font=dict(size=12))),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, traceorder="reversed"),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    return fig


def create_cs_expansion_plot(data: pd.DataFrame) -> go.Figure:
    """Create Chemical Space expansion plot (Figure 2a) on a logarithmic scale."""
    fig = go.Figure()
    for country in data['country'].unique():
        c_df = data[data['country'] == country].sort_values('year')
        fig.add_trace(go.Scatter(
            x=c_df['year'],
            y=c_df['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2.5),
            marker=dict(size=5)
        ))

    fig.add_vline(x=2008, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")
    fig.add_vline(x=2020, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")

    fig.update_layout(
        height=480,
        yaxis=dict(
            fixedrange=True,
            type="log",
            title=dict(text="Number of New Substances (Log Scale)", font=dict(size=12))
        ),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, traceorder="reversed"),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    return fig


def create_china_us_dual_axis_plot(data: pd.DataFrame) -> go.Figure:
    """Create dual-axis plot for China-US domestic vs. bilateral collaboration (Figure 1c)."""
    fig = go.Figure()
    secondary_yaxis_categories = ["CN-US collab/US", "CN-US collab/CN"]
    colors = {
        "United States": "#002852",
        "China": "#c5051b",
        "CN-US collab/US": "#5b9bd5",
        "CN-US collab/CN": "#f17c67"
    }

    for country_name in data['country'].unique():
        c_df = data[data['country'] == country_name].sort_values('year')
        trace_params = dict(
            x=c_df['year'],
            y=c_df['value'],
            mode='lines+markers',
            name=country_name,
            line=dict(width=2.5, color=colors.get(country_name)),
            marker=dict(size=6, color=colors.get(country_name))
        )
        if country_name in secondary_yaxis_categories:
            trace_params['yaxis'] = 'y2'
        fig.add_trace(go.Scatter(**trace_params))

    fig.add_vrect(
        x0=2009, x1=2017,
        fillcolor="rgba(91, 155, 213, 0.08)", opacity=1.0,
        layer="below", line_width=0,
        annotation_text="2009–2017: US–China Bilateral Science Agreements",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#444")
    )
    fig.add_vline(x=2018, line_dash="dot", line_color="rgba(180, 50, 50, 0.5)")
    fig.add_annotation(
        x=2018, y=97, text="2018: US China Initiative",
        showarrow=False, textangle=-90, font=dict(size=10, color="#a02020"), xshift=10
    )

    fig.update_layout(
        height=520,
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="Domestic Solo Share of National CS (%)", font=dict(size=12))
        ),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=12))),
        yaxis2=dict(
            overlaying='y', side='right', ticksuffix='%', fixedrange=True,
            title=dict(text="China–US Bilateral Share of National CS (%)", font=dict(size=11))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'lasso', 'select', 'zoomIn', 'zoomOut', 'autoScale']
    )
    return fig