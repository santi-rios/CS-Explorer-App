"""
Helper functions for the Chemical Space Explorer Python Shiny App.
Includes responsive desktop/mobile Plotly layouts, fixed collaboration matching,
touch-friendly Folium maps, and enriched Top 10 Global Trends analytics.
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


def format_entity_with_flags(entity_name: str, iso2c_val: str = "") -> str:
    """Attach flag emoji(s) to either a single country or a hyphenated collaboration."""
    if not isinstance(iso2c_val, str) or not iso2c_val.strip():
        return str(entity_name)
    parts = [p.strip().upper() for p in iso2c_val.replace("_", "-").split("-") if p.strip()]
    if not parts:
        return str(entity_name)
    if len(parts) == 1:
        return f"{iso_to_flag_emoji(parts[0])} {entity_name}"
    flags = "".join(iso_to_flag_emoji(p) for p in parts[:4])
    if len(parts) > 4:
        flags += "+"
    return f"{flags} {entity_name}"


@lru_cache(maxsize=256)
def get_flag_url(iso2c: str, for_plotly: bool = False) -> str:
    """Return local flag asset path or base64 data URI, falling back to FlagCDN."""
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
    """Read world_boundaries.geojson using built-in json."""
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


def _parse_iso_partners(iso_string) -> List[str]:
    """Parse hyphen/underscore-separated ISO2 codes from a collaboration row."""
    if pd.isna(iso_string):
        return []
    raw = str(iso_string).replace("_", "-").replace(",", "-")
    return [p.strip().upper() for p in raw.split("-") if p.strip()]


def get_display_data(
    df: pd.DataFrame,
    selected_isos: List[str],
    year_range: Tuple[int, int],
    chemical_category: str,
    display_mode: str,
    region_filter: str = "All",
    country_list: pd.DataFrame = None,
    collab_match_mode: str = "any_pair",
    collab_tier_filter: str = "All",
) -> pd.DataFrame:
    """
    Optimized data fetching with early filtering and flexible collaboration matching.

    collab_match_mode options:
      - 'any_pair' (default): collaborations containing >=2 selected countries
        (or containing the single selected country if only 1 is chosen).
      - 'only_selected': collaborations whose partners are strictly a subset of selected countries.
      - 'all_selected': collaborations that include EVERY selected country simultaneously.
    """
    if df.empty or not selected_isos:
        return pd.DataFrame()

    norm_selected = [str(s).strip().upper() for s in selected_isos if s and str(s).strip()]
    if not norm_selected:
        return pd.DataFrame()

    filtered_df = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1]) &
        (df['chemical'] == chemical_category)
    ]
    if filtered_df.empty:
        return pd.DataFrame()

    if display_mode in ["individual", "compare_individuals"]:
        result = filtered_df[
            (filtered_df['is_collab'] == False) &
            (filtered_df['iso2c'].astype(str).str.strip().str.upper().isin(norm_selected))
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

        selected_set = set(norm_selected)
        n_sel = len(selected_set)

        # Fallback mapping if any collaboration row lacks iso2c but has country names
        name_to_iso = {}
        if country_list is not None and {'country', 'iso2c'}.issubset(country_list.columns):
            name_to_iso = dict(zip(
                country_list['country'].astype(str).str.strip().str.lower(),
                country_list['iso2c'].astype(str).str.strip().str.upper()
            ))

        def extract_partner_list(row) -> List[str]:
            partners = _parse_iso_partners(row.get('iso2c'))
            if len(partners) >= 2:
                return partners
            # Fallback to splitting country string if iso2c is missing/malformed
            cname = row.get('country')
            if isinstance(cname, str) and '-' in cname and name_to_iso:
                mapped = [name_to_iso.get(p.strip().lower(), "") for p in cname.split('-')]
                mapped = [m for m in mapped if m]
                if len(mapped) >= 2:
                    return mapped
            return partners

        partners_series = collab_df.apply(extract_partner_list, axis=1)

        def matches_selection(partner_list: List[str]) -> bool:
            if not partner_list:
                return False
            pset = set(partner_list)
            if n_sel == 1:
                return selected_set.issubset(pset)
            if collab_match_mode == "all_selected":
                return selected_set.issubset(pset)
            if collab_match_mode == "only_selected":
                return len(pset) >= 2 and pset.issubset(selected_set)
            # Default: 'any_pair' -> at least 2 of the selected countries collaborate together
            return len(selected_set.intersection(pset)) >= 2

        mask = partners_series.apply(matches_selection)
        result = collab_df[mask].copy()
        if result.empty:
            return pd.DataFrame()

        result['partners'] = partners_series[mask]
        result['collab_size'] = result['partners'].str.len()
        result['collab_type'] = result['collab_size'].map({
            2: "Bilateral",
            3: "Trilateral",
            4: "4-country"
        }).fillna("5-country+")

        if collab_tier_filter and collab_tier_filter != "All":
            result = result[result['collab_type'] == collab_tier_filter]
            if result.empty:
                return pd.DataFrame()

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


def apply_responsive_layout(
    fig: go.Figure,
    is_mobile: bool = False,
    desktop_height: int = 490,
    mobile_height: int = 380,
    top_desktop: int = 40,
    top_mobile: int = 36,
    bottom_desktop: int = 90,
    bottom_mobile: int = 105,
    allow_desktop_zoom: bool = False,
) -> go.Figure:
    """Apply consistent responsive heights, margins, typography, and touch-safe behavior."""
    if is_mobile:
        fig.update_layout(
            autosize=True,
            height=mobile_height,
            font=dict(size=10),
            dragmode=False,
            margin=dict(l=40, r=10, t=top_mobile, b=bottom_mobile),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=9),
            ),
        )
        fig.update_xaxes(title_font=dict(size=10), tickfont=dict(size=9), automargin=True)
        fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=9), automargin=True)
    else:
        fig.update_layout(
            autosize=True,
            height=desktop_height,
            dragmode='zoom' if allow_desktop_zoom else False,
            margin=dict(l=58, r=24, t=top_desktop, b=bottom_desktop),
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
    return fig


def create_empty_plot(message: str, is_mobile: bool = False) -> go.Figure:
    """Create a clean empty placeholder plot with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message.replace("\n", "<br>"),
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=12 if is_mobile else 14, color="#475569")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_white',
        height=300 if is_mobile else 420,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def add_flags_to_trend_plot(
    fig: go.Figure,
    data: pd.DataFrame,
    value_col: str = "total_percentage",
    is_mobile: bool = False
) -> go.Figure:
    """Attach national flag images at the latest year of each country's line trace."""
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

    # On narrow mobile viewports, keep flags slightly smaller and cap clutter
    mob_mult = 0.78 if is_mobile else 1.0
    scale_factor = (1.0 if n_flags <= 6 else (0.85 if n_flags <= 12 else 0.72)) * mob_mult
    sizex = max(1.0, year_span * 0.056 * scale_factor)
    sizey = max(0.85, y_max * 0.085 * scale_factor)

    use_two_cols = n_flags > 6 and not is_mobile
    min_vertical_gap = sizey * (0.58 if use_two_cols else 0.85)

    adjusted_y = []
    for _, row in latest_points.iterrows():
        raw_y = float(row[value_col])
        if not adjusted_y:
            adjusted_y.append(raw_y)
        else:
            prev_y = adjusted_y[-1]
            adjusted_y.append(prev_y + min_vertical_gap if raw_y - prev_y < min_vertical_gap else raw_y)

    images_to_add = []
    for idx, (_, row) in enumerate(latest_points.iterrows()):
        iso = str(row['iso2c']).strip().lower()
        if len(iso) != 2:
            continue
        col_offset = (idx % 2) * (sizex * 0.92) if use_two_cols else 0.0
        x_pos = float(row['year']) + 0.30 + col_offset
        y_pos = adjusted_y[idx]

        images_to_add.append(dict(
            source=get_flag_url(iso, for_plotly=True),
            xref="x", yref="y",
            x=x_pos, y=y_pos,
            sizex=sizex, sizey=sizey,
            xanchor="left", yanchor="middle",
            layer="above"
        ))

    right_pad = 3.3 if use_two_cols else (2.1 if is_mobile else 2.4)
    fig.update_layout(
        images=images_to_add,
        xaxis=dict(range=[min_year - 0.5, last_year + right_pad])
    )
    return fig


def create_trends_plot(
    data: pd.DataFrame,
    selected_countries: List[str],
    mode: str,
    show_flags: bool = True,
    log_scale: bool = False,
    show_events: bool = True,
    max_collab_traces: int = 15,
    is_mobile: bool = False,
) -> go.Figure:
    """Create responsive interactive trends plot for individual countries or collaborations."""
    value_column = 'total_percentage'
    if value_column not in data.columns or data.empty:
        return create_empty_plot("No data available for the current selection.", is_mobile=is_mobile)

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

            iso_code = str(c_df['iso2c'].iloc[0]) if 'iso2c' in c_df.columns else ""
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

            y_vals = c_df[value_column].where(c_df[value_column] > 0) if log_scale else c_df[value_column]
            avg_val = float(c_df[value_column].mean())
            fig.add_trace(go.Scatter(
                x=c_df['year'],
                y=y_vals,
                mode='lines+markers',
                name=f"{flag_emoji} {country_name} ({avg_val:.1f}%)",
                line=dict(color=color, width=2.2 if not is_mobile else 1.9),
                marker=dict(
                    symbol='circle',
                    size=6.5 if not is_mobile else 5,
                    color=color,
                    line=dict(width=1, color="white")
                ),
                showlegend=True,
                hovertemplate=(
                    f"<b>{flag_emoji} {country_name} ({iso_code})</b><br>"
                    "Year: <b>%{x}</b><br>"
                    "CS Contribution: <b>%{y:.2f}%</b><extra></extra>"
                )
            ))

        if show_flags and not log_scale:
            fig = add_flags_to_trend_plot(fig, data_sorted, value_col=value_column, is_mobile=is_mobile)

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
        line_dashes = ["solid", "dash", "dot", "dashdot"]

        collab_order = (
            data_sorted.groupby('plot_group')[value_column]
            .mean()
            .sort_values(ascending=False)
            .head(max_collab_traces)
            .index.tolist()
        )

        selected_upper = {str(s).strip().upper() for s in selected_countries}

        for rank_idx, collab_id_str in enumerate(collab_order):
            collab_id = str(collab_id_str)
            c_df = data_sorted[data_sorted['plot_group'] == collab_id]
            if c_df.empty:
                continue

            collab_type = "Unknown"
            if 'plot_color_group' in c_df.columns:
                valid_type = c_df['plot_color_group'].dropna()
                if not valid_type.empty:
                    collab_type = str(valid_type.iloc[0])

            trace_color = distinct_palette[rank_idx % len(distinct_palette)]
            dash_style = "solid" if rank_idx < 8 else line_dashes[(rank_idx // 8) % len(line_dashes)]
            marker_sym = collab_type_symbols.get(collab_type, "circle")

            display_name = collab_id
            short_iso_label = collab_id
            if 'iso2c' in c_df.columns:
                iso_codes = _parse_iso_partners(c_df['iso2c'].iloc[0])
                if iso_codes:
                    highlighted = [
                        f"{iso_to_flag_emoji(code)}<b>{code}</b>" if code in selected_upper
                        else f"{iso_to_flag_emoji(code)}{code}"
                        for code in iso_codes
                    ]
                    display_name = " + ".join(highlighted)
                    short_iso_label = "-".join(iso_codes)

            avg_share = float(c_df[value_column].mean())
            peak_share = float(c_df[value_column].max())
            y_vals = c_df[value_column].where(c_df[value_column] > 0) if log_scale else c_df[value_column]

            is_top3 = rank_idx < 3
            line_width = 2.5 if is_top3 else (1.8 if rank_idx < 8 else 1.2)
            marker_size = (7.5 if is_top3 else 5.5) if not is_mobile else 5

            text_labels = [""] * len(c_df)
            if is_top3 and not is_mobile and len(c_df) > 0:
                text_labels[-1] = f"  {short_iso_label}"

            fig.add_trace(go.Scatter(
                x=c_df['year'],
                y=y_vals,
                mode='lines+markers+text' if (is_top3 and not is_mobile) else 'lines+markers',
                name=f"{display_name} ({avg_share:.2f}%)",
                meta=collab_type,
                text=text_labels,
                textposition="middle right",
                textfont=dict(size=10, color=trace_color),
                opacity=0.95 if rank_idx < 8 else 0.72,
                showlegend=(rank_idx < (8 if is_mobile else 12)),
                line=dict(color=trace_color, width=line_width, dash=dash_style),
                marker=dict(
                    symbol=marker_sym,
                    size=marker_size,
                    color=trace_color,
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    f"<span style='color:#555;'>Tier: <b>{collab_type}</b> (Rank #{rank_idx + 1})</span><br>"
                    f"<span style='color:#666;'>{collab_id}</span><br>"
                    "Year: <b>%{x}</b><br>"
                    "Annual Share: <b>%{y:.3f}%</b><br>"
                    f"<span style='color:#666;'>Avg: {avg_share:.3f}% | Peak: {peak_share:.3f}%</span>"
                    "<extra></extra>"
                )
            ))

    if fig.data and show_events:
        min_yr_data = int(data_sorted['year'].min())
        max_yr_data = int(data_sorted['year'].max())
        for event_year, label in [(2008, "2008 Crisis"), (2020, "COVID-19")]:
            if min_yr_data <= event_year <= max_yr_data:
                fig.add_vline(x=event_year, line_dash="dot", line_color="rgba(120, 120, 120, 0.5)")
                fig.add_annotation(
                    x=event_year, y=0.98, xref="x", yref="paper",
                    text=label, showarrow=False, textangle=-90,
                    yanchor="top", font=dict(size=9 if is_mobile else 10, color="#64748b"),
                    xshift=-9
                )

    max_yr = int(data_sorted['year'].max()) if not data_sorted.empty else 2022
    min_yr = int(data_sorted['year'].min()) if not data_sorted.empty else 1996

    if mode != "compare_individuals" or not show_flags or log_scale:
        pad_right = 0.8 if is_mobile else (1.8 if mode == "find_collaborations" else 1.0)
        fig.update_layout(xaxis=dict(range=[min_yr - 0.5, max_yr + pad_right]))

    yaxis_cfg = dict(
        ticksuffix='%',
        type='log' if log_scale else 'linear',
        fixedrange=is_mobile,
        title=dict(
            text="New Substances (%)" if is_mobile else "Percentage of New Substances (%)",
            standoff=4,
            font=dict(size=10 if is_mobile else 12, color='#0f172a')
        )
    )
    if not log_scale:
        yaxis_cfg['rangemode'] = 'tozero'

    fig.update_layout(
        yaxis=yaxis_cfg,
        xaxis=dict(
            fixedrange=is_mobile,
            title=dict(
                text="Year" if is_mobile else "Year (Drag to zoom; double-click to reset)",
                standoff=4,
                font=dict(size=10 if is_mobile else 12, color='#0f172a')
            )
        ),
        hovermode='closest' if (mode == "find_collaborations" or is_mobile) else 'x unified',
        hoverdistance=25,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#cbd5e1",
            font_size=11 if is_mobile else 12,
            font_family="Segoe UI, Arial, sans-serif",
            align="left"
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(size=10 if not is_mobile else 9),
        ),
    )

    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=495 if mode == "compare_individuals" else 520,
        mobile_height=375,
        top_desktop=22,
        top_mobile=18,
        bottom_desktop=95,
        bottom_mobile=110,
        allow_desktop_zoom=True,
    )


def create_enhanced_folium_map(
    country_list: pd.DataFrame,
    selected_countries: List[str],
    contributions_df: Optional[pd.DataFrame] = None,
    map_style: str = "polygons",
    is_mobile: bool = False,
) -> folium.Map:
    """
    Touch-friendly Folium map supporting 3 visual modes with persistent zoom/pan memory.
    Disabled scrollWheelZoom prevents accidental scroll-trapping on desktop trackpads and phones.
    """
    if not country_list.empty:
        center_lat = float(country_list['lat'].mean())
        center_lng = float(country_list['lng'].mean())
        lat_range = float(country_list['lat'].max() - country_list['lat'].min())
        lng_range = float(country_list['lng'].max() - country_list['lng'].min())
        if lat_range > 60 or lng_range > 120:
            zoom_start = 1 if is_mobile else 2
        elif lat_range > 25 or lng_range > 50:
            zoom_start = 2 if is_mobile else 3
        else:
            zoom_start = 3 if is_mobile else 4
    else:
        center_lat, center_lng, zoom_start = 22.0, 10.0, (1 if is_mobile else 2)

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        scrollWheelZoom=False,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr='&copy; OpenStreetMap contributors &copy; HOT',
        name="Humanitarian Map (HOT)",
        max_zoom=18,
        control=True,
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; OpenStreetMap contributors',
        name="Standard OpenStreetMap",
        max_zoom=18,
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

    colormap = cm.LinearColormap(
        colors=['#f0f9ff', '#bae6fd', '#60a5fa', '#2563eb', '#f59e0b', '#dc2626', '#991b1b'],
        vmin=0,
        vmax=np.sqrt(max_avg_share),
        caption="Avg CS Share (%) [√-scale]"
    )

    iso_to_geometry = get_world_geometries() if map_style == "polygons" else {}

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

        btn_bg = '#e74c3c' if is_selected else '#0284c7'
        btn_label = '🗑️ Deselect' if is_selected else '✅ Select Country'

        popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; text-align: center; width: 170px; padding: 2px;">
            <img src="{flag_url}" width="44" style="border: 1px solid #ddd; border-radius: 4px; margin-bottom: 5px;"/><br/>
            <div style="font-size: 13px; font-weight: 700; color: #1f2937;">{name} ({iso})</div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 5px;">{region}</div>
            <div style="background: #f3f4f6; border-radius: 5px; padding: 5px; font-size: 11px; color: #111827; text-align: left;">
                <div>📊 <b>Avg:</b> {avg_val:.3f}%</div>
                <div>🚀 <b>Peak:</b> {peak_val:.3f}%</div>
            </div>
            <button onclick="
                if (window.parent && window.parent.Shiny) {{
                    window.parent.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                }} else if (window.Shiny) {{
                    window.Shiny.setInputValue('map_click_iso', '{iso}', {{priority: 'event'}});
                }}
            " style="
                margin-top: 7px; padding: 6px 10px; cursor: pointer;
                background-color: {btn_bg}; color: white; border: none;
                border-radius: 5px; font-weight: 600; font-size: 11.5px; width: 100%;
            ">
                {btn_label}
            </button>
        </div>
        """

        tooltip_html = (
            f"<div style='font-family:sans-serif; font-size:11.5px;'>"
            f"<b>{iso_to_flag_emoji(iso)} {name} ({iso})</b><br/>"
            f"Avg: <b>{avg_val:.3f}%</b> | Peak: {peak_val:.3f}%</div>"
        )

        if map_style == "flags":
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
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)
            else:
                size_px = int(np.clip(11 + np.sqrt(max(avg_val, 0.15)) * (5.5 if is_mobile else 7.2), 12, 42 if is_mobile else 54))
                if is_selected and size_px < 20:
                    size_px = 20
                border_css = "3px solid #dc2626" if is_selected else "2px solid #ffffff"
                shadow_css = "0 0 9px rgba(220, 38, 38, 0.85)" if is_selected else "0 2px 5px rgba(0,0,0,0.35)"
                z_boost = 1000 if is_selected else int(avg_val * 20)

                badge_html = f"""
                <div style="
                    width: {size_px}px; height: {size_px}px; border-radius: 50%;
                    background-image: url('{flag_url}'); background-size: cover;
                    background-position: center; border: {border_css}; box-shadow: {shadow_css};
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
                    popup=folium.Popup(popup_html, max_width=200)
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
                    popup=folium.Popup(popup_html, max_width=200)
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
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)

        else:
            radius_px = (
                3.5 if (avg_val < 0.10 and not is_selected)
                else float(np.clip(5 + np.sqrt(max(avg_val, 0.05)) * (4.2 if is_mobile else 5.8), 6, 30 if is_mobile else 40))
            )
            fill_col = colormap(np.sqrt(max(avg_val, 0.0)))
            folium.CircleMarker(
                location=[lat, lng],
                radius=radius_px,
                color="#dc2626" if is_selected else "#334155",
                weight=2.5 if is_selected else 1,
                fill=True,
                fill_color=fill_col,
                fill_opacity=0.90 if is_selected else (0.50 if avg_val < 0.10 else 0.80),
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

    if map_style in ["bubbles", "polygons"]:
        colormap.add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Mobile-responsive legend CSS inside iframe
    m.get_root().header.add_child(folium.Element("""
    <style>
        .leaflet-container { font-family: 'Segoe UI', Arial, sans-serif; }
        @media (max-width: 640px) {
            .legend.leaflet-control {
                max-width: 195px !important;
                transform: scale(0.82);
                transform-origin: top right;
            }
            .leaflet-popup-content { margin: 8px 10px !important; }
        }
    </style>
    """))

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


GEOPOLITICAL_CS_COLORSCALE = [
    [0.00, "#f8fafc"],
    [0.04, "#e0f2fe"],
    [0.18, "#bae6fd"],
    [0.38, "#60a5fa"],
    [0.62, "#2563eb"],
    [0.76, "#1d4ed8"],
    [0.86, "#1e3a8a"],
    [0.92, "#f97316"],
    [0.96, "#dc2626"],
    [1.00, "#991b1b"],
]


def create_contribution_choropleth(data: pd.DataFrame, is_mobile: bool = False) -> go.Figure:
    """Create smooth animated yearly world choropleth map optimized for desktop and mobile."""
    if data.empty or 'total_percentage' not in data.columns:
        return create_empty_plot("No data available for choropleth", is_mobile=is_mobile)

    df_clean = data.dropna(subset=['total_percentage', 'year', 'iso3c']).copy()
    if df_clean.empty:
        return create_empty_plot("No valid country data available for choropleth", is_mobile=is_mobile)

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

    colorbar_cfg = (
        dict(
            orientation="h",
            title=dict(text="CS Share (%)", side="top", font=dict(size=9)),
            thickness=8,
            len=0.85,
            x=0.5, xanchor="center",
            y=1.02, yanchor="bottom",
            tickfont=dict(size=8),
            tickvals=[np.log10(1.0 + v * 10.0) for v in tick_vals_real],
            ticktext=[f"{v}%" for v in tick_vals_real],
        )
        if is_mobile
        else dict(
            title=dict(text="CS Share", side="right", font=dict(size=10)),
            thickness=9,
            len=0.72,
            x=0.99,
            xpad=2,
            outlinewidth=0,
            tickfont=dict(size=9),
            tickvals=[np.log10(1.0 + v * 10.0) for v in tick_vals_real],
            ticktext=[f"{v}%" for v in tick_vals_real],
        )
    )

    fig.update_layout(
        autosize=True,
        coloraxis_colorbar=colorbar_cfg,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.6)",
            showland=True,
            landcolor="#f1f5f9",
            lataxis_range=[-55, 85],
            projection_scale=1.08,
        ),
        template='plotly_white',
        height=390 if is_mobile else 490,
        margin=dict(l=0, r=0 if is_mobile else 10, t=35 if is_mobile else 10, b=0),
        dragmode=False,
    )

    if fig.layout.sliders:
        fig.layout.sliders[0].currentvalue = dict(
            prefix="Year: ",
            font=dict(size=12 if is_mobile else 14, color="#1e293b"),
            visible=True,
            xanchor="left"
        )
        fig.layout.sliders[0].pad = dict(t=4, b=4)
        fig.layout.sliders[0].len = 0.78 if is_mobile else 0.85
        fig.layout.sliders[0].x = 0.20 if is_mobile else 0.12
        fig.layout.sliders[0].transition = dict(duration=300, easing="cubic-in-out")

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].pad = dict(r=6, t=10)
        fig.layout.updatemenus[0].x = 0.01
        fig.layout.updatemenus[0].y = 0.04
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 320
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["easing"] = "cubic-in-out"

    return fig


def create_contribution_map_plot(
    processed_data_df: pd.DataFrame,
    fill_label: str = "Avg Share",
    main_title: str = "Average Contribution Over Selected Period",
    is_mobile: bool = False,
) -> go.Figure:
    """Static choropleth map optimized for desktop and mobile."""
    if processed_data_df.empty or 'total_percentage' not in processed_data_df.columns:
        return create_empty_plot("No data for map", is_mobile=is_mobile)

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

    colorbar_cfg = (
        dict(
            orientation="h",
            title=dict(text=fill_label, side="top", font=dict(size=9)),
            thickness=8,
            len=0.85,
            x=0.5, xanchor="center",
            y=-0.05, yanchor="top",
            tickfont=dict(size=8),
            tickvals=tick_vals_scaled,
            ticktext=[f"{v}%" for v in tick_vals_real],
        )
        if is_mobile
        else dict(
            title=dict(text=fill_label, side="right", font=dict(size=10)),
            thickness=9,
            len=0.72,
            x=0.99,
            xpad=2,
            outlinewidth=0,
            tickfont=dict(size=9),
            tickvals=tick_vals_scaled,
            ticktext=[f"{v}%" for v in tick_vals_real],
        )
    )

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
        colorbar=colorbar_cfg
    ))

    fig.update_layout(
        autosize=True,
        title={'text': f"<b>{main_title}</b>", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 11 if is_mobile else 13}},
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
        height=370 if is_mobile else 490,
        margin=dict(l=0, r=0 if is_mobile else 10, t=35, b=40 if is_mobile else 5),
        dragmode=False,
    )
    return fig


def create_animated_bubble_figure(
    df: pd.DataFrame,
    chemical_category: str = "All",
    top_n: int = 20,
    is_mobile: bool = False,
) -> go.Figure:
    """Client-side animated Bubble & Flag Explorer across 1996-2022."""
    sub = df[(df['is_collab'] == False) & (df['chemical'] == chemical_category)].copy()
    if sub.empty:
        return create_empty_plot("No data available for animated bubble chart.", is_mobile=is_mobile)

    val_col = 'percentage' if 'percentage' in sub.columns else 'value'
    sub['year'] = pd.to_numeric(sub['year'], errors='coerce')
    sub[val_col] = pd.to_numeric(sub[val_col], errors='coerce')
    sub = sub.dropna(subset=['year', val_col, 'country'])
    sub['region'] = sub['region'].fillna('Other')

    effective_top_n = min(top_n, 15) if is_mobile else top_n
    top_countries = (
        sub.groupby('country')[val_col]
        .mean()
        .sort_values(ascending=False)
        .head(effective_top_n)
        .index.tolist()
    )
    anim_df = sub[sub['country'].isin(top_countries)].copy()
    anim_df['year'] = anim_df['year'].astype(int)
    anim_df = anim_df.sort_values(['year', val_col], ascending=[True, False])

    anim_df['yearly_rank'] = anim_df.groupby('year')[val_col].rank(ascending=False, method='first')
    # Compact labels on mobile (flag + ISO2) prevent overlapping text walls
    anim_df['label_with_flag'] = anim_df.apply(
        lambda r: (
            f"{iso_to_flag_emoji(str(r.get('iso2c', '')))} {str(r.get('iso2c', ''))}"
            if is_mobile
            else f"{iso_to_flag_emoji(str(r.get('iso2c', '')))} {r['country']}"
        ),
        axis=1
    )
    anim_df['bubble_size'] = np.sqrt(anim_df[val_col].clip(lower=0.2)) * 8 + 8
    max_y = max(float(anim_df[val_col].max()) * 1.18, 10.0)

    fig = px.scatter(
        anim_df,
        x="yearly_rank",
        y=val_col,
        animation_frame="year",
        animation_group="country",
        size="bubble_size",
        color="region",
        hover_name="country",
        text="label_with_flag",
        custom_data=["country", "iso2c", "year", val_col, "yearly_rank", "region"],
        size_max=34 if is_mobile else 46,
        range_x=[0.3, effective_top_n + 0.8],
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
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=8.5 if is_mobile else 10),
        hovertemplate=hover_tmpl
    )
    for frame in fig.frames:
        for trace in frame.data:
            trace.hovertemplate = hover_tmpl
            trace.textposition = "top center"

    fig.update_layout(
        title=dict(
            text=f"<b>Global Ranking & Share ({chemical_category})</b>",
            x=0.5, xanchor="center",
            font=dict(size=12 if is_mobile else 14)
        ),
        xaxis=dict(
            title="Global Rank (1 = Leader)",
            tickmode="linear", tick0=1, dtick=2 if is_mobile else 1,
            autorange=False, fixedrange=True
        ),
        yaxis=dict(
            title="Share (%)", ticksuffix="%",
            autorange=False, fixedrange=True
        ),
        template="plotly_white",
        legend=dict(title=dict(text=""), orientation="h", y=1.08, x=0.5, xanchor="center"),
    )

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 550
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=540,
        mobile_height=430,
        top_desktop=75,
        top_mobile=65,
        bottom_desktop=60,
        bottom_mobile=75,
    )


# ==============================================================================
# UPGRADED TOP 10 GLOBAL TRENDS ANALYTICS & VISUALIZATIONS
# ==============================================================================

def compute_top10_analytics(
    df: pd.DataFrame,
    is_collab: bool,
    chemical_category: str,
    year_range: Tuple[int, int],
    ranking_basis: str = "avg",
) -> dict:
    """
    Compute rich Top 10 analytics for countries or collaborations over any year_range:
    - Top 10 entity trajectories with global yearly rank
    - Leaderboard DataFrame with start/end share, pp shift, rank delta, avg, and peak
    - Executive KPI metrics (Leader, Biggest Gainer, Top 10 Concentration)
    """
    y0, y1 = int(year_range[0]), int(year_range[1])
    sub = df[
        (df['is_collab'] == is_collab) &
        (df['chemical'] == chemical_category) &
        (df['year'] >= y0) &
        (df['year'] <= y1)
    ].copy()

    if sub.empty:
        return {"empty": True}

    sub['year'] = pd.to_numeric(sub['year'], errors='coerce')
    sub['percentage'] = pd.to_numeric(sub['percentage'], errors='coerce')
    sub = sub.dropna(subset=['year', 'percentage', 'country'])
    sub['year'] = sub['year'].astype(int)

    if sub.empty:
        return {"empty": True}

    actual_start = int(sub['year'].min())
    actual_end = int(sub['year'].max())

    # Aggregate duplicates safely (if any) per entity & year
    agg = (
        sub.groupby(['country', 'year'], as_index=False)
        .agg(
            percentage=('percentage', 'mean'),
            iso2c=('iso2c', 'first'),
            cc=('cc', 'first'),
        )
    )
    # Compute global yearly rank across ALL entities in this subspace/category
    agg['global_rank'] = agg.groupby('year')['percentage'].rank(ascending=False, method='first').astype(int)

    # Choose Top 10 based on user's ranking basis ('avg' or 'latest')
    if ranking_basis == "latest":
        latest_slice = agg[agg['year'] == actual_end]
        top_names = (
            latest_slice.sort_values('percentage', ascending=False)
            .head(10)['country']
            .tolist()
        )
    else:
        top_names = (
            agg.groupby('country')['percentage']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )

    top_df = agg[agg['country'].isin(top_names)].copy()

    # Build leaderboard rows
    rows = []
    for entity in top_names:
        edf = top_df[top_df['country'] == entity].sort_values('year')
        iso_val = str(edf['iso2c'].iloc[0]) if not edf.empty and pd.notna(edf['iso2c'].iloc[0]) else ""
        color_val = str(edf['cc'].iloc[0]) if not edf.empty and pd.notna(edf['cc'].iloc[0]) else ""

        start_row = edf[edf['year'] == actual_start]
        end_row = edf[edf['year'] == actual_end]

        start_share = float(start_row['percentage'].iloc[0]) if not start_row.empty else 0.0
        end_share = float(end_row['percentage'].iloc[0]) if not end_row.empty else 0.0
        start_rank = int(start_row['global_rank'].iloc[0]) if not start_row.empty else None
        end_rank = int(end_row['global_rank'].iloc[0]) if not end_row.empty else None

        shift_pp = end_share - start_share
        avg_share = float(edf['percentage'].mean())
        peak_idx = edf['percentage'].idxmax()
        peak_share = float(edf.loc[peak_idx, 'percentage'])
        peak_year = int(edf.loc[peak_idx, 'year'])

        if start_rank is not None and end_rank is not None:
            delta_r = start_rank - end_rank  # positive = improved rank (e.g. #6 -> #1 is +5)
            rank_badge = f"▲ +{delta_r}" if delta_r > 0 else (f"▼ {delta_r}" if delta_r < 0 else "— 0")
        elif end_rank is not None:
            rank_badge = "🆕 NEW"
        else:
            rank_badge = "—"

        rows.append({
            "entity_raw": entity,
            "iso2c": iso_val,
            "color": color_val,
            "Entity": format_entity_with_flags(entity, iso_val),
            "end_rank": end_rank if end_rank is not None else 999,
            "start_rank": start_rank,
            "Rank Δ": rank_badge,
            "start_share": start_share,
            "end_share": end_share,
            "shift_pp": shift_pp,
            "avg_share": avg_share,
            "peak_share": peak_share,
            "peak_year": peak_year,
        })

    lb_raw = pd.DataFrame(rows)
    sort_col = "end_share" if ranking_basis == "latest" else "avg_share"
    lb_raw = lb_raw.sort_values(sort_col, ascending=False).reset_index(drop=True)
    lb_raw["Pos"] = lb_raw.index + 1

    # Display-formatted leaderboard DataFrame
    if actual_start == actual_end:
        lb_display = pd.DataFrame({
            "#": lb_raw["Pos"],
            "Entity": lb_raw["Entity"],
            f"Share {actual_end} (%)": lb_raw["end_share"].round(2),
            "Global Rank": lb_raw["end_rank"].apply(lambda r: f"#{r}" if r < 999 else "—"),
        })
    else:
        lb_display = pd.DataFrame({
            "#": lb_raw["Pos"],
            f"Rank ({actual_end})": lb_raw["end_rank"].apply(lambda r: f"#{r}" if r < 999 else "—"),
            f"Δ ({actual_start}→{actual_end})": lb_raw["Rank Δ"],
            "Entity": lb_raw["Entity"],
            f"{actual_start} (%)": lb_raw["start_share"].round(2),
            f"{actual_end} (%)": lb_raw["end_share"].round(2),
            "Shift (pp)": lb_raw["shift_pp"].apply(lambda x: f"{x:+.2f}"),
            "Avg (%)": lb_raw["avg_share"].round(2),
            "Peak (%)": lb_raw.apply(lambda r: f"{r['peak_share']:.2f}% ({r['peak_year']})", axis=1),
        })

    # KPI Calculations
    leader_row = lb_raw.sort_values("end_share", ascending=False).iloc[0]
    gainer_row = lb_raw.sort_values("shift_pp", ascending=False).iloc[0]
    conc_end = float(lb_raw["end_share"].sum())
    conc_start = float(lb_raw["start_share"].sum())

    kpis = {
        "actual_start": actual_start,
        "actual_end": actual_end,
        "leader_label": leader_row["Entity"],
        "leader_val": f"{leader_row['end_share']:.2f}%",
        "leader_sub": f"Avg: {leader_row['avg_share']:.2f}% | Peak: {leader_row['peak_share']:.1f}% ({leader_row['peak_year']})",
        "gainer_label": gainer_row["Entity"],
        "gainer_val": f"{gainer_row['shift_pp']:+.2f} pp",
        "gainer_sub": f"{gainer_row['start_share']:.2f}% ({actual_start}) → {gainer_row['end_share']:.2f}% ({actual_end})",
        "conc_end": f"{conc_end:.1f}%",
        "conc_delta": f"{conc_end - conc_start:+.1f} pp vs {actual_start} ({conc_start:.1f}%)",
    }

    return {
        "empty": False,
        "top_df": top_df,
        "lb_raw": lb_raw,
        "lb_display": lb_display,
        "kpis": kpis,
    }


def create_top_trends_plot(
    data: pd.DataFrame,
    title: str,
    is_collab: bool = False,
    show_flags: bool = True,
    is_mobile: bool = False,
) -> go.Figure:
    """Create enhanced Top 10 share trend plot with flag icons and responsive layout."""
    if data.empty:
        return create_empty_plot("No data available.", is_mobile=is_mobile)

    fig = go.Figure()
    palette = px.colors.qualitative.Bold + px.colors.qualitative.D3
    avg_percentages = data.groupby('country')['percentage'].mean().sort_values(ascending=False)
    used_colors = set()

    for idx, entity in enumerate(avg_percentages.index):
        entity_data = data[data['country'] == entity].sort_values('year')
        avg_value = avg_percentages[entity]
        iso_val = str(entity_data['iso2c'].iloc[0]) if 'iso2c' in entity_data.columns and pd.notna(entity_data['iso2c'].iloc[0]) else ""
        label = format_entity_with_flags(entity, iso_val)

        color = None
        if 'cc' in entity_data.columns and not entity_data['cc'].dropna().empty:
            cand = str(entity_data['cc'].iloc[0])
            if cand and cand not in ("#3b82f6", "#808080") and cand not in used_colors:
                color = cand
        if not color:
            color = palette[idx % len(palette)]
        used_colors.add(color)

        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data['percentage'],
            mode='lines+markers',
            name=f"{label} ({avg_value:.1f}%)",
            line=dict(width=2.3 if idx < 3 else 1.8, color=color),
            marker=dict(color=color, size=5 if is_mobile else 6, line=dict(width=0.8, color="white")),
            hovertemplate=f"<b>{label}</b><br>Year: <b>%{{x}}</b><br>Share: <b>%{{y:.2f}}%</b><extra></extra>"
        ))

    if show_flags and not is_collab:
        flag_df = data.rename(columns={'percentage': 'total_percentage'})
        fig = add_flags_to_trend_plot(fig, flag_df, value_col='total_percentage', is_mobile=is_mobile)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=12 if is_mobile else 14)),
        yaxis=dict(
            ticksuffix='%', fixedrange=True, rangemode='tozero',
            title=dict(text="Share (%)" if is_mobile else "New Substances (%)", font=dict(size=10 if is_mobile else 12))
        ),
        xaxis=dict(
            fixedrange=True,
            title=dict(text="Year", font=dict(size=10 if is_mobile else 12))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode='closest' if is_mobile else 'x unified',
    )
    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=490,
        mobile_height=385,
        top_desktop=42,
        top_mobile=34,
        bottom_desktop=105,
        bottom_mobile=125,
    )


def create_top10_bump_chart(
    top_df: pd.DataFrame,
    lb_raw: pd.DataFrame,
    title: str,
    is_mobile: bool = False,
) -> go.Figure:
    """Create a Bump Chart tracking global rank trajectories (#1 at top) for Top 10 entities."""
    if top_df.empty or lb_raw.empty:
        return create_empty_plot("Not enough yearly data for a rank trajectory chart.", is_mobile=is_mobile)

    fig = go.Figure()
    palette = px.colors.qualitative.Bold + px.colors.qualitative.D3
    max_display_rank = 15
    used_colors = set()

    min_yr = int(top_df['year'].min())
    max_yr = int(top_df['year'].max())

    for idx, row in lb_raw.iterrows():
        entity = row["entity_raw"]
        label = row["Entity"]
        edf = top_df[top_df['country'] == entity].sort_values('year')
        if edf.empty:
            continue

        color = row["color"] if row["color"] and row["color"] not in ("#3b82f6", "#808080") and row["color"] not in used_colors else palette[idx % len(palette)]
        used_colors.add(color)

        clipped_rank = edf['global_rank'].clip(upper=max_display_rank)
        end_label = [""] * len(edf)
        if not is_mobile:
            short_lbl = f" #{int(edf['global_rank'].iloc[-1])} {row['iso2c'] or entity[:10]}"
            end_label[-1] = short_lbl

        fig.add_trace(go.Scatter(
            x=edf['year'],
            y=clipped_rank,
            mode='lines+markers+text' if not is_mobile else 'lines+markers',
            name=label,
            text=end_label,
            textposition="middle right",
            textfont=dict(size=10, color=color),
            line=dict(color=color, width=2.8 if idx < 3 else 2.0, shape='spline', smoothing=0.5),
            marker=dict(size=6 if not is_mobile else 5, color=color, line=dict(width=1, color="white")),
            customdata=np.stack([edf['global_rank'], edf['percentage']], axis=-1),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Year: <b>%{x}</b><br>"
                "Global Rank: <b>#%{customdata[0]:.0f}</b><br>"
                "Share: <b>%{customdata[1]:.2f}%</b><extra></extra>"
            )
        ))

    tick_vals = list(range(1, max_display_rank + 1))
    tick_text = [f"#{r}" if r < max_display_rank else f"#{max_display_rank}+" for r in tick_vals]

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=12 if is_mobile else 14)),
        xaxis=dict(
            range=[min_yr - 0.5, max_yr + (0.8 if is_mobile else 3.2)],
            fixedrange=True,
            title=dict(text="Year", font=dict(size=10 if is_mobile else 12))
        ),
        yaxis=dict(
            autorange="reversed",
            tickvals=tick_vals,
            ticktext=tick_text,
            fixedrange=True,
            gridcolor="rgba(148,163,184,0.22)",
            title=dict(text="Global Rank (#1 = Top)", font=dict(size=10 if is_mobile else 12))
        ),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode="closest",
    )
    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=490,
        mobile_height=395,
        top_desktop=42,
        top_mobile=34,
        bottom_desktop=105,
        bottom_mobile=125,
    )


def create_top10_shift_chart(
    lb_raw: pd.DataFrame,
    start_yr: int,
    end_yr: int,
    title: str,
    is_mobile: bool = False,
) -> go.Figure:
    """Create a horizontal Dumbbell Chart showing Start Year vs End Year share shift."""
    if lb_raw.empty or start_yr == end_yr:
        return create_empty_plot(
            "Select a multi-year range (at least 2 years) to compare period shifts.",
            is_mobile=is_mobile
        )

    df_plot = lb_raw.sort_values("end_share", ascending=True).reset_index(drop=True)
    fig = go.Figure()

    # Connecting stems colored by direction (emerald for growth, crimson for decline)
    for _, r in df_plot.iterrows():
        stem_color = "#059669" if r["shift_pp"] >= 0 else "#dc2626"
        fig.add_trace(go.Scatter(
            x=[r["start_share"], r["end_share"]],
            y=[r["Entity"], r["Entity"]],
            mode="lines",
            line=dict(color=stem_color, width=3),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Start year markers
    fig.add_trace(go.Scatter(
        x=df_plot["start_share"],
        y=df_plot["Entity"],
        mode="markers",
        name=f"Start ({start_yr})",
        marker=dict(size=9 if not is_mobile else 7, color="#94a3b8", line=dict(width=1.5, color="#475569")),
        hovertemplate=f"<b>%{{y}}</b><br>{start_yr}: <b>%{{x:.2f}}%</b><extra></extra>"
    ))

    # End year markers colored by gain/loss
    end_colors = ["#059669" if v >= 0 else "#dc2626" for v in df_plot["shift_pp"]]
    fig.add_trace(go.Scatter(
        x=df_plot["end_share"],
        y=df_plot["Entity"],
        mode="markers",
        name=f"End ({end_yr}) — Green: Gain, Red: Decline",
        marker=dict(size=11 if not is_mobile else 9, color=end_colors, line=dict(width=1.5, color="#ffffff")),
        customdata=df_plot[["start_share", "shift_pp"]].values,
        hovertemplate=(
            f"<b>%{{y}}</b><br>{end_yr}: <b>%{{x:.2f}}%</b><br>"
            f"{start_yr}: %{{customdata[0]:.2f}}%<br>"
            "Shift: <b>%{customdata[1]:+.2f} pp</b><extra></extra>"
        )
    ))

    # Delta annotations beside the right-most dot
    max_x = max(float(df_plot[["start_share", "end_share"]].max().max()), 1.0)
    for _, r in df_plot.iterrows():
        x_pos = max(r["start_share"], r["end_share"]) + max_x * 0.025
        txt_col = "#065f46" if r["shift_pp"] >= 0 else "#991b1b"
        fig.add_annotation(
            x=x_pos, y=r["Entity"],
            text=f"<b>{r['shift_pp']:+.1f} pp</b>",
            showarrow=False, xanchor="left",
            font=dict(size=9 if is_mobile else 10.5, color=txt_col)
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=12 if is_mobile else 14)),
        xaxis=dict(
            ticksuffix="%",
            range=[0, max_x * (1.26 if is_mobile else 1.18)],
            fixedrange=True,
            title=dict(text="Share of New Substances (%)", font=dict(size=10 if is_mobile else 12))
        ),
        yaxis=dict(
            fixedrange=True,
            tickfont=dict(size=9.5 if is_mobile else 11),
            automargin=True,
        ),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"),
        hovermode="closest",
    )
    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=470,
        mobile_height=400,
        top_desktop=42,
        top_mobile=34,
        bottom_desktop=75,
        bottom_mobile=85,
    )


# ==============================================================================
# SUMMARY TABLE & ARTICLE FIGURES (MOBILE-OPTIMIZED)
# ==============================================================================

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
            data.groupby(['plot_group', 'iso2c', 'chemical', 'collab_type'], as_index=False)
            .agg(
                avg_percentage=(value_column, 'mean'),
                max_percentage=(value_column, 'max'),
                years_present=('year', 'nunique')
            )
            .round(3)
            .sort_values('avg_percentage', ascending=False)
        )
        summary['Collaboration'] = summary.apply(
            lambda r: format_entity_with_flags(r['plot_group'], str(r['iso2c'])), axis=1
        )
        summary = summary[['Collaboration', 'collab_type', 'chemical', 'avg_percentage', 'max_percentage', 'years_present']]
        summary.columns = ['Collaboration', 'Type', 'Subspace', 'Avg Share (%)', 'Peak Share (%)', 'Active Years']

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
        summary.columns = ['Country', 'ISO', 'Subspace', 'Avg Share (%)', 'Peak Share (%)', 'Active Years']
    else:
        return pd.DataFrame({'Message': [f"Summary not available for mode: {mode}"]})

    return summary


def create_article_plot(
    data: pd.DataFrame,
    title: str,
    show_exp_fit: bool = True,
    is_mobile: bool = False,
) -> go.Figure:
    """Create main article plot (Figure 1a) with responsive layout."""
    if data.empty:
        return create_empty_plot("No country participation data available", is_mobile=is_mobile)

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
                size=c_df['value'].abs().clip(lower=4, upper=9 if is_mobile else 11),
                opacity=0.82,
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
            name='China Exp. Fit (R²=0.98)',
            line=dict(color='#c5051b', width=1.5, dash='dot'),
            hovertemplate="<b>China Exp. Fit</b><br>Year: %{x}<br>Fit: %{y:.2f}%<extra></extra>"
        ))

    fig.add_vline(x=2013, line_dash="dash", line_color="rgba(197, 5, 27, 0.45)")
    fig.add_annotation(
        x=2013, y=36,
        text="<b>2013:</b> CN #1" if is_mobile else "<b>2013:</b> China becomes #1",
        showarrow=True, arrowhead=2, ax=-42 if is_mobile else -55, ay=-22,
        font=dict(size=9.5 if is_mobile else 11, color="#c5051b"),
        bgcolor="rgba(255,255,255,0.88)", bordercolor="#c5051b", borderwidth=1
    )

    updatemenus = [] if is_mobile else [
        dict(
            type="buttons",
            direction="right",
            active=2,
            x=1, y=1.14,
            buttons=[
                dict(label="1996–2012", method="relayout", args=[{"xaxis.range": [1996, 2013]}]),
                dict(label="2012–2022", method="relayout", args=[{"xaxis.range": [max_year - 10, max_year + 1]}]),
                dict(label="All Years", method="relayout", args=[{"xaxis.range": [min_year - 1, max_year + 1]}])
            ]
        )
    ]

    fig.update_layout(
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="Share (%)" if is_mobile else "Share of New Substances (%)", font=dict(size=10 if is_mobile else 12))
        ),
        xaxis=dict(
            autorange=False,
            range=[min_year - 1, max_year + 1],
            fixedrange=True,
            title=dict(text="Year", font=dict(size=10 if is_mobile else 12))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", traceorder="reversed"),
        hovermode='closest' if is_mobile else 'x unified',
        updatemenus=updatemenus,
    )
    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=460,
        mobile_height=360,
        top_desktop=48,
        top_mobile=22,
        bottom_desktop=95,
        bottom_mobile=105,
    )


def create_gdp_plot(data: pd.DataFrame, is_mobile: bool = False) -> go.Figure:
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
            line=dict(width=2.2, color=color),
            marker=dict(size=c_df['value'].abs().clip(upper=10 if is_mobile else 13) + 2, color=color)
        ))

    fig.add_vline(x=2007.5, line_dash="dash", line_color="grey")
    fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")

    if fig.data:
        max_val = max([max(trace['y']) for trace in fig.data if len(trace['y']) > 0])
        fig.add_annotation(
            x=2007.5, y=max_val * 0.9,
            text="2008 Crisis" if is_mobile else "Global Financial Crisis",
            showarrow=True, arrowhead=1, font=dict(size=9.5 if is_mobile else 11)
        )
        fig.add_annotation(
            x=2019.5, y=max_val * 0.78,
            text="COVID-19", showarrow=True, arrowhead=1, font=dict(size=9.5 if is_mobile else 11)
        )

    fig.update_layout(
        yaxis=dict(ticksuffix='%', fixedrange=True, title=dict(text="GDP Growth (%)", font=dict(size=10 if is_mobile else 12))),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=10 if is_mobile else 12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", traceorder="reversed"),
        hovermode='closest' if is_mobile else 'x unified',
    )
    return apply_responsive_layout(fig, is_mobile=is_mobile, desktop_height=430, mobile_height=340, top_desktop=25, top_mobile=18)


def create_researchers_plot(data: pd.DataFrame, is_mobile: bool = False) -> go.Figure:
    """Create researchers headcount plot (Figure 1e)."""
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
            line=dict(width=2.2, color=color),
            marker=dict(size=scaled_values.abs().clip(lower=4, upper=10) + 2, color=color)
        ))

    fig.add_vline(x=2005, line_dash="dot", line_color="rgba(100,100,100,0.6)")
    fig.add_annotation(
        x=2005, y=2.1,
        text="2005: CN > US" if is_mobile else "2005: China surpasses US in R&D",
        showarrow=True, arrowhead=2, ax=40 if is_mobile else 65, ay=-18,
        font=dict(size=9.5 if is_mobile else 10.5)
    )

    fig.update_layout(
        yaxis=dict(ticksuffix='M', fixedrange=True, title=dict(text="Researchers (M)" if is_mobile else "Number of Researchers (Millions)", font=dict(size=10 if is_mobile else 12))),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=10 if is_mobile else 12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", traceorder="reversed"),
        hovermode='closest' if is_mobile else 'x unified',
    )
    return apply_responsive_layout(fig, is_mobile=is_mobile, desktop_height=430, mobile_height=340, top_desktop=25, top_mobile=18)


def create_cs_expansion_plot(data: pd.DataFrame, is_mobile: bool = False) -> go.Figure:
    """Create Chemical Space expansion plot (Figure 2a) on a logarithmic scale."""
    fig = go.Figure()
    for country in data['country'].unique():
        c_df = data[data['country'] == country].sort_values('year')
        fig.add_trace(go.Scatter(
            x=c_df['year'],
            y=c_df['value'],
            mode='lines+markers',
            name=country,
            line=dict(width=2.2),
            marker=dict(size=4.5 if is_mobile else 5.5)
        ))

    fig.add_vline(x=2008, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")
    fig.add_vline(x=2020, line_dash="dot", line_color="rgba(128, 128, 128, 0.5)")

    fig.update_layout(
        yaxis=dict(
            fixedrange=True,
            type="log",
            title=dict(text="New Substances (Log)", font=dict(size=10 if is_mobile else 12))
        ),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=10 if is_mobile else 12))),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", traceorder="reversed"),
        hovermode='closest' if is_mobile else 'x unified',
    )
    return apply_responsive_layout(fig, is_mobile=is_mobile, desktop_height=430, mobile_height=340, top_desktop=25, top_mobile=18)


def create_china_us_dual_axis_plot(data: pd.DataFrame, is_mobile: bool = False) -> go.Figure:
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
            line=dict(width=2.3, color=colors.get(country_name)),
            marker=dict(size=5 if is_mobile else 6, color=colors.get(country_name))
        )
        if country_name in secondary_yaxis_categories:
            trace_params['yaxis'] = 'y2'
        fig.add_trace(go.Scatter(**trace_params))

    fig.add_vrect(
        x0=2009, x1=2017,
        fillcolor="rgba(91, 155, 213, 0.08)", opacity=1.0,
        layer="below", line_width=0,
        annotation_text="2009–17 Agreements" if is_mobile else "2009–2017: US–China Bilateral Agreements",
        annotation_position="top left",
        annotation_font=dict(size=9 if is_mobile else 10, color="#475569")
    )
    fig.add_vline(x=2018, line_dash="dot", line_color="rgba(180, 50, 50, 0.5)")
    fig.add_annotation(
        x=2018, y=95,
        text="2018 US China Init." if is_mobile else "2018: US China Initiative",
        showarrow=False, textangle=-90, font=dict(size=9 if is_mobile else 10, color="#a02020"), xshift=10
    )

    fig.update_layout(
        yaxis=dict(
            ticksuffix='%', fixedrange=True,
            title=dict(text="Domestic Share (%)" if is_mobile else "Domestic Solo Share (%)", font=dict(size=10 if is_mobile else 12))
        ),
        xaxis=dict(fixedrange=True, title=dict(text="Year", font=dict(size=10 if is_mobile else 12))),
        yaxis2=dict(
            overlaying='y', side='right', ticksuffix='%', fixedrange=True,
            title=dict(text="CN–US Share (%)" if is_mobile else "China–US Bilateral Share (%)", font=dict(size=9.5 if is_mobile else 11))
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        hovermode='closest' if is_mobile else 'x unified',
    )
    return apply_responsive_layout(
        fig,
        is_mobile=is_mobile,
        desktop_height=460,
        mobile_height=360,
        top_desktop=28,
        top_mobile=22,
        bottom_desktop=95,
        bottom_mobile=105,
    )