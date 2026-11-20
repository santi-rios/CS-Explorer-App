import pandas as pd
import plotly.offline as plotly_offline
from shiny import App, ui, render, reactive
from pathlib import Path
from functools import lru_cache
import shinyswatch

from utils.functions import (
    get_display_data,
    create_enhanced_folium_map,
    create_trends_plot,
    create_contribution_choropleth,
    create_contribution_map_plot,
    create_summary_dataframe,
    create_article_plot,
    create_top_trends_plot,
    compute_top10_analytics,
    create_top10_bump_chart,
    create_top10_shift_chart,
    create_empty_plot,
    create_gdp_plot,
    create_researchers_plot,
    create_cs_expansion_plot,
    create_china_us_dual_axis_plot,
    get_raw_dataframe,
    load_article_data,
    create_animated_bubble_figure,
    iso_to_flag_emoji,
)

www_dir = Path(__file__).resolve().parent / "www"


@lru_cache(maxsize=1)
def load_country_list() -> pd.DataFrame:
    """Cached country metadata table including iso3c for compact selector badges."""
    df = get_raw_dataframe()
    cols = [c for c in ['country', 'iso2c', 'iso3c', 'lat', 'lng', 'cc', 'region'] if c in df.columns]
    res = (
        df[df['is_collab'] == False]
        .drop_duplicates(subset=cols)
        .dropna(subset=['country', 'iso2c'])
        .query("country != '' and iso2c != ''")
        .fillna({'region': 'Other'})
        .sort_values('country')
        .reset_index(drop=True)
    )
    if 'iso3c' not in res.columns:
        res['iso3c'] = res['iso2c']
    else:
        res['iso3c'] = res['iso3c'].fillna(res['iso2c'])
    return res


@lru_cache(maxsize=1)
def get_default_top10_isos() -> list[str]:
    """Compute the overall Top 10 countries (1996–2022, All substances) for initial load."""
    try:
        df = get_raw_dataframe()
        sub = df[(df['is_collab'] == False) & (df['chemical'] == "All")]
        top10 = (
            sub.groupby('iso2c')['percentage']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .index.dropna()
            .tolist()
        )
        return top10 if top10 else ["CN", "US", "JP", "DE", "IN", "GB", "FR", "RU", "ES", "IT"]
    except Exception:
        return ["CN", "US", "JP", "DE", "IN", "GB", "FR", "RU", "ES", "IT"]


@lru_cache(maxsize=256)
def cached_get_display_data(
    selected_isos_tuple: tuple,
    year_range: tuple,
    chemical_category: str,
    display_mode: str,
    region_filter: str = "All",
    collab_match_mode: str = "any_pair",
    collab_tier_filter: str = "All",
) -> pd.DataFrame:
    """Cached slice of display data."""
    selected_isos = list(selected_isos_tuple) if selected_isos_tuple else []
    df = get_raw_dataframe()
    country_list = load_country_list()
    return get_display_data(
        df=df,
        selected_isos=selected_isos,
        year_range=year_range,
        chemical_category=chemical_category,
        display_mode=display_mode,
        region_filter=region_filter,
        country_list=country_list,
        collab_match_mode=collab_match_mode,
        collab_tier_filter=collab_tier_filter,
    )


def figure_to_html(fig):
    """Render Plotly figure as touch-safe, responsive embedded HTML."""
    html = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )
    return ui.div(ui.HTML(html), class_="plotly-responsive-wrap")


def create_app():
    default_top10 = get_default_top10_isos()
    try:
        df_sample = get_raw_dataframe()[['chemical', 'year', 'region']]
        unique_chemicals = df_sample['chemical'].dropna().unique().tolist()
        chemical_categories = list(dict.fromkeys(
            sorted([chem for chem in unique_chemicals if chem and str(chem).strip()])
        ))

        unique_regions = df_sample['region'].fillna('Other').unique().tolist()
        regions = ['All'] + sorted(list(set(
            region for region in unique_regions if region and str(region).strip() and region != 'All'
        )))

        min_year = int(df_sample['year'].min())
        max_year = int(df_sample['year'].max())

        countries_df = load_country_list()
        country_choices = {
            row['iso2c']: f"{iso_to_flag_emoji(row['iso2c'])} {str(row['iso3c']).upper()} — {row['country']}"
            for _, row in countries_df.iterrows()
        }

        initial_data = {
            'chemical_categories': chemical_categories,
            'regions': regions,
            'min_year': min_year,
            'max_year': max_year,
            'country_choices': country_choices,
        }
    except Exception as e:
        print(f"Error loading initial data for UI: {e}")
        initial_data = {
            'chemical_categories': ["All"],
            'regions': ["All"],
            'min_year': 1996,
            'max_year': 2022,
            'country_choices': {},
        }

    chemical_radio_choices = {
        "All": "All Substances",
        "Organic": "Organic",
        "Organometallic": "Organometallic",
        "Rare-Earths": "Rare-Earths (REE)",
    }

    app_ui = ui.page_navbar(
        # ======================================================================
        # MAIN DASHBOARD
        # ======================================================================
        ui.nav_panel(
            "Dashboard",
            # Compact Responsive Title Banner
            ui.div(
                ui.h3("China's Rise in the Chemical Space & the Decline of US Influence"),
                ui.p(
                    "Interactive explorer of 15.7M new chemical substances across 189 countries (1996–2022) — ",
                    ui.tags.a(
                        "Bermúdez-Montaña et al. (2025), QSS",
                        href="https://doi.org/10.1162/QSS.a.471",
                        target="_blank",
                        class_="article-doi-link",
                    ),
                ),
                class_="app-hero-banner",
            ),

            ui.navset_card_tab(
                # --------------------------------------------------------------
                # TAB 1: PLOT-FIRST INTERACTIVE EXPLORER (WITH COLLAPSIBLE SIDEBAR)
                # --------------------------------------------------------------
                ui.nav_panel(
                    "🗺️ Interactive Explorer",
                    ui.layout_sidebar(
                        ui.sidebar(
                            # 1. Year Range right at eye level beside the main chart
                            ui.div("📅 Year Range", class_="sidebar-section-label"),
                            ui.input_slider(
                                "years",
                                None,
                                min=initial_data['min_year'],
                                max=initial_data['max_year'],
                                value=[initial_data['min_year'], initial_data['max_year']],
                                step=1,
                                sep="",
                                width="100%",
                                animate=ui.AnimationOptions(
                                    interval=750,
                                    loop=True,
                                    play_button=ui.tags.span("▶ Play", class_="btn btn-xs btn-success fw-bold"),
                                    pause_button=ui.tags.span("⏸ Pause", class_="btn btn-xs btn-warning fw-bold"),
                                ),
                            ),
                            ui.div(
                                ui.input_action_button("preset_era_all", "All", class_="btn-sm btn-outline-secondary"),
                                ui.input_action_button("preset_decade_90s", "96–05", class_="btn-sm btn-outline-secondary"),
                                ui.input_action_button("preset_decade_crossover", "06–14", class_="btn-sm btn-outline-secondary"),
                                ui.input_action_button("preset_decade_modern", "15–22", class_="btn-sm btn-outline-secondary"),
                                class_="era-btn-row",
                            ),
                            ui.div(class_="sidebar-divider"),

                            # 2. Country Search + Compact Quick Presets
                            ui.div("🔍 Countries", class_="sidebar-section-label"),
                            ui.div(
                                ui.input_selectize(
                                    "country_picker",
                                    None,
                                    choices=initial_data['country_choices'],
                                    selected=default_top10,
                                    multiple=True,
                                    remove_button=True,
                                    width="100%",
                                    options={"placeholder": "Search country or ISO..."},
                                ),
                                class_="compact-iso-picker",
                            ),
                            ui.div(
                                ui.input_action_button("preset_top10_current", "🏆 Top 10", class_="btn-sm btn-primary"),
                                ui.input_action_button("preset_biggest_risers", "🔥 Top 8 Risers", class_="btn-sm btn-outline-success"),
                                ui.input_action_button("clear_selection", "🗑️ Clear", class_="btn-sm btn-outline-danger"),
                                class_="era-btn-row mt-1",
                            ),
                            ui.div(
                                ui.output_text("selection_info"),
                                class_="text-muted small mt-1",
                            ),
                            ui.div(class_="sidebar-divider"),

                            # 3. Chemical Subspace
                            ui.div("🧪 Chemical Subspace", class_="sidebar-section-label"),
                            ui.input_radio_buttons(
                                "chemical_category",
                                None,
                                choices=chemical_radio_choices,
                                selected="All",
                                inline=False,
                            ),
                            ui.div(class_="sidebar-divider"),

                            # 4. Region Filter
                            ui.div("🌍 Region Filter", class_="sidebar-section-label"),
                            ui.input_select(
                                "region_filter",
                                None,
                                choices=initial_data['regions'],
                                selected="All",
                                width="100%",
                            ),

                            id="explorer_sidebar",
                            title="🎛️ Filters & Controls",
                            open="desktop",  # Open beside plot on desktop; closed drawer on mobile
                            width=305,
                            padding="12px",
                        ),

                        # --- MAIN CONTENT CANVAS (PLOT-FIRST!) ---
                        # 1. Primary Trend Chart Card (immediately visible above the fold)
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.div(
                                        ui.tags.span("📈 Country Contribution Trends", class_="fw-bold"),
                                        ui.tags.small(
                                            "Drag to zoom on desktop; tap points on mobile.",
                                            class_="text-muted hide-on-mobile",
                                        ),
                                        class_="card-title-group",
                                    ),
                                    ui.div(
                                        ui.popover(
                                            ui.tags.button(
                                                "⚙️ Plot Options",
                                                type="button",
                                                class_="btn btn-sm btn-outline-secondary",
                                            ),
                                            ui.input_checkbox("show_flags_on_trends", "🏳️ Flags on Lines", value=True),
                                            ui.input_checkbox("trends_log_scale", "📐 Logarithmic Y-Axis", value=False),
                                            ui.input_checkbox("trends_show_events", "📌 2008 & 2020 Markers", value=True),
                                            title="Chart Display Options",
                                            placement="bottom",
                                        ),
                                        class_="card-controls-group",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_ui("main_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card mb-3",
                        ),

                        # 2. Responsive Live KPI Strip (Individual data only — zero collab lag!)
                        ui.output_ui("dynamic_explorer_kpis"),

                        # 3. Unified Full-Width Map & Choropleth Card
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.div(
                                        ui.tags.span("🌐 Geopolitical Map & Choropleth", class_="fw-bold"),
                                        ui.tags.small(
                                            "Click any country in Selector mode to toggle it.",
                                            class_="text-muted hide-on-mobile",
                                        ),
                                        class_="card-title-group",
                                    ),
                                    ui.div(
                                        ui.input_radio_buttons(
                                            "map_view_mode",
                                            None,
                                            choices={
                                                "interactive": "📍 Selector Map",
                                                "animated": "🎬 Yearly Map",
                                                "static": "🗺️ Period Avg",
                                            },
                                            selected="interactive",
                                            inline=True,
                                        ),
                                        ui.popover(
                                            ui.tags.button(
                                                "🎨 Style",
                                                type="button",
                                                class_="btn btn-sm btn-outline-secondary",
                                            ),
                                            ui.input_radio_buttons(
                                                "map_visual_style",
                                                "Selector Map Layer:",
                                                choices={
                                                    "polygons": "🗺️ Shaded Country Boundaries",
                                                    "flags": "🏳️ Proportional Flag Badges",
                                                    "bubbles": "🟣 Proportional Bubbles",
                                                },
                                                selected="polygons",
                                            ),
                                            title="Interactive Map Layer",
                                            placement="bottom",
                                        ),
                                        class_="card-controls-group",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_ui("unified_map_output"),
                            full_screen=True,
                            class_="dashboard-plot-card mb-3",
                        ),

                        # 4. Collapsible Individual Summary Table + CSV Download
                        ui.accordion(
                            ui.accordion_panel(
                                "📋 Selected Countries Summary Table & CSV Export",
                                ui.div(
                                    ui.download_button(
                                        "download_individual_csv",
                                        "📥 Download CSV",
                                        class_="btn-sm btn-outline-secondary mb-2",
                                    ),
                                    class_="d-flex justify-content-end",
                                ),
                                ui.output_data_frame("individual_summary_table"),
                            ),
                            open=False,
                            class_="mb-2",
                        ),
                        border=False,
                        fillable=False,
                    ),
                ),

                # --------------------------------------------------------------
                # TAB 2: DEDICATED COLLABORATIONS EXPLORER (FIXED & FAST)
                # --------------------------------------------------------------
                ui.nav_panel(
                    "🤝 Collaborations Explorer",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div("🌍 Countries in Network", class_="sidebar-section-label"),
                            ui.div(
                                ui.input_selectize(
                                    "collab_country_picker",
                                    None,
                                    choices=initial_data['country_choices'],
                                    selected=["CN", "US", "DE", "GB", "JP", "IN"],
                                    multiple=True,
                                    remove_button=True,
                                    width="100%",
                                    options={"placeholder": "Pick 1 or more countries..."},
                                ),
                                class_="compact-iso-picker",
                            ),
                            ui.div(
                                ui.input_action_button(
                                    "sync_from_explorer",
                                    "🔄 Sync Explorer",
                                    class_="btn-sm btn-outline-primary",
                                ),
                                ui.input_action_button(
                                    "collab_clear",
                                    "🗑️ Clear",
                                    class_="btn-sm btn-outline-danger",
                                ),
                                class_="era-btn-row mt-1",
                            ),
                            ui.input_select(
                                "collab_preset",
                                "⚡ Load Example Network:",
                                choices={
                                    "": "— Choose a preset —",
                                    "cn_us": "China + United States",
                                    "top6": "Top 6 Powers (CN, US, DE, GB, JP, IN)",
                                    "transatlantic": "Transatlantic (US, DE, GB, FR)",
                                    "asia_hub": "East & South Asia (CN, JP, IN, KR)",
                                    "latam": "Latin America + US/EU (MX, BR, US, ES)",
                                },
                                selected="",
                                width="100%",
                            ),
                            ui.div(class_="sidebar-divider"),

                            ui.div("🔗 Matching Logic", class_="sidebar-section-label"),
                            ui.input_radio_buttons(
                                "collab_match_mode",
                                None,
                                choices={
                                    "any_pair": "Shared networks (≥2 selected)",
                                    "only_selected": "Strictly among selected only",
                                    "all_selected": "Must include ALL selected",
                                },
                                selected="any_pair",
                            ),
                            ui.div(class_="sidebar-divider"),

                            ui.div("🧩 Network Size Tier", class_="sidebar-section-label"),
                            ui.input_select(
                                "collab_tier_filter",
                                None,
                                choices=["All", "Bilateral", "Trilateral", "4-country", "5-country+"],
                                selected="All",
                                width="100%",
                            ),

                            ui.div("🧪 Chemical Subspace", class_="sidebar-section-label"),
                            ui.input_select(
                                "collab_chem",
                                None,
                                choices=initial_data['chemical_categories'],
                                selected="All",
                                width="100%",
                            ),

                            ui.div("📅 Year Range", class_="sidebar-section-label"),
                            ui.input_slider(
                                "collab_years",
                                None,
                                min=initial_data['min_year'],
                                max=initial_data['max_year'],
                                value=[initial_data['min_year'], initial_data['max_year']],
                                step=1,
                                sep="",
                                width="100%",
                            ),

                            ui.input_slider(
                                "collab_max_traces",
                                "Max Lines Displayed:",
                                min=5,
                                max=25,
                                value=12,
                                step=1,
                                width="100%",
                            ),

                            id="collab_sidebar",
                            title="🤝 Network Filters",
                            open="desktop",
                            width=305,
                            padding="12px",
                        ),

                        # Main Collaboration Content
                        ui.output_ui("collab_kpis"),

                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.div(
                                        ui.tags.span("🤝 International Collaboration Trends", class_="fw-bold"),
                                        ui.tags.small(
                                            "Ranked by average share across selected period.",
                                            class_="text-muted hide-on-mobile",
                                        ),
                                        class_="card-title-group",
                                    ),
                                    ui.div(
                                        ui.popover(
                                            ui.tags.button(
                                                "⚙️ Plot Options",
                                                type="button",
                                                class_="btn btn-sm btn-outline-secondary",
                                            ),
                                            ui.input_checkbox("collab_log_scale", "📐 Logarithmic Y-Axis", value=False),
                                            ui.input_checkbox("collab_show_events", "📌 2008 & 2020 Markers", value=True),
                                            title="Collaboration Plot Options",
                                            placement="bottom",
                                        ),
                                        class_="card-controls-group",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_ui("collab_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card mb-3",
                        ),

                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.tags.span("📋 Matching Collaboration Networks", class_="fw-bold"),
                                    ui.download_button(
                                        "download_collab_csv",
                                        "📥 Download CSV",
                                        class_="btn-sm btn-outline-secondary",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_data_frame("collab_summary_table"),
                            class_="dashboard-plot-card mb-2",
                        ),
                        border=False,
                        fillable=False,
                    ),
                ),

                # --------------------------------------------------------------
                # TAB 3: UPGRADED TOP 10 GLOBAL TRENDS
                # --------------------------------------------------------------
                ui.nav_panel(
                    "🏆 Top 10 Global Trends",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div("🏆 Entity Type", class_="sidebar-section-label"),
                            ui.input_radio_buttons(
                                "top_data_type_filter",
                                None,
                                choices={
                                    "individuals": "🌍 Individual Countries",
                                    "collabs": "🤝 International Collaborations",
                                },
                                selected="individuals",
                            ),
                            ui.div(class_="sidebar-divider"),

                            ui.div("🧪 Chemical Subspace", class_="sidebar-section-label"),
                            ui.input_select(
                                "top_collabs_chem_filter",
                                None,
                                choices=initial_data['chemical_categories'],
                                selected="All",
                                width="100%",
                            ),
                            ui.div(class_="sidebar-divider"),

                            ui.div("📅 Period Range", class_="sidebar-section-label"),
                            ui.input_slider(
                                "top10_years",
                                None,
                                min=initial_data['min_year'],
                                max=initial_data['max_year'],
                                value=[initial_data['min_year'], initial_data['max_year']],
                                step=1,
                                sep="",
                                width="100%",
                            ),
                            ui.div(class_="sidebar-divider"),

                            ui.div("📊 Ranking Basis", class_="sidebar-section-label"),
                            ui.input_radio_buttons(
                                "top10_ranking_basis",
                                None,
                                choices={
                                    "avg": "Period Average Share",
                                    "latest": "End-Year Share",
                                },
                                selected="avg",
                            ),

                            id="top10_sidebar",
                            title="🏆 Top 10 Settings",
                            open="desktop",
                            width=290,
                            padding="12px",
                        ),

                        # 1. 3-Card Executive KPI Strip
                        ui.output_ui("top10_kpis"),

                        # 2. Switchable Multi-View Visualization Card
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.tags.span("📊 Global Top 10 Dynamics", class_="fw-bold"),
                                    ui.div(
                                        ui.input_radio_buttons(
                                            "top10_chart_view",
                                            None,
                                            choices={
                                                "trends": "📈 Share Trends",
                                                "bump": "🔀 Rank Bump Chart",
                                                "shift": "↔️ Period Shift",
                                            },
                                            selected="trends",
                                            inline=True,
                                        ),
                                        class_="card-controls-group",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_ui("top10_main_chart"),
                            full_screen=True,
                            class_="dashboard-plot-card mb-3",
                        ),

                        # 3. Ranked Flag Leaderboard Table + Export
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    ui.tags.span("🏅 Top 10 Ranked Leaderboard", class_="fw-bold"),
                                    ui.download_button(
                                        "download_top10_csv",
                                        "📥 Download CSV",
                                        class_="btn-sm btn-outline-secondary",
                                    ),
                                    class_="card-header-flex",
                                )
                            ),
                            ui.output_data_frame("top10_leaderboard_table"),
                            class_="dashboard-plot-card mb-2",
                        ),
                        border=False,
                        fillable=False,
                    ),
                ),

                # --------------------------------------------------------------
                # TAB 4: KEY ARTICLE FIGURES & ANIMATIONS
                # --------------------------------------------------------------
                ui.nav_panel(
                    "📊 Key Article Figures",
                    ui.div(
                        ui.input_radio_buttons(
                            "article_view_layout",
                            "📐 Layout Mode:",
                            choices={
                                "grid": "📊 Executive Grid",
                                "tabs": "📑 Individual Tabs",
                                "animated_bubbles": "🎬 Animated Bubble Explorer",
                            },
                            selected="grid",
                            inline=True,
                        ),
                        class_="d-flex justify-content-center p-2 bg-light border rounded mb-3",
                    ),
                    ui.output_ui("article_figures_container"),
                ),
            ),
        ),

        # ======================================================================
        # ORIGINAL ARTICLE (MOBILE-SAFE FALLBACK + IFRAME)
        # ======================================================================
        ui.nav_panel(
            "📖 Original Article",
            ui.div(
                ui.div(
                    ui.tags.span("Published Article PDF (Bermúdez-Montaña et al., 2025)", class_="fw-bold"),
                    ui.div(
                        ui.tags.a(
                            "🔗 Open PDF in New Tab",
                            href="chemrxiv-2025-d2zc8.pdf",
                            target="_blank",
                            class_="btn btn-sm btn-primary text-white me-2",
                        ),
                        ui.tags.a(
                            "📥 Download PDF",
                            href="chemrxiv-2025-d2zc8.pdf",
                            download="chemrxiv-2025-d2zc8.pdf",
                            class_="btn btn-sm btn-outline-secondary",
                        ),
                        class_="d-flex flex-wrap gap-1",
                    ),
                    class_="d-flex flex-wrap justify-content-between align-items-center bg-light border rounded p-2 mb-2 gap-2",
                ),
                ui.tags.iframe(
                    src="chemrxiv-2025-d2zc8.pdf",
                    style="width: 100%; height: 78vh; border: 1px solid #cbd5e1; border-radius: 6px;",
                ),
            ),
        ),

        # ======================================================================
        # INTERACTIVE TUTORIAL
        # ======================================================================
        ui.nav_panel(
            "📓 Interactive Tutorial",
            ui.div(
                ui.panel_well(
                    ui.div(
                        ui.div(
                            ui.h4("🧪 Chemical Space Explorer — Interactive Notebook Walkthrough", class_="mb-1 fw-bold"),
                            ui.p(
                                "Modify the parameters inside the live cells below to test how the app loads, filters, and plots data in real time.",
                                class_="text-muted small mb-0",
                            ),
                        ),
                        ui.div(
                            ui.input_radio_buttons(
                                "tut_lang",
                                None,
                                choices={"es": "🇲🇽 Español", "en": "🇬🇧 English"},
                                selected="es",
                                inline=True,
                            ),
                            ui.tags.a(
                                "🚀 Colab",
                                href="https://colab.research.google.com/github/santi-rios/CS-Explorer-App/blob/main/www/Tutorial_Marisol_CS_Explorer.ipynb",
                                target="_blank",
                                class_="btn btn-primary btn-sm ms-2 me-1 text-white fw-bold",
                            ),
                            ui.tags.a(
                                "📥 .ipynb",
                                href="Tutorial_Marisol_CS_Explorer.ipynb",
                                download="Tutorial_Marisol_CS_Explorer.ipynb",
                                class_="btn btn-outline-secondary btn-sm",
                            ),
                            class_="d-flex flex-wrap align-items-center gap-1 mt-2 mt-md-0",
                        ),
                        class_="d-flex flex-wrap justify-content-between align-items-center",
                    ),
                    class_="bg-light border rounded p-3 mb-3 shadow-sm",
                ),
                ui.output_ui("interactive_notebook_body"),
                style="max-width: 1180px; margin: 0 auto;",
            ),
        ),

        # ======================================================================
        # USEFUL LINKS
        # ======================================================================
        ui.nav_panel(
            "🔗 Useful Links",
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("📓 Interactive App Tutorial"),
                    ui.p(
                        "Use the '📓 Interactive Tutorial' tab in the top navigation bar to run live Python cells inside the browser, "
                        "or launch the standalone notebook in Google Colab:"
                    ),
                    ui.div(
                        ui.tags.a(
                            "🚀 Open Interactive Tutorial in Google Colab",
                            href="https://colab.research.google.com/github/santi-rios/CS-Explorer-App/blob/main/www/Tutorial_Marisol_CS_Explorer.ipynb",
                            target="_blank",
                            class_="btn btn-primary btn-sm me-2 mb-2 text-white fw-bold",
                        ),
                        ui.tags.a(
                            "📥 Download Jupyter Notebook (.ipynb)",
                            href="Tutorial_Marisol_CS_Explorer.ipynb",
                            download="Tutorial_Marisol_CS_Explorer.ipynb",
                            class_="btn btn-outline-secondary btn-sm mb-2",
                        ),
                        class_="d-flex flex-wrap align-items-center mt-2",
                    ),
                    class_="dashboard-plot-card",
                ),
                ui.card(
                    ui.card_header("🔗 Project, Data & Author Resources"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.b("Published Article: "),
                            ui.tags.a(
                                "Bermúdez-Montaña et al. (2025) — Quantitative Science Studies (MIT Press)",
                                href="https://doi.org/10.1162/QSS.a.471",
                                target="_blank",
                            ),
                        ),
                        ui.tags.li(
                            ui.tags.b("Raw Data & Article Scripts: "),
                            ui.tags.a(
                                "China's Great Leap Forward in the Chemical Space (GitHub)",
                                href="https://github.com/mbermudezmoTec/China-s-great-leap-forward-in-the-chemical-space",
                                target="_blank",
                            ),
                        ),
                        ui.tags.li(
                            ui.tags.b("Web Application Repository: "),
                            ui.tags.a(
                                "Interactive CS-Explorer-App Source Code (GitHub)",
                                href="https://github.com/santi-rios/CS-Explorer-App",
                                target="_blank",
                            ),
                        ),
                        ui.tags.li(
                            ui.tags.b("Developer Portfolio: "),
                            ui.tags.a(
                                "Site Built by Santiago G-Rios",
                                href="https://santi-rios.github.io/",
                                target="_blank",
                            ),
                        ),
                        class_="mb-0",
                    ),
                    class_="dashboard-plot-card",
                ),
                width="360px",
            ),
        ),

        header=ui.tags.head(
            ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            ui.tags.link(rel="stylesheet", href="styles.css"),
            ui.tags.script(plotly_offline.get_plotlyjs()),
            ui.tags.script("""
                // 1. Viewport detection (mobile vs desktop) + automatic Plotly resize on tab/sidebar transitions
                (function() {
                    var lastIsMobile = null;
                    function reportViewport() {
                        var isMob = window.innerWidth < 768;
                        if (isMob !== lastIsMobile && window.Shiny && Shiny.setInputValue) {
                            lastIsMobile = isMob;
                            Shiny.setInputValue('is_mobile', isMob);
                        }
                    }
                    function resizeVisiblePlots() {
                        if (!window.Plotly) return;
                        document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
                            if (gd.offsetParent !== null) {
                                try { Plotly.Plots.resize(gd); } catch (e) {}
                            }
                        });
                    }
                    document.addEventListener('shiny:connected', reportViewport);
                    var resizeTimer = null;
                    window.addEventListener('resize', function() {
                        clearTimeout(resizeTimer);
                        resizeTimer = setTimeout(function() {
                            reportViewport();
                            resizeVisiblePlots();
                        }, 240);
                    });
                    document.addEventListener('shown.bs.tab', function() {
                        setTimeout(resizeVisiblePlots, 120);
                    });
                    document.addEventListener('shown.bs.collapse', function() {
                        setTimeout(resizeVisiblePlots, 120);
                    });
                    Shiny.addCustomMessageHandler('trigger_plotly_resize', function() {
                        setTimeout(resizeVisiblePlots, 260);
                    });
                    Shiny.addCustomMessageHandler('reset_saved_map_view', function() {
                        window._cs_saved_map_view = null;
                    });
                })();

                // 2. Condense selected country pills to compact "🇨🇳 CHN" across all country pickers
                document.addEventListener('DOMContentLoaded', function() {
                    var pickerIds = ['country_picker', 'collab_country_picker', 'tut_m4_isos'];
                    var patched = {};
                    var checkPickers = setInterval(function() {
                        var allDone = true;
                        pickerIds.forEach(function(id) {
                            if (patched[id]) return;
                            var el = document.getElementById(id);
                            if (el && el.selectize) {
                                var sel = el.selectize;
                                sel.settings.render.item = function(item, escape) {
                                    var shortLabel = String(item.text).split(' — ')[0];
                                    return '<div class="item" title="' + escape(item.text) + '">' + escape(shortLabel) + '</div>';
                                };
                                var currentVals = sel.getValue();
                                sel.setValue(currentVals, true);
                                patched[id] = true;
                            } else {
                                allDone = false;
                            }
                        });
                        if (allDone) clearInterval(checkPickers);
                    }, 120);
                    setTimeout(function() { clearInterval(checkPickers); }, 10000);
                });
            """),
        ),
        title="Chemical Space Explorer 🧪",
        theme=shinyswatch.theme.flatly,
        footer=ui.div(
            ui.hr(),
            ui.p(
                "Source: Bermúdez-Montaña, M., García-Chung, A., Stadler, P. F., Jost, J., & Restrepo, G. (2025). "
                "China's rise in the chemical space and the decline of US influence. ",
                ui.tags.i("Quantitative Science Studies (MIT Press): "),
                ui.tags.a("https://doi.org/10.1162/QSS.a.471", href="https://doi.org/10.1162/QSS.a.471", target="_blank"),
                style="text-align: center; padding: 10px; font-size: 0.88em; color: #475569;",
            ),
        ),
    )

    # ==========================================================================
    # SERVER LOGIC
    # ==========================================================================
    def server(input, output, session):
        selected_countries = reactive.Value(default_top10)

        @reactive.Calc
        def is_mobile_view() -> bool:
            return bool(input.is_mobile()) if input.is_mobile.is_set() else False

        @reactive.Calc
        def country_list():
            return load_country_list()

        # --- TAB 1: Individual Explorer Reactives ---
        @reactive.Calc
        def individual_filtered_data():
            selected_tuple = tuple(sorted(selected_countries.get())) if selected_countries.get() else ()
            return cached_get_display_data(
                selected_isos_tuple=selected_tuple,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode="compare_individuals",
                region_filter=input.region_filter(),
            )

        @reactive.Calc
        def region_all_countries_data():
            all_c = country_list()
            reg = input.region_filter()
            if reg != "All":
                all_c = all_c[all_c['region'] == reg]
            isos = tuple(sorted(all_c['iso2c'].unique().tolist()))
            if not isos:
                return pd.DataFrame()
            return cached_get_display_data(
                selected_isos_tuple=isos,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode="compare_individuals",
                region_filter=reg,
            )

        def _compute_top_isos_for_period(y_start: int, y_end: int, top_n: int = 10) -> list[str]:
            df = get_raw_dataframe()
            reg = input.region_filter()
            chem = input.chemical_category()
            sub = df[
                (df['is_collab'] == False) &
                (df['chemical'] == chem) &
                (df['year'] >= y_start) &
                (df['year'] <= y_end)
            ]
            if reg != "All":
                sub = sub[sub['region'] == reg]
            if sub.empty:
                return []
            return (
                sub.groupby('iso2c')['percentage']
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index.dropna()
                .tolist()
            )

        # Explorer Preset Buttons
        @reactive.Effect
        @reactive.event(input.preset_era_all)
        def _apply_era_all():
            ui.update_slider("years", value=[initial_data['min_year'], initial_data['max_year']])

        @reactive.Effect
        @reactive.event(input.preset_decade_90s)
        def _apply_decade_90s():
            ui.update_slider("years", value=[1996, 2005])
            top_isos = _compute_top_isos_for_period(1996, 2005, top_n=10)
            selected_countries.set(top_isos)
            ui.update_selectize("country_picker", selected=top_isos)

        @reactive.Effect
        @reactive.event(input.preset_decade_crossover)
        def _apply_decade_crossover():
            ui.update_slider("years", value=[2006, 2014])
            top_isos = _compute_top_isos_for_period(2006, 2014, top_n=10)
            selected_countries.set(top_isos)
            ui.update_selectize("country_picker", selected=top_isos)

        @reactive.Effect
        @reactive.event(input.preset_decade_modern)
        def _apply_decade_modern():
            ui.update_slider("years", value=[2015, 2022])
            top_isos = _compute_top_isos_for_period(2015, 2022, top_n=10)
            selected_countries.set(top_isos)
            ui.update_selectize("country_picker", selected=top_isos)

        @reactive.Effect
        @reactive.event(input.preset_top10_current)
        def _apply_top10_current():
            y0, y1 = input.years()
            top_isos = _compute_top_isos_for_period(y0, y1, top_n=10)
            selected_countries.set(top_isos)
            ui.update_selectize("country_picker", selected=top_isos)

        @reactive.Effect
        @reactive.event(input.preset_biggest_risers)
        def _apply_biggest_risers():
            y0, y1 = input.years()
            df = get_raw_dataframe()
            reg = input.region_filter()
            chem = input.chemical_category()
            sub = df[(df['is_collab'] == False) & (df['chemical'] == chem)]
            if reg != "All":
                sub = sub[sub['region'] == reg]
            start_df = sub[sub['year'] == y0].groupby('iso2c')['percentage'].mean()
            end_df = sub[sub['year'] == y1].groupby('iso2c')['percentage'].mean()
            diff = (end_df - start_df).dropna().sort_values(ascending=False)
            risers = diff.head(8).index.tolist()
            if risers:
                selected_countries.set(risers)
                ui.update_selectize("country_picker", selected=risers)

        @reactive.Effect
        @reactive.event(input.country_picker, ignore_none=False, ignore_init=True)
        def _sync_picker_to_selected():
            picker_isos = list(input.country_picker() or [])
            if picker_isos != selected_countries.get():
                selected_countries.set(picker_isos)

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

        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])
            ui.update_selectize("country_picker", selected=[])

        @reactive.Effect
        @reactive.event(input.region_filter, ignore_init=True)
        async def _handle_region_change():
            await session.send_custom_message("reset_saved_map_view", {})
            all_countries = country_list()
            current_region = input.region_filter()
            region_df = all_countries[all_countries['region'] == current_region] if current_region != "All" else all_countries
            new_choices = {
                row['iso2c']: f"{iso_to_flag_emoji(row['iso2c'])} {str(row['iso3c']).upper()} — {row['country']}"
                for _, row in region_df.iterrows()
            }
            valid_selected = [iso for iso in selected_countries.get() if iso in new_choices]
            if not valid_selected:
                y0, y1 = input.years()
                valid_selected = _compute_top_isos_for_period(y0, y1, top_n=8)
            selected_countries.set(valid_selected)
            ui.update_selectize("country_picker", choices=new_choices, selected=valid_selected)

        @reactive.Effect
        @reactive.event(input.explorer_sidebar)
        async def _resize_on_explorer_sidebar():
            await session.send_custom_message("trigger_plotly_resize", {})

        @output
        @render.text
        def selection_info():
            count = len(selected_countries.get())
            current_region = input.region_filter()
            all_countries = country_list()
            avail = len(all_countries[all_countries['region'] == current_region] if current_region != "All" else all_countries)
            return f"Selected {count} of {avail} countries ({current_region})"

        @output
        @render.ui
        def main_plot():
            selected = selected_countries.get()
            data = individual_filtered_data()
            mob = is_mobile_view()
            if data.empty:
                return figure_to_html(create_empty_plot(
                    "Select one or more countries from the Filters panel or map to view trends.",
                    is_mobile=mob,
                ))
            return figure_to_html(create_trends_plot(
                data=data,
                selected_countries=selected,
                mode="compare_individuals",
                show_flags=bool(input.show_flags_on_trends()),
                log_scale=bool(input.trends_log_scale()),
                show_events=bool(input.trends_show_events()),
                is_mobile=mob,
            ))

        @output
        @render.ui
        def dynamic_explorer_kpis():
            data = individual_filtered_data()
            y_start, y_end = input.years()
            if data.empty:
                return ui.div()

            by_country = data.groupby(['iso2c', 'country'])['total_percentage'].agg(['mean', 'max']).reset_index()
            by_country = by_country.sort_values('mean', ascending=False)
            top_row = by_country.iloc[0]
            leader_flag = iso_to_flag_emoji(top_row['iso2c'])

            latest_yr = int(data['year'].max())
            first_yr = int(data['year'].min())
            latest_sum = float(data[data['year'] == latest_yr]['total_percentage'].sum())

            # Fastest mover among selected countries
            start_s = data[data['year'] == first_yr].groupby(['iso2c', 'country'])['total_percentage'].mean()
            end_s = data[data['year'] == latest_yr].groupby(['iso2c', 'country'])['total_percentage'].mean()
            shift_s = (end_s - start_s).dropna().sort_values(ascending=False)
            if not shift_s.empty and first_yr != latest_yr:
                m_iso, m_name = shift_s.index[0]
                mover_val = f"{iso_to_flag_emoji(m_iso)} {m_name} ({shift_s.iloc[0]:+.1f} pp)"
                mover_sub = f"Largest net change ({first_yr}→{latest_yr})"
            else:
                mover_val = f"{len(by_country)} Countries Active"
                mover_sub = f"Single-year snapshot ({latest_yr})"

            return ui.layout_column_wrap(
                ui.div(
                    ui.div(f"{leader_flag} Period Leader ({y_start}–{y_end})", class_="kpi-card-label"),
                    ui.div(f"{top_row['country']}: {top_row['mean']:.2f}% avg", class_="kpi-card-value"),
                    ui.div(f"Peak: {top_row['max']:.1f}%", class_="kpi-card-sub"),
                    class_="kpi-card kpi-teal",
                ),
                ui.div(
                    ui.div(f"📈 Combined Share ({latest_yr})", class_="kpi-card-label"),
                    ui.div(f"{latest_sum:.2f}%", class_="kpi-card-value"),
                    ui.div(f"Across {len(by_country)} selected countries", class_="kpi-card-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div("🚀 Top Gainer in Selection", class_="kpi-card-label"),
                    ui.div(mover_val, class_="kpi-card-value"),
                    ui.div(mover_sub, class_="kpi-card-sub"),
                    class_="kpi-card kpi-indigo",
                ),
                width="230px",
                class_="mb-3",
            )

        @output
        @render.ui
        def unified_map_output():
            mode = input.map_view_mode()
            mob = is_mobile_view()
            if mode == "interactive":
                all_countries = country_list()
                current_region = input.region_filter()
                filtered_countries = (
                    all_countries[all_countries['region'] == current_region]
                    if current_region != "All" else all_countries
                )
                m = create_enhanced_folium_map(
                    country_list=filtered_countries,
                    selected_countries=selected_countries.get(),
                    contributions_df=region_all_countries_data(),
                    map_style=input.map_visual_style(),
                    is_mobile=mob,
                )
                return ui.div(ui.HTML(m._repr_html_()), class_="folium-map-container")

            choropleth_data = region_all_countries_data()
            if choropleth_data.empty:
                return figure_to_html(create_empty_plot("No data for global choropleth.", is_mobile=mob))

            y_start, y_end = input.years()
            chem_cat = input.chemical_category()
            if mode == "static":
                return figure_to_html(create_contribution_map_plot(
                    choropleth_data,
                    fill_label="Avg Share",
                    main_title=f"Average CS Share ({y_start}–{y_end}) — {chem_cat}",
                    is_mobile=mob,
                ))
            return figure_to_html(create_contribution_choropleth(choropleth_data, is_mobile=mob))

        @output
        @render.data_frame
        def individual_summary_table():
            data = individual_filtered_data()
            if data.empty:
                return pd.DataFrame({"Message": ["No countries selected."]})
            return create_summary_dataframe(data, "compare_individuals")

        @render.download(filename=lambda: f"CS_Explorer_Countries_{input.chemical_category()}_{input.years()[0]}_{input.years()[1]}.csv")
        def download_individual_csv():
            yield create_summary_dataframe(individual_filtered_data(), "compare_individuals").to_csv(index=False)

        # --- TAB 2: DEDICATED COLLABORATIONS EXPLORER ---
        @reactive.Effect
        @reactive.event(input.sync_from_explorer)
        def _sync_collab_from_explorer():
            isos = selected_countries.get()
            if isos:
                ui.update_selectize("collab_country_picker", selected=isos)
            ui.update_slider("collab_years", value=list(input.years()))
            ui.update_select("collab_chem", selected=input.chemical_category())

        @reactive.Effect
        @reactive.event(input.collab_clear)
        def _clear_collab_picker():
            ui.update_selectize("collab_country_picker", selected=[])

        @reactive.Effect
        @reactive.event(input.collab_preset, ignore_init=True)
        def _apply_collab_preset():
            preset = input.collab_preset()
            preset_map = {
                "cn_us": (["CN", "US"], "all_selected"),
                "top6": (["CN", "US", "DE", "GB", "JP", "IN"], "any_pair"),
                "transatlantic": (["US", "DE", "GB", "FR"], "any_pair"),
                "asia_hub": (["CN", "JP", "IN", "KR"], "any_pair"),
                "latam": (["MX", "BR", "US", "ES"], "any_pair"),
            }
            if preset in preset_map:
                isos, match_mode = preset_map[preset]
                ui.update_selectize("collab_country_picker", selected=isos)
                ui.update_radio_buttons("collab_match_mode", selected=match_mode)

        @reactive.Calc
        def collab_tab_data():
            isos = tuple(sorted(input.collab_country_picker() or []))
            if not isos:
                return pd.DataFrame()
            return cached_get_display_data(
                selected_isos_tuple=isos,
                year_range=tuple(input.collab_years()),
                chemical_category=input.collab_chem(),
                display_mode="find_collaborations",
                region_filter="All",
                collab_match_mode=input.collab_match_mode(),
                collab_tier_filter=input.collab_tier_filter(),
            )

        @output
        @render.ui
        def collab_kpis():
            df_c = collab_tab_data()
            if df_c.empty:
                return ui.div()

            n_networks = int(df_c['plot_group'].nunique())
            by_net = df_c.groupby(['plot_group', 'iso2c'])['total_percentage'].mean().sort_values(ascending=False)
            top_name, top_iso = by_net.index[0]
            top_avg = float(by_net.iloc[0])

            tier_counts = df_c.drop_duplicates('plot_group')['collab_type'].value_counts()
            n_bi = int(tier_counts.get("Bilateral", 0))
            n_multi = n_networks - n_bi

            return ui.layout_column_wrap(
                ui.div(
                    ui.div("🤝 Matching Networks", class_="kpi-card-label"),
                    ui.div(f"{n_networks:,}", class_="kpi-card-value"),
                    ui.div(f"{n_bi} Bilateral · {n_multi} Multilateral", class_="kpi-card-sub"),
                    class_="kpi-card kpi-indigo",
                ),
                ui.div(
                    ui.div("🏆 Strongest Partnership", class_="kpi-card-label"),
                    ui.div(f"{top_iso} ({top_avg:.3f}% avg)", class_="kpi-card-value"),
                    ui.div(top_name, class_="kpi-card-sub"),
                    class_="kpi-card kpi-teal",
                ),
                ui.div(
                    ui.div("🧪 Active Subspace & Period", class_="kpi-card-label"),
                    ui.div(f"{input.collab_chem()}", class_="kpi-card-value"),
                    ui.div(f"{input.collab_years()[0]}–{input.collab_years()[1]}", class_="kpi-card-sub"),
                    class_="kpi-card kpi-blue",
                ),
                width="230px",
                class_="mb-3",
            )

        @output
        @render.ui
        def collab_plot():
            isos = list(input.collab_country_picker() or [])
            mob = is_mobile_view()
            if not isos:
                return figure_to_html(create_empty_plot(
                    "Select 1 or more countries in the left panel (or pick an Example Network) to explore joint collaborations.",
                    is_mobile=mob,
                ))

            df_c = collab_tab_data()
            if df_c.empty:
                hint = (
                    "Try switching Matching Logic to 'Shared networks (≥2 selected)'."
                    if input.collab_match_mode() == "all_selected" and len(isos) > 2
                    else "Try broadening the Year Range, Network Tier, or Chemical Subspace."
                )
                return figure_to_html(create_empty_plot(
                    f"No matching collaborations found for {', '.join(isos)}.\n{hint}",
                    is_mobile=mob,
                ))

            return figure_to_html(create_trends_plot(
                data=df_c,
                selected_countries=isos,
                mode="find_collaborations",
                show_flags=False,
                log_scale=bool(input.collab_log_scale()),
                show_events=bool(input.collab_show_events()),
                max_collab_traces=int(input.collab_max_traces()),
                is_mobile=mob,
            ))

        @output
        @render.data_frame
        def collab_summary_table():
            df_c = collab_tab_data()
            if df_c.empty:
                return pd.DataFrame({"Message": ["No matching collaborations for current filters."]})
            return create_summary_dataframe(df_c, "find_collaborations")

        @render.download(filename=lambda: f"CS_Explorer_Collaborations_{input.collab_chem()}_{input.collab_years()[0]}_{input.collab_years()[1]}.csv")
        def download_collab_csv():
            yield create_summary_dataframe(collab_tab_data(), "find_collaborations").to_csv(index=False)

        # --- TAB 3: UPGRADED TOP 10 GLOBAL TRENDS ---
        @reactive.Calc
        def top10_analytics():
            df = get_raw_dataframe()
            is_collab = (input.top_data_type_filter() == "collabs")
            return compute_top10_analytics(
                df=df,
                is_collab=is_collab,
                chemical_category=input.top_collabs_chem_filter(),
                year_range=tuple(input.top10_years()),
                ranking_basis=input.top10_ranking_basis(),
            )

        @output
        @render.ui
        def top10_kpis():
            res = top10_analytics()
            if res.get("empty"):
                return ui.div()
            k = res["kpis"]
            return ui.layout_column_wrap(
                ui.div(
                    ui.div(f"🥇 #1 Leader in {k['actual_end']}", class_="kpi-card-label"),
                    ui.div(f"{k['leader_label']} — {k['leader_val']}", class_="kpi-card-value"),
                    ui.div(k['leader_sub'], class_="kpi-card-sub"),
                    class_="kpi-card kpi-teal",
                ),
                ui.div(
                    ui.div(f"🚀 Biggest Gainer ({k['actual_start']}→{k['actual_end']})", class_="kpi-card-label"),
                    ui.div(f"{k['gainer_label']} ({k['gainer_val']})", class_="kpi-card-value"),
                    ui.div(k['gainer_sub'], class_="kpi-card-sub"),
                    class_="kpi-card kpi-blue",
                ),
                ui.div(
                    ui.div(f"📊 Top 10 Combined Share ({k['actual_end']})", class_="kpi-card-label"),
                    ui.div(k['conc_end'], class_="kpi-card-value"),
                    ui.div(k['conc_delta'], class_="kpi-card-sub"),
                    class_="kpi-card kpi-amber",
                ),
                width="230px",
                class_="mb-3",
            )

        @output
        @render.ui
        def top10_main_chart():
            res = top10_analytics()
            mob = is_mobile_view()
            if res.get("empty"):
                return figure_to_html(create_empty_plot("No Top 10 data available for current filters.", is_mobile=mob))

            is_collab = (input.top_data_type_filter() == "collabs")
            chem = input.top_collabs_chem_filter()
            k = res["kpis"]
            entity_lbl = "Collaborations" if is_collab else "Countries"
            chart_view = input.top10_chart_view()

            if chart_view == "bump":
                return figure_to_html(create_top10_bump_chart(
                    top_df=res["top_df"],
                    lb_raw=res["lb_raw"],
                    title=f"Global Rank Trajectory: Top 10 {entity_lbl} ({chem})",
                    is_mobile=mob,
                ))
            if chart_view == "shift":
                return figure_to_html(create_top10_shift_chart(
                    lb_raw=res["lb_raw"],
                    start_yr=k["actual_start"],
                    end_yr=k["actual_end"],
                    title=f"Period Share Shift ({k['actual_start']} → {k['actual_end']}) — {chem}",
                    is_mobile=mob,
                ))
            return figure_to_html(create_top_trends_plot(
                data=res["top_df"],
                title=f"Top 10 {entity_lbl} Share Trends — {chem}",
                is_collab=is_collab,
                show_flags=not is_collab,
                is_mobile=mob,
            ))

        @output
        @render.data_frame
        def top10_leaderboard_table():
            res = top10_analytics()
            if res.get("empty"):
                return pd.DataFrame({"Message": ["No data available."]})
            return res["lb_display"]

        @render.download(filename=lambda: f"CS_Explorer_Top10_{input.top_data_type_filter()}_{input.top_collabs_chem_filter()}.csv")
        def download_top10_csv():
            res = top10_analytics()
            df_out = pd.DataFrame() if res.get("empty") else res["lb_display"]
            yield df_out.to_csv(index=False)

        # --- TAB 4: KEY ARTICLE FIGURES ---
        @output
        @render.ui
        def article_figures_container():
            layout_mode = input.article_view_layout()

            if layout_mode == "animated_bubbles":
                return ui.card(
                    ui.card_header("🎬 Animated Global Ranking & Bubble Share (1996–2022)"),
                    ui.layout_column_wrap(
                        ui.input_select(
                            "anim_bubble_chem",
                            "🧪 Chemical Subspace:",
                            choices=initial_data['chemical_categories'],
                            selected="All",
                            width="100%",
                        ),
                        ui.input_slider(
                            "anim_top_n",
                            "🌍 Top Countries to Track:",
                            min=10, max=30, value=15, step=5,
                            width="100%",
                        ),
                        width="260px",
                    ),
                    ui.output_ui("animated_bubble_plot"),
                    ui.p(
                        "Press ▶ Play below the chart to watch China's surge from #6 in 1996 to #1 from 2013 onward.",
                        class_="text-center text-muted small mt-1 mb-0",
                    ),
                    full_screen=True,
                    class_="dashboard-plot-card",
                )

            if layout_mode == "grid":
                return ui.div(
                    ui.layout_column_wrap(
                        ui.card(
                            ui.card_header("🏆 Main Countries in the CS (Fig. 1a)"),
                            ui.output_ui("country_cs_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card",
                        ),
                        ui.card(
                            ui.card_header("🥼 China–US Domestic vs. Collaborative Share (Fig. 1c)"),
                            ui.output_ui("china_us_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card",
                        ),
                        width="440px",
                        class_="mb-3",
                    ),
                    ui.layout_column_wrap(
                        ui.card(
                            ui.card_header("🧪 Chemical Space Expansion (Fig. 2a — Log)"),
                            ui.output_ui("article_cs_expansion_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card",
                        ),
                        ui.card(
                            ui.card_header("👥 R&D Researchers Headcount (Fig. 1e)"),
                            ui.output_ui("article_researchers_plot"),
                            full_screen=True,
                            class_="dashboard-plot-card",
                        ),
                        width="440px",
                        class_="mb-3",
                    ),
                    ui.card(
                        ui.card_header("💰 Annual GDP per Capita Growth Rate (Fig. 1d)"),
                        ui.output_ui("article_gdp_plot"),
                        full_screen=True,
                        class_="dashboard-plot-card",
                    ),
                )

            return ui.navset_card_tab(
                ui.nav_panel("🏆 Main Countries", ui.output_ui("country_cs_plot")),
                ui.nav_panel("🥼 China–US", ui.output_ui("china_us_plot")),
                ui.nav_panel("🧪 CS Expansion", ui.output_ui("article_cs_expansion_plot")),
                ui.nav_panel("👥 Researchers", ui.output_ui("article_researchers_plot")),
                ui.nav_panel("💰 GDP Growth", ui.output_ui("article_gdp_plot")),
            )

        @output
        @render.ui
        def animated_bubble_plot():
            df = get_raw_dataframe()
            chem = input.anim_bubble_chem() if input.anim_bubble_chem.is_set() else "All"
            top_n = int(input.anim_top_n()) if input.anim_top_n.is_set() else 15
            return figure_to_html(create_animated_bubble_figure(
                df, chemical_category=chem, top_n=top_n, is_mobile=is_mobile_view()
            ))

        @output
        @render.ui
        def country_cs_plot():
            article_data = load_article_data()
            df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
            if df_filtered.empty:
                return figure_to_html(create_empty_plot("No country participation data available", is_mobile=is_mobile_view()))
            return figure_to_html(create_article_plot(
                df_filtered, "Country participation in the growth of the CS", is_mobile=is_mobile_view()
            ))

        @output
        @render.ui
        def china_us_plot():
            article_data = load_article_data()
            cs_data = article_data[article_data['source'] == "China-US in the CS"]
            if cs_data.empty:
                return figure_to_html(create_empty_plot("No 'China-US in the CS' data found", is_mobile=is_mobile_view()))
            return figure_to_html(create_china_us_dual_axis_plot(cs_data, is_mobile=is_mobile_view()))

        @output
        @render.ui
        def article_gdp_plot():
            article_data = load_article_data()
            gdp_data = article_data[article_data['source'] == "Annual growth rate of the GDP"]
            if gdp_data.empty:
                return figure_to_html(create_empty_plot("No GDP data available", is_mobile=is_mobile_view()))
            return figure_to_html(create_gdp_plot(gdp_data, is_mobile=is_mobile_view()))

        @output
        @render.ui
        def article_researchers_plot():
            article_data = load_article_data()
            researchers_data = article_data[article_data['source'] == "Number of Researchers"]
            if researchers_data.empty:
                return figure_to_html(create_empty_plot("No researchers data available", is_mobile=is_mobile_view()))
            return figure_to_html(create_researchers_plot(researchers_data, is_mobile=is_mobile_view()))

        @output
        @render.ui
        def article_cs_expansion_plot():
            article_data = load_article_data()
            cs_data = article_data[article_data['source'] == "Expansion of the CS"]
            if cs_data.empty:
                return figure_to_html(create_empty_plot("No CS expansion data available", is_mobile=is_mobile_view()))
            return figure_to_html(create_cs_expansion_plot(cs_data, is_mobile=is_mobile_view()))

        # --- INTERACTIVE TUTORIAL CELLS ---
        @output
        @render.ui
        def interactive_notebook_body():
            es = (input.tut_lang() == "es")
            return ui.div(
                ui.div(
                    ui.div(
                        "🏗️ Módulo 1 y 2: Arquitectura General y Preparación del Entorno (Git)"
                        if es else "🏗️ Modules 1 & 2: General Architecture & Environment Setup (Git)",
                        class_="jp-cell-header",
                    ),
                    ui.div(
                        ui.p(
                            "La aplicación se divide en 4 capas: (1) Base de datos columnar Apache Parquet (data/data.parquet) con 189 países y colaboraciones, "
                            "(2) Motor de filtrado y graficación en utils/functions.py, (3) Interfaz reactiva Shiny en app.py, y (4) Contenedores Docker + Cloudflare Tunnel."
                            if es else
                            "The app is structured into 4 layers: (1) Apache Parquet database (data/data.parquet) covering 189 countries and collaborations, "
                            "(2) Filtering and plotting engine in utils/functions.py, (3) Reactive Shiny UI in app.py, and (4) Docker + Cloudflare Tunnel stack.",
                            class_="p-3 mb-0",
                        ),
                        ui.div(
                            ui.div("In [1]:", class_="jp-prompt-in"),
                            ui.pre(
                                "# Clonar el repositorio e instalar dependencias / Clone repo & install dependencies\n"
                                "git clone https://github.com/santi-rios/CS-Explorer-App.git\n"
                                "cd CS-Explorer-App\n"
                                "pip install -r requirements.txt\n"
                                "shiny run app.py --port 8888",
                            ),
                            class_="jp-input-area",
                        ),
                    ),
                    class_="jp-cell",
                ),

                ui.div(
                    ui.div(
                        "🗄️ Módulo 3: Cómo se Manejan los Datos (Prueba la celda en vivo)"
                        if es else "🗄️ Module 3: Data Management & Caching (Try live cell)",
                        class_="jp-cell-header",
                    ),
                    ui.div(
                        ui.layout_column_wrap(
                            ui.input_radio_buttons(
                                "tut_m3_subset",
                                "Subconjunto en data.parquet:" if es else "Subset of data.parquet:",
                                choices={
                                    "solo": "Países individuales (is_collab == False)" if es else "Individual Countries (is_collab == False)",
                                    "collab": "Colaboraciones (is_collab == True)" if es else "Collaborations (is_collab == True)",
                                },
                                selected="solo",
                                inline=True,
                            ),
                            ui.input_slider(
                                "tut_m3_rows",
                                "Filas (.head(n)):" if es else "Rows (.head(n)):",
                                min=3, max=15, value=5, step=1, width="100%",
                            ),
                            width="280px",
                            class_="p-3 bg-light border-bottom",
                        ),
                        ui.output_ui("tut_m3_code_and_output"),
                    ),
                    class_="jp-cell",
                ),

                ui.div(
                    ui.div(
                        "📈 Módulos 4 y 5: Filtrado (get_display_data) y Gráficas Interactivas (create_trends_plot)"
                        if es else "📈 Modules 4 & 5: Filtering Pipeline & Interactive Plotly Charts",
                        class_="jp-cell-header",
                    ),
                    ui.div(
                        ui.layout_column_wrap(
                            ui.div(
                                ui.input_selectize(
                                    "tut_m4_isos",
                                    "1. Países (selected_isos):" if es else "1. Countries (selected_isos):",
                                    choices=initial_data['country_choices'],
                                    selected=["CN", "US", "DE", "JP"],
                                    multiple=True,
                                    width="100%",
                                ),
                                class_="compact-iso-picker",
                            ),
                            ui.input_select(
                                "tut_m4_chem",
                                "2. Subespacio (chemical_category):" if es else "2. Subspace (chemical_category):",
                                choices=initial_data['chemical_categories'],
                                selected="All",
                                width="100%",
                            ),
                            ui.input_radio_buttons(
                                "tut_m4_mode",
                                "3. Modo (display_mode):" if es else "3. Mode (display_mode):",
                                choices={
                                    "compare_individuals": "compare_individuals",
                                    "find_collaborations": "find_collaborations",
                                },
                                selected="compare_individuals",
                            ),
                            width="250px",
                            class_="p-3 bg-light border-bottom",
                        ),
                        ui.output_ui("tut_m4_m5_code_and_output"),
                    ),
                    class_="jp-cell",
                ),

                ui.div(
                    ui.div(
                        "⚡ Módulos 6 y 7: Reactividad en Shiny y Administración del Servidor"
                        if es else "⚡ Modules 6 & 7: Shiny Reactivity & Server Administration",
                        class_="jp-cell-header",
                    ),
                    ui.div(
                        ui.p(
                            "En Shiny, cada control visual (input) dispara automáticamente los bloques @reactive.Calc y redibuja las salidas @output @render.ui. "
                            "Para actualizar la aplicación en producción tras hacer git push, ejecuta en el servidor:"
                            if es else
                            "In Shiny, every UI input automatically triggers @reactive.Calc functions and re-renders @output @render.ui blocks. "
                            "To update the production server after git push, run on the server:",
                            class_="p-3 mb-0",
                        ),
                        ui.div(
                            ui.div("In [4]:", class_="jp-prompt-in"),
                            ui.pre(
                                "# Atajos de administración en el servidor Linux (definidos en ~/.bashrc):\n"
                                "app-update      # Hace git pull y recrea el contenedor shiny-chemical en segundos\n"
                                "app-rebuild     # Reconstruye la imagen Docker desde cero si cambió requirements.txt\n"
                                "tunnel-restart  # Reinicia el conector de Cloudflare Tunnel",
                            ),
                            class_="jp-input-area",
                        ),
                    ),
                    class_="jp-cell",
                ),
            )

        @output
        @render.ui
        def tut_m3_code_and_output():
            df = get_raw_dataframe()
            is_collab_val = (input.tut_m3_subset() == "collab")
            n_rows = int(input.tut_m3_rows())
            sub_df = df[df['is_collab'] == is_collab_val][
                ['year', 'country', 'iso2c', 'region', 'chemical', 'percentage']
            ].head(n_rows)

            code_str = (
                "from utils.functions import get_raw_dataframe\n"
                "df = get_raw_dataframe('./data/data.parquet')\n"
                f"subset = df[df['is_collab'] == {is_collab_val}]\n"
                f"subset[['year', 'country', 'iso2c', 'region', 'chemical', 'percentage']].head({n_rows})"
            )
            return ui.div(
                ui.div(ui.div("In [2]:", class_="jp-prompt-in"), ui.pre(code_str), class_="jp-input-area"),
                ui.div(
                    ui.div("Out[2]:", class_="jp-prompt-out"),
                    ui.div(
                        ui.HTML(sub_df.to_html(index=False, classes="table table-sm table-striped table-bordered mb-0")),
                        class_="w-100",
                    ),
                    class_="jp-output-area",
                ),
            )

        @output
        @render.ui
        def tut_m4_m5_code_and_output():
            isos = list(input.tut_m4_isos() or ["CN", "US"])
            chem = input.tut_m4_chem()
            mode = input.tut_m4_mode()

            df_filtered = cached_get_display_data(
                selected_isos_tuple=tuple(sorted(isos)),
                year_range=(1996, 2022),
                chemical_category=chem,
                display_mode=mode,
                region_filter="All",
            )
            fig = create_trends_plot(
                data=df_filtered,
                selected_countries=isos,
                mode=mode,
                show_flags=(mode == "compare_individuals"),
                is_mobile=is_mobile_view(),
            )

            code_str = (
                "from utils.functions import get_display_data, create_trends_plot\n\n"
                f"filtered = get_display_data(\n"
                f"    df=df,\n"
                f"    selected_isos={isos},\n"
                f"    year_range=(1996, 2022),\n"
                f"    chemical_category='{chem}',\n"
                f"    display_mode='{mode}'\n"
                f")\n"
                f"fig = create_trends_plot(filtered, selected_countries={isos}, mode='{mode}')\n"
                "fig.show()"
            )
            return ui.div(
                ui.div(ui.div("In [3]:", class_="jp-prompt-in"), ui.pre(code_str), class_="jp-input-area"),
                ui.div(
                    ui.div("Out[3]:", class_="jp-prompt-out"),
                    ui.div(figure_to_html(fig), class_="w-100"),
                    class_="jp-output-area",
                ),
            )

    return App(app_ui, server, static_assets=www_dir)


app = create_app()

if __name__ == "__main__":
    app.run(port=8000)