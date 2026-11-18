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
    create_empty_plot,
    create_gdp_plot,
    create_researchers_plot,
    create_cs_expansion_plot,
    create_china_us_dual_axis_plot,
    get_raw_dataframe,
    load_article_data,
    create_animated_bubble_figure,
    iso_to_flag_emoji
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
    region_filter: str = "All"
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
        country_list=country_list
    )


def figure_to_html(fig):
    """Render Plotly figure as responsive embedded HTML."""
    return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False, config={"responsive": True}))


def create_app():
    default_top10 = get_default_top10_isos()
    try:
        df_sample = get_raw_dataframe()[['chemical', 'year', 'region']]
        unique_chemicals = df_sample['chemical'].dropna().unique().tolist()
        chemical_categories = sorted([chem for chem in unique_chemicals if chem and str(chem).strip()])
        chemical_categories = list(dict.fromkeys(chemical_categories))

        unique_regions = df_sample['region'].fillna('Other').unique().tolist()
        regions = ['All'] + sorted(list(set(
            region for region in unique_regions if region and str(region).strip() and region != 'All'
        )))

        min_year = int(df_sample['year'].min())
        max_year = int(df_sample['year'].max())

        countries_df = load_country_list()
        # Format label as "🇨🇳 CHN — China" so Selectize can display just "🇨🇳 CHN" in the selected pill
        # while still letting the user search by full country name ("China") in the dropdown list!
        country_choices = {
            row['iso2c']: f"{iso_to_flag_emoji(row['iso2c'])} {str(row['iso3c']).upper()} — {row['country']}"
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
            'country_choices': {}
        }

    app_ui = ui.page_navbar(
        ui.nav_panel(
            "Dashboard",
            # --- Title Banner ---
            ui.div(
                ui.h3(
                    "China's Rise in the Chemical Space & the Decline of US Influence",
                    style="text-align: center; font-weight: 700; margin-top: 10px; margin-bottom: 4px; color: #0f172a;"
                ),
                ui.p(
                    "Interactive geopolitical explorer of 15.7 million new chemical substances across 189 countries and territories (1996–2022) — ",
                    ui.tags.a(
                        "Bermúdez-Montaña et al. (2025), Quantitative Science Studies",
                        href="https://doi.org/10.1162/QSS.a.471",
                        target="_blank",
                        class_="article-doi-link"
                    ),
                    style="text-align: center; color: #334155; margin-bottom: 14px;"
                ),
            ),

            # --- Main Navigation Tabs ---
            ui.navset_card_tab(
                # TAB 1: Unified Interactive Explorer
                ui.nav_panel(
                    "🗺️ Interactive Explorer",

                    # 1. FIRST ELEMENT: COMPACT COUNTRY SELECTION & QUICK PRESETS BAR
                    ui.panel_well(
                        ui.row(
                            ui.column(
                                6,
                                ui.input_selectize(
                                    "country_picker",
                                    "🔍 Select Countries (search by name or ISO3, or click on the map below):",
                                    choices=initial_data['country_choices'],
                                    selected=default_top10,
                                    multiple=True,
                                    remove_button=True,
                                    width="100%",
                                    options={"placeholder": "Type country name or ISO code..."}
                                )
                            ),
                            ui.column(
                                6,
                                ui.div(
                                    ui.div(
                                        ui.tags.span("⚡ Quick Presets:", style="font-weight: 700; font-size: 0.85rem; color: #1e293b;"),
                                        ui.tags.span(ui.output_text("selection_info"), class_="text-muted small ms-2"),
                                        class_="d-flex align-items-center justify-content-between mb-1"
                                    ),
                                    ui.div(
                                        ui.input_action_button(
                                            "preset_top10_current",
                                            "🏆 Top 10",
                                            class_="btn-sm btn-primary me-1 mb-1"
                                        ),
                                        ui.input_action_button(
                                            "preset_decade_90s",
                                            "🕰️ 1996–2005",
                                            class_="btn-sm btn-outline-secondary me-1 mb-1"
                                        ),
                                        ui.input_action_button(
                                            "preset_decade_crossover",
                                            "⚡ 2006–2014",
                                            class_="btn-sm btn-outline-secondary me-1 mb-1"
                                        ),
                                        ui.input_action_button(
                                            "preset_decade_modern",
                                            "🚀 2015–2022",
                                            class_="btn-sm btn-outline-secondary me-1 mb-1"
                                        ),
                                        ui.input_action_button(
                                            "preset_biggest_risers",
                                            "🔥 Top 8 Risers",
                                            class_="btn-sm btn-outline-success me-1 mb-1"
                                        ),
                                        ui.input_action_button(
                                            "clear_selection",
                                            "🗑️ Clear",
                                            class_="btn-sm btn-outline-danger me-2 mb-1"
                                        ),
                                        ui.input_checkbox(
                                            "show_flags_on_trends",
                                            "🏳️ Flags on Lines",
                                            value=True
                                        ),
                                        class_="d-flex flex-wrap align-items-center"
                                    )
                                )
                            )
                        ),
                        class_="bg-light border rounded p-2 mb-3 shadow-sm"
                    ),

                    # 2. SECOND ELEMENT: FULL-WIDTH MAIN PLOT + COLLAPSIBLE SIDE-BY-SIDE CHOROPLETH DRAWER
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.div(
                                    ui.tags.span("📈 Individual Country Contribution Trends", style="font-weight: 700; color: #0f172a;"),
                                    ui.tags.small(
                                        "Drag on chart to zoom into any period; double-click to reset.",
                                        class_="text-muted ms-2"
                                    )
                                ),
                                ui.input_action_button(
                                    "toggle_choropleth_drawer",
                                    "🌎 Toggle Side-by-Side Global Choropleth",
                                    class_="btn-sm btn-outline-primary"
                                ),
                                class_="d-flex align-items-center justify-content-between w-100"
                            )
                        ),
                        ui.layout_sidebar(
                            ui.sidebar(
                                ui.div(
                                    ui.div(
                                        ui.tags.strong("🌎 Global Choropleth Snapshot"),
                                        ui.tags.small(
                                            "Light Blue (low) to Deep Blue / Crimson Red (dominant)",
                                            class_="d-block text-muted mb-2"
                                        )
                                    ),
                                    ui.input_radio_buttons(
                                        "snapshot_mode",
                                        "🗺️ Choropleth Mode:",
                                        choices={
                                            "static": "🗺️ Period Average (Static)",
                                            "animated": "🎬 Yearly Evolution (1996–2022)"
                                        },
                                        selected="animated",
                                        inline=True
                                    ),
                                    class_="snapshot-controls mb-2 pb-2 border-bottom"
                                ),
                                ui.output_ui("contribution_map"),
                                id="choropleth_right_sidebar",
                                position="right",
                                open="closed",
                                width="48%",
                                bg="#f8f9fa",
                                padding="12px"
                            ),
                            ui.output_ui("main_plot"),
                            fillable=True,
                            border=False
                        ),
                        full_screen=True,
                        fill=True,
                        class_="dashboard-plot-card mb-3 shadow-sm"
                    ),

                    # 3. THIRD ELEMENT: 100% FULL-WIDTH INTERACTIVE GEOPOLITICAL MAP
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.div(
                                    ui.tags.span("🌐 Interactive Geopolitical Map & Country Selector", style="font-weight: 700; color: #0f172a;"),
                                    ui.tags.small(
                                        "Click any country on the map to add/remove it (zoom & pan position are preserved).",
                                        class_="text-muted ms-2"
                                    )
                                ),
                                ui.div(
                                    ui.input_radio_buttons(
                                        "map_visual_style",
                                        None,
                                        choices={
                                            "polygons": "🗺️ Shaded Country Boundaries",
                                            "flags": "🏳️ Proportional Flag Badges",
                                            "bubbles": "🟣 Proportional Bubbles (Superzip)"
                                        },
                                        selected="polygons",
                                        inline=True
                                    ),
                                    style="margin-bottom: -6px;"
                                ),
                                class_="d-flex flex-wrap align-items-center justify-content-between w-100"
                            )
                        ),
                        ui.output_ui("map_output"),
                        full_screen=True,
                        class_="border rounded mb-3 shadow-sm"
                    ),

                    # 4. FOURTH ELEMENT: FILTERING CONTROLS & LIVE KPI INFORMATION BOXES
                    ui.panel_well(
                        ui.row(
                            ui.column(
                                3,
                                ui.input_select(
                                    "region_filter", "🌍 Filter by Region:",
                                    choices=initial_data['regions'], selected="All"
                                )
                            ),
                            ui.column(
                                5,
                                ui.input_radio_buttons(
                                    "chemical_category",
                                    "🧪 Select Chemical Subspace:",
                                    choices={
                                        "All": "All Substances (Total CS)",
                                        "Organic": "Organic",
                                        "Organometallic": "Organometallic",
                                        "Rare-Earths": "Rare-Earths (REE)"
                                    },
                                    selected="All",
                                    inline=True
                                )
                            ),
                            ui.column(
                                4,
                                ui.input_slider(
                                    "years", "📅 Year Range:",
                                    min=initial_data['min_year'], max=initial_data['max_year'],
                                    value=[initial_data['min_year'], initial_data['max_year']],
                                    step=1, sep="",
                                    animate=ui.AnimationOptions(
                                        interval=700,
                                        loop=True,
                                        play_button=ui.tags.span("▶ Play Period", class_="btn btn-sm btn-success fw-bold"),
                                        pause_button=ui.tags.span("⏸ Pause", class_="btn btn-sm btn-warning fw-bold")
                                    )
                                )
                            )
                        ),
                        class_="bg-light border rounded p-3 mb-3 shadow-sm"
                    ),

                    ui.output_ui("dynamic_explorer_kpis"),

                    # 5. FIFTH ELEMENT: IN-DEPTH EXPLORATION (COLLABORATIONS & SUMMARY TABLE)
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.div(
                                    ui.tags.span("🔬 In-Depth Exploration: Joint Collaborations & Summary Statistics", style="font-weight: 700; color: #0f172a;"),
                                    ui.tags.small(
                                        "Explore bilateral & multilateral partnerships among your selected countries or export the summary table.",
                                        class_="text-muted ms-2"
                                    )
                                ),
                                ui.download_button(
                                    "download_summary_csv",
                                    "📥 Download Summary CSV",
                                    class_="btn-sm btn-outline-secondary"
                                ),
                                class_="d-flex align-items-center justify-content-between w-100"
                            )
                        ),
                        ui.navset_card_pill(
                            ui.nav_panel(
                                "🤝 Joint Collaborations Between Selected Countries",
                                ui.div(
                                    ui.p(
                                        "Shows all bilateral, trilateral, 4-country, and 5-country+ research networks that include the countries currently selected above. "
                                        "Tip: Click a Focus button below to inspect specific bilateral/trilateral partnerships.",
                                        class_="text-muted small mb-2"
                                    ),
                                    ui.div(
                                        ui.input_action_button(
                                            "preset_cn_us_collab",
                                            "🥼 Focus: China + US (CN–US)",
                                            class_="btn-sm btn-outline-primary me-2"
                                        ),
                                        ui.input_action_button(
                                            "preset_eu_us_collab",
                                            "🇪🇺 Focus: Germany + UK + US",
                                            class_="btn-sm btn-outline-primary me-2"
                                        ),
                                        ui.input_action_button(
                                            "preset_asia_collab",
                                            "🌏 Focus: China + Japan + India",
                                            class_="btn-sm btn-outline-primary"
                                        ),
                                        class_="mb-2"
                                    )
                                ),
                                ui.output_ui("collab_plot")
                            ),
                            ui.nav_panel(
                                "📋 Summary Statistics Table",
                                ui.div(
                                    ui.input_radio_buttons(
                                        "summary_table_mode",
                                        "Table View:",
                                        choices={
                                            "compare_individuals": "Individual Selected Countries",
                                            "find_collaborations": "Joint Collaborations Among Selected"
                                        },
                                        selected="compare_individuals",
                                        inline=True
                                    ),
                                    class_="mb-2"
                                ),
                                ui.output_data_frame("summary_table")
                            )
                        ),
                        class_="border rounded mb-3 shadow-sm"
                    )
                ),

                # TAB 2: Key Article Figures (with Executive Side-by-Side View + Animated Bubble Explorer)
                ui.nav_panel(
                    "📊 Key Article Figures & Animations",
                    ui.div(
                        ui.input_radio_buttons(
                            "article_view_layout",
                            "📐 Layout Mode:",
                            choices={
                                "grid": "📊 Executive Side-by-Side Grid (All Figures at a Glance)",
                                "tabs": "📑 Individual Figure Tabs",
                                "animated_bubbles": "🎬 Animated Bubble & Flag Ranking Explorer (1996–2022)"
                            },
                            selected="grid",
                            inline=True
                        ),
                        style="display: flex; justify-content: center; padding: 10px; background: #f8f9fa; border-radius: 6px; margin-bottom: 15px;"
                    ),
                    ui.output_ui("article_figures_container")
                ),

                # TAB 3: Top 10 Collaborations & Contributors
                ui.nav_panel(
                    "🤝 Top 10 Global Trends",
                    ui.panel_well(
                        ui.h4("Explore Top 10 Collaborations and Countries in the Chemical Space", style="text-align: center;"),
                        ui.row(
                            ui.column(
                                6,
                                ui.input_select(
                                    "top_collabs_chem_filter",
                                    "🧪 Chemical Subspace:",
                                    choices=initial_data['chemical_categories'],
                                    selected="All"
                                )
                            ),
                            ui.column(
                                6,
                                ui.input_radio_buttons(
                                    "top_data_type_filter",
                                    "🏆 Show Top 10:",
                                    choices={"collabs": "International Collaborations", "individuals": "Individual Countries"},
                                    selected="collabs",
                                    inline=True
                                )
                            )
                        ),
                        ui.output_ui("article_top_collabs_plot"),
                        ui.p(
                            "Legend is sorted by average contribution (shown in parentheses) between 1996 and 2022.",
                            style="font-size: 0.9em; color: #475569; text-align: center; margin-top: 8px;"
                        ),
                        class_="border rounded p-3 bg-light"
                    )
                )
            )
        ),

        ui.nav_panel(
            "📖 Original Article",
            ui.tags.iframe(
                src="chemrxiv-2025-d2zc8.pdf",
                style="width: 100%; height: 82vh; border: none;"
            )
        ),
ui.nav_panel(
            "📓 Interactive Tutorial",
            ui.div(
                # Barra superior del Tutorial: Selector de Idioma + Botones de Colab y Descarga
                ui.panel_well(
                    ui.div(
                        ui.div(
                            ui.h4("🧪 Chemical Space Explorer — Interactive Notebook Walkthrough", class_="mb-1 fw-bold"),
                            ui.p(
                                "Modify the parameters inside the live cells below to test how the app loads, filters, and plots data in real time.",
                                class_="text-muted small mb-0"
                            )
                        ),
                        ui.div(
                            ui.input_radio_buttons(
                                "tut_lang",
                                None,
                                choices={"es": "🇲🇽 Español", "en": "🇬🇧 English"},
                                selected="es",
                                inline=True
                            ),
                            ui.tags.a(
                                "🚀 Open in Google Colab",
                                href="https://colab.research.google.com/github/santi-rios/CS-Explorer-App/blob/main/www/Tutorial_Marisol_CS_Explorer.ipynb",
                                target="_blank",
                                class_="btn btn-primary btn-sm ms-2 me-1 text-white fw-bold"
                            ),
                            ui.tags.a(
                                "📥 Download .ipynb",
                                href="Tutorial_Marisol_CS_Explorer.ipynb",
                                download="Tutorial_Marisol_CS_Explorer.ipynb",
                                class_="btn btn-outline-secondary btn-sm"
                            ),
                            class_="d-flex flex-wrap align-items-center"
                        ),
                        class_="d-flex flex-wrap justify-content-between align-items-center"
                    ),
                    class_="bg-light border rounded p-3 mb-3 shadow-sm"
                ),

                # Contenido dinámico de los 7 módulos (cambia entre Español e Inglés y ejecuta celdas en vivo)
                ui.output_ui("interactive_notebook_body"),
                style="max-width: 1200px; margin: 0 auto;"
            )
        ),

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
                            class_="btn btn-primary btn-sm me-2 mb-2 text-white fw-bold"
                        ),
                        ui.tags.a(
                            "📥 Download Jupyter Notebook (.ipynb)",
                            href="Tutorial_Marisol_CS_Explorer.ipynb",
                            download="Tutorial_Marisol_CS_Explorer.ipynb",
                            class_="btn btn-outline-secondary btn-sm mb-2"
                        ),
                        class_="d-flex flex-wrap align-items-center mt-2"
                    )
                ),
                ui.card(
                    ui.card_header("🔗 Project, Data & Author Resources"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.b("Published Article: "),
                            ui.tags.a(
                                "Bermúdez-Montaña et al. (2025) — Quantitative Science Studies (MIT Press)",
                                href="https://doi.org/10.1162/QSS.a.471",
                                target="_blank"
                            )
                        ),
                        ui.tags.li(
                            ui.tags.b("Raw Data & Article Scripts: "),
                            ui.tags.a(
                                "China's Great Leap Forward in the Chemical Space (GitHub)",
                                href="https://github.com/mbermudezmoTec/China-s-great-leap-forward-in-the-chemical-space",
                                target="_blank"
                            )
                        ),
                        ui.tags.li(
                            ui.tags.b("Web Application Repository: "),
                            ui.tags.a(
                                "Interactive CS-Explorer-App Source Code (GitHub)",
                                href="https://github.com/santi-rios/CS-Explorer-App",
                                target="_blank"
                            )
                        ),
                        ui.tags.li(
                            ui.tags.b("Developer Portfolio: "),
                            ui.tags.a(
                                "Site Built by Santiago G-Rios",
                                href="https://santi-rios.github.io/",
                                target="_blank"
                            )
                        ),
                        class_="mb-0"
                    )
                ),
                width=1/2
            )
        ),
        header=ui.tags.head(
            ui.tags.script(plotly_offline.get_plotlyjs()),
            ui.tags.script("""
                Shiny.addCustomMessageHandler('trigger_plotly_resize', function(msg) {
                    setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 260);
                });
                Shiny.addCustomMessageHandler('reset_saved_map_view', function(msg) {
                    window._cs_saved_map_view = null;
                });

                // Condense selected country pills in #country_picker to show just "🇨🇳 CHN"
                // while keeping the full "🇨🇳 CHN — China" text searchable in the dropdown!
                document.addEventListener('DOMContentLoaded', function() {
                    var checkSelectize = setInterval(function() {
                        var el = document.getElementById('country_picker');
                        if (el && el.selectize) {
                            clearInterval(checkSelectize);
                            var sel = el.selectize;
                            sel.settings.render.item = function(item, escape) {
                                var shortLabel = String(item.text).split(' — ')[0];
                                return '<div class="item" title="' + escape(item.text) + '">' + escape(shortLabel) + '</div>';
                            };
                            // Refresh already-rendered initial items
                            var currentVals = sel.getValue();
                            sel.setValue(currentVals, true);
                        }
                    }, 80);
                });
            """),
            ui.tags.style("""
                /* 1. HIGH-CONTRAST TOP NAVBAR (White text on dark navy background) */
                .navbar {
                    background-color: #1e293b !important;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                }
                .navbar .navbar-brand {
                    color: #ffffff !important;
                    font-weight: 700 !important;
                }
                .navbar .nav-link {
                    color: rgba(255, 255, 255, 0.85) !important;
                    font-weight: 600 !important;
                }
                .navbar .nav-link:hover {
                    color: #ffffff !important;
                    background-color: rgba(255, 255, 255, 0.1) !important;
                    border-radius: 5px;
                }
                .navbar .nav-link.active {
                    color: #ffffff !important;
                    background-color: rgba(56, 189, 248, 0.22) !important;
                    border-bottom: 3px solid #38bdf8 !important;
                    border-radius: 5px 5px 0 0;
                }

                /* 2. High-contrast links inside the page body (without affecting navbar or buttons) */
                a.article-doi-link,
                .card-body a:not(.btn):not(.nav-link),
                footer a {
                    color: #0b4f6c !important;
                    font-weight: 600;
                    text-decoration-thickness: 1.5px;
                }
                a.article-doi-link:hover,
                .card-body a:not(.btn):not(.nav-link):hover,
                footer a:hover {
                    color: #032b3d !important;
                }

                /* 3. High-contrast Tab & Pill Headers */
                .nav-tabs .nav-link,
                .nav-pills .nav-link {
                    color: #334155 !important;
                    font-weight: 600 !important;
                }
                .nav-tabs .nav-link:hover,
                .nav-pills .nav-link:hover {
                    color: #0f172a !important;
                    background-color: #e2e8f0 !important;
                }
                .nav-tabs .nav-link.active {
                    color: #0f172a !important;
                    font-weight: 700 !important;
                    border-bottom: 3px solid #0284c7 !important;
                    background-color: #ffffff !important;
                }
                .nav-pills .nav-link.active {
                    color: #ffffff !important;
                    background-color: #0284c7 !important;
                }

                /* 4. Compact, Clean ISO3 Pills inside the Country Search Box */
                #country_picker + .selectize-control.multi .selectize-input {
                    padding: 4px 8px !important;
                    min-height: 38px !important;
                    background-color: #ffffff !important;
                    border: 1px solid #94a3b8 !important;
                }
                #country_picker + .selectize-control.multi .selectize-input > div.item {
                    background: #e0f2fe !important;
                    color: #0f172a !important;
                    border: 1px solid #7dd3fc !important;
                    border-radius: 5px !important;
                    padding: 1px 6px !important;
                    margin: 2px 3px 2px 0 !important;
                    font-size: 0.80rem !important;
                    font-weight: 700 !important;
                }
                #country_picker + .selectize-control.multi .selectize-input > div.item .remove {
                    border-left: 1px solid #7dd3fc !important;
                    color: #0369a1 !important;
                    padding-left: 4px !important;
                }

                /* 5. High-contrast Buttons & Slider Animation Play Button */
                .slider-animate-button {
                    opacity: 1 !important;
                    text-decoration: none !important;
                }
                .btn-outline-primary {
                    color: #0369a1 !important;
                    border-color: #0284c7 !important;
                    background-color: #ffffff !important;
                    font-weight: 600 !important;
                }
                .btn-outline-primary:hover {
                    color: #ffffff !important;
                    background-color: #0284c7 !important;
                }
                .btn-outline-secondary {
                    color: #1e293b !important;
                    border-color: #64748b !important;
                    background-color: #ffffff !important;
                    font-weight: 600 !important;
                }
                .btn-outline-secondary:hover {
                    color: #ffffff !important;
                    background-color: #334155 !important;
                }
                .btn-outline-success {
                    color: #065f46 !important;
                    border-color: #059669 !important;
                    background-color: #ffffff !important;
                    font-weight: 600 !important;
                }
                .btn-outline-success:hover {
                    color: #ffffff !important;
                    background-color: #059669 !important;
                }
                .btn-outline-danger {
                    color: #991b1b !important;
                    border-color: #dc2626 !important;
                    background-color: #ffffff !important;
                    font-weight: 600 !important;
                }
                .btn-outline-danger:hover {
                    color: #ffffff !important;
                    background-color: #dc2626 !important;
                }
                .text-muted, small.text-muted {
                    color: #475569 !important;
                }
                
                /* 6. Estilos de Celdas Interactivas tipo Jupyter Notebook */
                .jp-cell {
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    background: #ffffff;
                    margin-bottom: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    overflow: hidden;
                }
                .jp-cell-header {
                    background: #f1f5f9;
                    padding: 8px 14px;
                    border-bottom: 1px solid #e2e8f0;
                    font-weight: 700;
                    color: #0f172a;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .jp-input-area {
                    display: flex;
                    background: #f8fafc;
                    color: #f8fafc;
                    padding: 12px;
                    font-family: 'Fira Code', 'Consolas', monospace;
                    font-size: 0.85rem;
                }
                .jp-prompt-in {
                    color: #0f172a;
                    font-weight: 700;
                    min-width: 65px;
                    user-select: none;
                }
                .jp-prompt-out {
                    color: #0f172a;
                    font-weight: 700;
                    min-width: 65px;
                    user-select: none;
                }
                .jp-output-area {
                    display: flex;
                    padding: 14px;
                    background: #ffffff;
                    border-top: 1px dashed #cbd5e1;
                    overflow-x: auto;
                }
                
            """)
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
                style="text-align: center; padding: 10px; font-size: 0.9em; color: #475569;"
            )
        )
    )

    def server(input, output, session):
        selected_countries = reactive.Value(default_top10)

        @reactive.Calc
        def country_list():
            return load_country_list()

        @reactive.Calc
        def individual_filtered_data():
            selected_tuple = tuple(sorted(selected_countries.get())) if selected_countries.get() else ()
            return cached_get_display_data(
                selected_isos_tuple=selected_tuple,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode="compare_individuals",
                region_filter=input.region_filter()
            )

        @reactive.Calc
        def collab_filtered_data():
            selected_tuple = tuple(sorted(selected_countries.get())) if selected_countries.get() else ()
            return cached_get_display_data(
                selected_isos_tuple=selected_tuple,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode="find_collaborations",
                region_filter=input.region_filter()
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
                region_filter=reg
            )

        def _compute_top_isos_for_period(y_start: int, y_end: int, top_n: int = 10) -> list[str]:
            """Helper to get Top N ISO codes for the active region and chemical subspace."""
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

        # --- Quick Discovery Preset Buttons ---
        @reactive.Effect
        @reactive.event(input.preset_top10_current)
        def _apply_top10_current():
            y0, y1 = input.years()
            top_isos = _compute_top_isos_for_period(y0, y1, top_n=10)
            selected_countries.set(top_isos)
            ui.update_selectize("country_picker", selected=top_isos)

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

        # Collaboration Quick-Focus Buttons
        @reactive.Effect
        @reactive.event(input.preset_cn_us_collab)
        def _focus_cn_us():
            isos = ["CN", "US"]
            selected_countries.set(isos)
            ui.update_selectize("country_picker", selected=isos)

        @reactive.Effect
        @reactive.event(input.preset_eu_us_collab)
        def _focus_eu_us():
            isos = ["DE", "GB", "US"]
            selected_countries.set(isos)
            ui.update_selectize("country_picker", selected=isos)

        @reactive.Effect
        @reactive.event(input.preset_asia_collab)
        def _focus_asia():
            isos = ["CN", "JP", "IN"]
            selected_countries.set(isos)
            ui.update_selectize("country_picker", selected=isos)

        # Sync dropdown -> reactive selected_countries
        @reactive.Effect
        @reactive.event(input.country_picker, ignore_none=False, ignore_init=True)
        def _sync_picker_to_selected():
            picker_isos = list(input.country_picker() or [])
            if picker_isos != selected_countries.get():
                selected_countries.set(picker_isos)

        # Header button to open/close the right sidebar drawer
        @reactive.Effect
        @reactive.event(input.toggle_choropleth_drawer)
        def _toggle_right_drawer():
            is_currently_open = bool(input.choropleth_right_sidebar())
            ui.update_sidebar("choropleth_right_sidebar", show=not is_currently_open)

        @reactive.Effect
        @reactive.event(input.choropleth_right_sidebar)
        async def _resize_plotly_on_sidebar_toggle():
            await session.send_custom_message("trigger_plotly_resize", {})

        # Sync Folium map click -> reactive selected_countries AND dropdown
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

        # Sync Reset button -> clear both
        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])
            ui.update_selectize("country_picker", selected=[])

        # Update dropdown choices and auto-select Top 10 of that region when Region Filter changes
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

        @output
        @render.ui
        def dynamic_explorer_kpis():
            data = individual_filtered_data()
            y_start, y_end = input.years()
            chem = input.chemical_category()

            if data.empty:
                return ui.div()

            by_country = data.groupby(['iso2c', 'country'])['total_percentage'].agg(['mean', 'max']).reset_index()
            by_country = by_country.sort_values('mean', ascending=False)
            top_row = by_country.iloc[0]
            leader_flag = iso_to_flag_emoji(top_row['iso2c'])
            leader_title = f"Top Selected ({y_start}–{y_end}): {top_row['country']}"
            leader_val = f"{top_row['mean']:.2f}% avg (Peak: {top_row['max']:.1f}%)"

            latest_yr = int(data['year'].max())
            latest_sum = data[data['year'] == latest_yr]['total_percentage'].sum()
            second_title = f"Combined Share in {latest_yr} ({len(by_country)} Countries)"
            second_val = f"{latest_sum:.2f}%"

            collab_df = collab_filtered_data()
            n_collabs = collab_df['plot_group'].nunique() if not collab_df.empty else 0

            return ui.layout_column_wrap(
                ui.div(
                    ui.div(f"{leader_flag} {leader_title}", style="font-size: 0.80rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(leader_val, style="font-size: 1.30rem; font-weight: 700;"),
                    style="background: #0d9488; color: white; padding: 10px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                ui.div(
                    ui.div(f"📈 {second_title}", style="font-size: 0.80rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(second_val, style="font-size: 1.30rem; font-weight: 700;"),
                    style="background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                ui.div(
                    ui.div(f"🤝 Joint Networks ({chem}, {y_start}–{y_end})", style="font-size: 0.80rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(f"{n_collabs} Shared Collaborations", style="font-size: 1.30rem; font-weight: 700;"),
                    style="background: #4f46e5; color: white; padding: 10px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                width=1/3,
                style="margin-bottom: 14px;"
            )

        @output
        @render.text
        def selection_info():
            count = len(selected_countries.get())
            current_region = input.region_filter()
            all_countries = country_list()
            available_count = len(
                all_countries[all_countries['region'] == current_region]
                if current_region != "All" else all_countries
            )
            return f"Showing {count} of {available_count} countries ({current_region})"

        @output
        @render.ui
        def map_output():
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
                map_style=input.map_visual_style()
            )
            return ui.HTML(m._repr_html_())

        @output
        @render.ui
        def main_plot():
            selected = selected_countries.get()
            data = individual_filtered_data()
            if data.empty:
                return figure_to_html(create_empty_plot(
                    "Select one or more countries from the search bar, preset buttons, or map to view trends."
                ))

            return figure_to_html(create_trends_plot(
                data=data,
                selected_countries=selected,
                mode="compare_individuals",
                show_flags=bool(input.show_flags_on_trends())
            ))

        @output
        @render.ui
        def collab_plot():
            selected = selected_countries.get()
            if len(selected) < 2:
                return figure_to_html(create_empty_plot(
                    "🤝 Select at least 2 countries (or click one of the Focus buttons above) to explore joint collaborations."
                ))

            data = collab_filtered_data()
            if data.empty:
                return figure_to_html(create_empty_plot(
                    f"No joint collaborations found linking all {len(selected)} selected countries ({', '.join(selected)}).\n"
                    "Tip: Click one of the Focus buttons above (e.g., China + US) to see all bilateral & multilateral networks linking them."
                ))

            return figure_to_html(create_trends_plot(
                data=data,
                selected_countries=selected,
                mode="find_collaborations",
                show_flags=False
            ))

        @output
        @render.ui
        def contribution_map():
            try:
                choropleth_data = region_all_countries_data()
                if choropleth_data.empty:
                    return figure_to_html(create_empty_plot("No data for global map with current selections"))

                mode = input.snapshot_mode()
                y_start, y_end = input.years()
                chem_cat = input.chemical_category()

                if mode == "static":
                    return figure_to_html(
                        create_contribution_map_plot(
                            choropleth_data,
                            fill_label="Avg Share",
                            main_title=f"Average CS Contribution ({y_start}–{y_end}) — {chem_cat}"
                        )
                    )
                return figure_to_html(create_contribution_choropleth(choropleth_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error creating map: {str(e)}"))

        @output
        @render.data_frame
        def summary_table():
            t_mode = input.summary_table_mode()
            data = individual_filtered_data() if t_mode == "compare_individuals" else collab_filtered_data()
            if data.empty:
                msg = (
                    "Select at least 2 countries (e.g. CN + US) to list joint collaborations."
                    if t_mode == "find_collaborations"
                    else "No data available for current selections."
                )
                return pd.DataFrame({"Message": [msg]})
            return create_summary_dataframe(data, t_mode)

        @render.download(filename=lambda: f"CS_Explorer_Summary_{input.chemical_category()}_{input.years()[0]}_{input.years()[1]}.csv")
        def download_summary_csv():
            t_mode = input.summary_table_mode()
            data = individual_filtered_data() if t_mode == "compare_individuals" else collab_filtered_data()
            df_out = create_summary_dataframe(data, t_mode)
            yield df_out.to_csv(index=False)

        @output
        @render.ui
        def article_figures_container():
            layout_mode = input.article_view_layout()

            if layout_mode == "animated_bubbles":
                return ui.card(
                    ui.card_header("🎬 Animated Global Ranking & Bubble Share (1996–2022)"),
                    ui.row(
                        ui.column(
                            4,
                            ui.input_select(
                                "anim_bubble_chem",
                                "🧪 Select Chemical Subspace:",
                                choices=initial_data['chemical_categories'],
                                selected="All"
                            )
                        ),
                        ui.column(
                            4,
                            ui.input_slider(
                                "anim_top_n",
                                "🌍 Number of Top Countries to Track:",
                                min=10, max=35, value=20, step=5
                            )
                        )
                    ),
                    ui.output_ui("animated_bubble_plot"),
                    ui.p(
                        "Press ▶ Play below the chart to watch China's surge from #6 in 1996 to #1 from 2013 onward.",
                        style="text-align: center; color: #475569; font-size: 0.9em;"
                    )
                )

            if layout_mode == "grid":
                return ui.div(
                    ui.layout_column_wrap(
                        ui.card(
                            ui.card_header("🏆 Main Countries in the CS (Fig. 1a)"),
                            ui.output_ui("country_cs_plot")
                        ),
                        ui.card(
                            ui.card_header("🥼 China–US Domestic vs. Collaborative Share (Fig. 1c)"),
                            ui.output_ui("china_us_plot")
                        ),
                        width=1/2
                    ),
                    ui.layout_column_wrap(
                        ui.card(
                            ui.card_header("🧪 Chemical Space & Subspace Expansion (Fig. 2a — Log Scale)"),
                            ui.output_ui("article_cs_expansion_plot")
                        ),
                        ui.card(
                            ui.card_header("👥 R&D Researchers Headcount (Fig. 1e)"),
                            ui.output_ui("article_researchers_plot")
                        ),
                        width=1/2
                    ),
                    ui.card(
                        ui.card_header("💰 Annual GDP per Capita Growth Rate (Fig. 1d)"),
                        ui.output_ui("article_gdp_plot")
                    )
                )

            return ui.navset_card_tab(
                ui.nav_panel("🏆 Main Countries", ui.output_ui("country_cs_plot")),
                ui.nav_panel("🥼 China–US", ui.output_ui("china_us_plot")),
                ui.nav_panel("🧪 CS Expansion (Log)", ui.output_ui("article_cs_expansion_plot")),
                ui.nav_panel("👥 Researchers", ui.output_ui("article_researchers_plot")),
                ui.nav_panel("💰 GDP Growth", ui.output_ui("article_gdp_plot"))
            )

        @output
        @render.ui
        def animated_bubble_plot():
            df = get_raw_dataframe()
            chem = input.anim_bubble_chem() if hasattr(input, "anim_bubble_chem") else "All"
            top_n = int(input.anim_top_n()) if hasattr(input, "anim_top_n") else 20
            return figure_to_html(create_animated_bubble_figure(df, chemical_category=chem, top_n=top_n))

        @output
        @render.ui
        def country_cs_plot():
            article_data = load_article_data()
            df_filtered = article_data[article_data['source'] == "Country participation in the CS"]
            if df_filtered.empty:
                return figure_to_html(create_empty_plot("No country participation data available"))
            return figure_to_html(create_article_plot(df_filtered, "Country participation in the growth of the CS"))

        @output
        @render.ui
        def china_us_plot():
            article_data = load_article_data()
            cs_data = article_data[article_data['source'] == "China-US in the CS"]
            if cs_data.empty:
                return figure_to_html(create_empty_plot("No 'China-US in the CS' data found"))
            return figure_to_html(create_china_us_dual_axis_plot(cs_data))

        @output
        @render.ui
        def article_gdp_plot():
            article_data = load_article_data()
            gdp_data = article_data[article_data['source'] == "Annual growth rate of the GDP"]
            if gdp_data.empty:
                return figure_to_html(create_empty_plot("No GDP data available"))
            return figure_to_html(create_gdp_plot(gdp_data))

        @output
        @render.ui
        def article_researchers_plot():
            article_data = load_article_data()
            researchers_data = article_data[article_data['source'] == "Number of Researchers"]
            if researchers_data.empty:
                return figure_to_html(create_empty_plot("No researchers data available"))
            return figure_to_html(create_researchers_plot(researchers_data))

        @output
        @render.ui
        def article_cs_expansion_plot():
            article_data = load_article_data()
            cs_data = article_data[article_data['source'] == "Expansion of the CS"]
            if cs_data.empty:
                return figure_to_html(create_empty_plot("No CS expansion data available"))
            return figure_to_html(create_cs_expansion_plot(cs_data))

        @output
        @render.ui
        def article_top_collabs_plot():
            df = get_raw_dataframe()
            is_collab = input.top_data_type_filter() == "collabs"
            chem_filter = input.top_collabs_chem_filter()
            filtered = df[(df['is_collab'] == is_collab) & (df['chemical'] == chem_filter)]
            if filtered.empty:
                return figure_to_html(create_empty_plot("No data available"))

            top_entities = (
                filtered.groupby('country')['percentage']
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            return figure_to_html(create_top_trends_plot(
                filtered[filtered['country'].isin(top_entities.index)],
                f"Top 10 {'Collaborations' if is_collab else 'Countries'} — {chem_filter}"
            ))
        # --- CELDAS EN VIVO DEL TUTORIAL INTERACTIVO (MÓDULOS 1 A 7) ---
        @output
        @render.ui
        def interactive_notebook_body():
            es = (input.tut_lang() == "es")
            return ui.div(
                # MÓDULO 1 & 2
                ui.div(
                    ui.div(
                        "🏗️ Módulo 1 y 2: Arquitectura General y Preparación del Entorno (Git)"
                        if es else "🏗️ Modules 1 & 2: General Architecture & Environment Setup (Git)",
                        class_="jp-cell-header"
                    ),
                    ui.div(
                        ui.p(
                            "La aplicación se divide en 4 capas: (1) Base de datos columnar Apache Parquet (data/data.parquet) con 189 países y colaboraciones, "
                            "(2) Motor de filtrado y graficación en utils/functions.py, (3) Interfaz reactiva Shiny en app.py, y (4) Contenedores Docker + Cloudflare Tunnel."
                            if es else
                            "The app is structured into 4 layers: (1) Apache Parquet database (data/data.parquet) covering 189 countries and collaborations, "
                            "(2) Filtering and plotting engine in utils/functions.py, (3) Reactive Shiny UI in app.py, and (4) Docker + Cloudflare Tunnel stack.",
                            class_="p-3 mb-0"
                        ),
                        ui.div(
                            ui.div("In [1]:", class_="jp-prompt-in"),
                            ui.pre(
                                "# Clonar el repositorio e instalar dependencias / Clone repo & install dependencies\n"
                                "git clone https://github.com/santi-rios/CS-Explorer-App.git\n"
                                "cd CS-Explorer-App\n"
                                "pip install -r requirements.txt\n"
                                "shiny run app.py --port 8888",
                                class_="mb-0 text-info"
                            ),
                            class_="jp-input-area"
                        )
                    ),
                    class_="jp-cell"
                ),

                # MÓDULO 3: CELDA EN VIVO PARA EXPLORAR DATA.PARQUET
                ui.div(
                    ui.div(
                        "🗄️ Módulo 3: Cómo se Manejan los Datos (Prueba la celda en vivo)"
                        if es else "🗄️ Module 3: Data Management & Caching (Try live cell)",
                        class_="jp-cell-header"
                    ),
                    ui.div(
                        ui.row(
                            ui.column(
                                6,
                                ui.input_radio_buttons(
                                    "tut_m3_subset",
                                    "Selecciona qué subconjunto inspeccionar en data.parquet:"
                                    if es else "Select subset of data.parquet to inspect:",
                                    choices={
                                        "solo": "Países individuales (is_collab == False)" if es else "Individual Countries (is_collab == False)",
                                        "collab": "Colaboraciones internacionales (is_collab == True)" if es else "Collaborations (is_collab == True)"
                                    },
                                    selected="solo",
                                    inline=True
                                )
                            ),
                            ui.column(
                                6,
                                ui.input_slider(
                                    "tut_m3_rows",
                                    "Número de filas a mostrar (.head(n)):" if es else "Number of rows (.head(n)):",
                                    min=3, max=15, value=5, step=1
                                )
                            ),
                            class_="p-3 bg-light m-0 border-bottom"
                        ),
                        ui.output_ui("tut_m3_code_and_output")
                    ),
                    class_="jp-cell"
                ),

                # MÓDULO 4 y 5: CELDA EN VIVO DE FILTRADO Y GRÁFICAS INTERACTIVAS
                ui.div(
                    ui.div(
                        "📈 Módulos 4 y 5: Filtrado (get_display_data) y Gráficas Interactivas (create_trends_plot)"
                        if es else "📈 Modules 4 & 5: Filtering Pipeline & Interactive Plotly Charts",
                        class_="jp-cell-header"
                    ),
                    ui.div(
                        ui.row(
                            ui.column(
                                4,
                                ui.input_selectize(
                                    "tut_m4_isos",
                                    "1. Países (selected_isos):" if es else "1. Countries (selected_isos):",
                                    choices=initial_data['country_choices'],
                                    selected=["CN", "US", "DE", "JP"],
                                    multiple=True
                                )
                            ),
                            ui.column(
                                4,
                                ui.input_select(
                                    "tut_m4_chem",
                                    "2. Subespacio (chemical_category):" if es else "2. Subspace (chemical_category):",
                                    choices=initial_data['chemical_categories'],
                                    selected="All"
                                )
                            ),
                            ui.column(
                                4,
                                ui.input_radio_buttons(
                                    "tut_m4_mode",
                                    "3. Modo (display_mode):" if es else "3. Mode (display_mode):",
                                    choices={
                                        "compare_individuals": "compare_individuals",
                                        "find_collaborations": "find_collaborations"
                                    },
                                    selected="compare_individuals"
                                )
                            ),
                            class_="p-3 bg-light m-0 border-bottom"
                        ),
                        ui.output_ui("tut_m4_m5_code_and_output")
                    ),
                    class_="jp-cell"
                ),

                # MÓDULO 6 & 7: REACTIVIDAD Y ADMINISTRACIÓN DEL SERVIDOR
                ui.div(
                    ui.div(
                        "⚡ Módulos 6 y 7: Reactividad en Shiny y Administración del Servidor"
                        if es else "⚡ Modules 6 & 7: Shiny Reactivity & Server Administration",
                        class_="jp-cell-header"
                    ),
                    ui.div(
                        ui.p(
                            "En Shiny, cada control visual (input) dispara automáticamente los bloques @reactive.Calc y redibuja las salidas @output @render.ui. "
                            "Para actualizar la aplicación en producción tras hacer git push, ejecuta en el servidor:"
                            if es else
                            "In Shiny, every UI input automatically triggers @reactive.Calc functions and re-renders @output @render.ui blocks. "
                            "To update the production server after git push, run on the server:",
                            class_="p-3 mb-0"
                        ),
                        ui.div(
                            ui.div("In [4]:", class_="jp-prompt-in"),
                            ui.pre(
                                "# Atajos de administración en el servidor Linux (definidos en ~/.bashrc):\n"
                                "app-update      # Hace git pull y recrea el contenedor shiny-chemical en segundos\n"
                                "app-rebuild     # Reconstruye la imagen Docker desde cero si cambió requirements.txt\n"
                                "tunnel-restart  # Reinicia el conector de Cloudflare Tunnel",
                                class_="mb-0 text-warning"
                            ),
                            class_="jp-input-area"
                        )
                    ),
                    class_="jp-cell"
                )
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
                ui.div(
                    ui.div("In [2]:", class_="jp-prompt-in"),
                    ui.pre(code_str, class_="mb-0"),
                    class_="jp-input-area"
                ),
                ui.div(
                    ui.div("Out[2]:", class_="jp-prompt-out"),
                    ui.div(
                        ui.HTML(sub_df.to_html(index=False, classes="table table-sm table-striped table-bordered mb-0")),
                        class_="w-100"
                    ),
                    class_="jp-output-area"
                )
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
                region_filter="All"
            )
            fig = create_trends_plot(
                data=df_filtered,
                selected_countries=isos,
                mode=mode,
                show_flags=(mode == "compare_individuals")
            )
            fig.update_layout(height=420)

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
                ui.div(
                    ui.div("In [3]:", class_="jp-prompt-in"),
                    ui.pre(code_str, class_="mb-0"),
                    class_="jp-input-area"
                ),
                ui.div(
                    ui.div("Out[3]:", class_="jp-prompt-out"),
                    ui.div(figure_to_html(fig), class_="w-100"),
                    class_="jp-output-area"
                )
            )

    return App(app_ui, server, static_assets=www_dir)


app = create_app()

if __name__ == "__main__":
    app.run(port=8000)