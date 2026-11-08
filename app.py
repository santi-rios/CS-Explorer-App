import pandas as pd
import plotly.offline as plotly_offline
from shiny import App, ui, render, reactive
from pathlib import Path
from functools import lru_cache

from utils.functions import (
    get_display_data,
    create_enhanced_folium_map,
    create_trends_plot,
    create_contribution_choropleth,
    create_contribution_map_plot,
    create_cartogram_plot,  # <-- ADD THIS IMPORT
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
import shinyswatch


www_dir = Path(__file__).resolve().parent / "www"


@lru_cache(maxsize=1)
def load_country_list() -> pd.DataFrame:
    """Cached country metadata table."""
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
        country_choices = {
            row['iso2c']: f"{iso_to_flag_emoji(row['iso2c'])} {row['country']} ({row['iso2c']})"
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
            ui.div(
                ui.h3(
                    "China's Rise in the Chemical Space & the Decline of US Influence",
                    style=(
                        "text-align: center; "
                        "font-weight: 700; "
                        "margin-top: 15px; "
                        "margin-bottom: 8px;"
                    )
                ),
                ui.p(
                    "Interactive geopolitical explorer of 15.7 million new chemical "
                    "substances across 189 countries and territories (1996–2022) — ",
                    ui.tags.a(
                        "Bermúdez-Montaña et al. (2025), "
                        "Quantitative Science Studies",
                        href="https://doi.org/10.1162/QSS.a.471",
                        target="_blank",
                        style="font-weight: 600;"
                    ),
                    style=(
                        "text-align: center; "
                        "color: #555; "
                        "margin-bottom: 20px;"
                    )
                ),
            ),
            ui.layout_column_wrap(
# Full-width Trends Card with Collapsible Right-Hand Choropleth Drawer
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.div(
                            ui.tags.span("📈 Contribution Trends", style="font-weight: 600;"),
                            ui.tags.small(
                                "Select countries and filters to update the chart.",
                                class_="text-muted ms-2"
                            )
                        ),
                        # Button in the header that also opens/closes the right sidebar!
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
                                    "Average or animated contribution by country",
                                    class_="d-block text-muted mb-2"
                                )
                            ),
                            ui.input_radio_buttons(
                                "snapshot_mode",
                                "🗺️ Map & Cartogram Mode:",
                                choices={
                                    "static": "🗺️ Standard Choropleth (Avg)",
                                    "animated": "🎬 Animated Choropleth (Yearly)",
                                    "cartogram_static": "🧩 Deformed Cartogram (Avg)",
                                    "cartogram_anim": "🚀 Animated Cartogram (1996–2022)"
                                },
                                selected="cartogram_static",
                                inline=True
                            ),
                            class_="snapshot-controls mb-2 pb-2 border-bottom"
                        ),
                        ui.output_ui("contribution_map"),
                        id="choropleth_right_sidebar",
                        position="right",
                        open="closed",   # Starts collapsed so main_plot spans 100% of the UI!
                        width="48%",     # Opens to a side-by-side split when clicked
                        bg="#f8f9fa",
                        padding="12px"
                    ),
                    # Main content area (100% width when sidebar is closed, 52% when open)
                    ui.output_ui("main_plot"),
                    fillable=True,
                    border=False
                ),
                full_screen=True,
                fill=True,
                class_="dashboard-plot-card mb-3"
                )
            ),

            # --- Main Navigation Tabs ---
            ui.navset_card_tab(
                # TAB 1: Interactive Explorer (Superzip + Flag Map + Dynamic KPIs)
                ui.nav_panel(
                    "🗺️ Interactive Explorer",

                    # Top Filter Bar
# Top Filter Bar
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
                                    "🧪 Chemical Subspace:",
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
                                        play_button=ui.tags.span("▶ Iniciar Animación", class_="btn btn-sm btn-outline-success"),
                                        pause_button=ui.tags.span("⏸ Pausar", class_="btn btn-sm btn-outline-warning")
                                    )
                                )
                            )
                        ),
                        class_="bg-light border rounded p-3 mb-3"
                    ),

                    # Superzip-inspired Map + Right Control Panel Layout
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.tags.span("🌐 Interactive Geopolitical Map & Country Selector", style="font-weight: 600;"),
                                ui.tags.span(
                                    "Bubble size & color intensity reflect average CS contribution over the selected period",
                                    style="font-size: 0.85em; color: #666; float: right;"
                                )
                            )
                        ),
                        ui.row(
                            ui.column(
                                8,
                                ui.output_ui("map_output")
                            ),
                            ui.column(
                                4,
                                ui.div(
                                    ui.input_radio_buttons(
                                        "map_visual_style",
                                        "🎨 Map Visual Style:",
                                        choices={
                                            "bubbles": "🟣 Proportional Bubbles + Color Intensity (Superzip Style)",
                                            "flags": "🏳️ Circular National Flag Badges",
                                            "polygons": "🗺️ Shaded Country Boundaries (GeoJSON)",
                                            "cartogram": "🧩 Deformed Cartogram Boundaries (Interactive)"
                                        },
                                        selected="cartogram"
                                    ),
                                    ui.hr(style="margin: 10px 0;"),
                                    ui.input_selectize(
                                        "country_picker",
                                        "🔍 Quick Search & Select Countries:",
                                        choices=initial_data['country_choices'],
                                        selected=["CN", "US"
                                                #   , "IN", "FR", "RU", "GB", "DE", "JP"
                                                  ],
                                        multiple=True,
                                        remove_button=True,
                                        options={"placeholder": "Type country name or ISO code..."}
                                    ),
                                    ui.input_radio_buttons(
                                        "display_mode_input",
                                        "📊 Analysis Mode:",
                                        choices={
                                            "compare_individuals": "Individual Country Trajectories",
                                            "find_collaborations": "Find Joint Collaborations Between Selected"
                                        },
                                        selected="compare_individuals"
                                    ),
                                    ui.input_checkbox(
                                        "show_flags_on_trends",
                                        "🏳️ Show Country Flags at End of Trend Lines",
                                        value=True
                                    ),
                                    ui.input_action_button(
                                        "clear_selection", "🗑️ Clear All Selected Countries",
                                        class_="btn-outline-danger w-100 mt-2 mb-2"
                                    ),
                                    ui.div(ui.output_text("selection_info"), class_="text-muted small text-center"),
                                    style="background-color: #f8f9fa; padding: 14px; border-radius: 8px; border: 1px solid #e5e7eb;"
                                )
                            )
                        ),
                        class_="border rounded mb-3"
                    ),

                                        # Dynamic KPI strip that updates with the user's map/country selection
                    ui.output_ui("dynamic_explorer_kpis"),

                    # Lower Output Tabs (Trends, Global Choropleth Snapshot, Summary Table)
                    ui.card(
                        ui.card_header("� Summary Statistics"),
                        ui.output_data_frame("summary_table"),
                        class_="mt-3"
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
                    "🤝 Top 10 Trends",
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
                            style="font-size: 0.9em; color: #666; text-align: center; margin-top: 8px;"
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
            "🔗 Useful Links & Methods",
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("📐 Fractional Authorship Formula & Subspace Definitions"),
                    ui.p(
                        "Each country's annual contribution to the expansion of the Chemical Space (CS) is weighted by the fractional affiliation of authors on papers reporting a substance for the first time:"
                    ),
                    ui.pre("C_{i,t} = (1 / n_t) * Σ_s (n_{i,s,t} / n_{a,s,t})"),
                    ui.tags.ul(
                        ui.tags.li(ui.tags.b("Organic CS:"), " Compounds containing Carbon (C). Dominated by CHNO compositions."),
                        ui.tags.li(ui.tags.b("Organometallic CS:"), " Compounds containing Carbon and at least one metal atom (Fe, Ru, Cu, Pd, Ni, Pt, etc.)."),
                        ui.tags.li(ui.tags.b("Rare-Earth (REE) CS:"), " Compounds containing Y, Sc, or lanthanides (La–Lu).")
                    )
                ),
                ui.card(
                    ui.card_header("🔗 Project & Author Resources"),
                    ui.tags.ul(
                        ui.tags.li(ui.tags.a("Published Article in Quantitative Science Studies (MIT Press)", href="https://doi.org/10.1162/QSS.a.471", target="_blank")),
                        ui.tags.li(ui.tags.a("Interactive App GitHub Repository", href="https://github.com/santi-rios/CS-Explorer-App", target="_blank")),
                        ui.tags.li(ui.tags.a("Site Built by Santiago G-Rios", href="https://santi-rios.github.io/", target="_blank"))
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
            """)
        ),
        title="Chemical Space Explorer",
        theme=shinyswatch.theme.flatly,
        footer=ui.div(
            ui.hr(),
            ui.p(
                "Source: Bermúdez-Montaña, M., García-Chung, A., Stadler, P. F., Jost, J., & Restrepo, G. (2025). "
                "China's rise in the chemical space and the decline of US influence. ",
                ui.tags.i("Quantitative Science Studies (MIT Press): "),
                ui.tags.a("https://doi.org/10.1162/QSS.a.471", href="https://doi.org/10.1162/QSS.a.471", target="_blank"),
                style="text-align: center; padding: 10px; font-size: 0.9em; color: #777;"
            )
        )
    )

    def server(input, output, session):
        # Default selection of top 5 countries so the app opens with rich plots immediately!
        selected_countries = reactive.Value(["CN", "US", 
                                            #  "IN", "FR", "RU", "GB", "DE", "JP"
                                             ])

        @reactive.Calc
        def country_list():
            return load_country_list()

        @reactive.Calc
        def filtered_data():
            selected_tuple = tuple(sorted(selected_countries.get())) if selected_countries.get() else ()
            return cached_get_display_data(
                selected_isos_tuple=selected_tuple,
                year_range=tuple(input.years()),
                chemical_category=input.chemical_category(),
                display_mode=input.display_mode_input(),
                region_filter=input.region_filter()
            )

        @reactive.Calc
        def region_all_countries_data():
            """Data for all countries in the current region/period (used to size & color map bubbles)."""
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

        # 1. Sync dropdown -> reactive selected_countries
        @reactive.Effect
        @reactive.event(input.country_picker, ignore_none=False, ignore_init=True)
        def _sync_picker_to_selected():
            picker_isos = list(input.country_picker() or [])
            if picker_isos != selected_countries.get():
                selected_countries.set(picker_isos)

        # 1. Allow the header button to open/close the right sidebar drawer
        @reactive.Effect
        @reactive.event(input.toggle_choropleth_drawer)
        def _toggle_right_drawer():
            is_currently_open = bool(input.choropleth_right_sidebar())
            ui.update_sidebar("choropleth_right_sidebar", show=not is_currently_open)

        # 2. Trigger Plotly responsive resize when the right sidebar opens or closes
        @reactive.Effect
        @reactive.event(input.choropleth_right_sidebar)
        async def _resize_plotly_on_sidebar_toggle():
            await session.send_custom_message("trigger_plotly_resize", {})
        
        # 2. Sync Folium map click -> reactive selected_countries AND dropdown
        
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

        # 3. Sync Reset button -> clear both
        @reactive.Effect
        @reactive.event(input.clear_selection)
        def _clear_all_selections():
            selected_countries.set([])
            ui.update_selectize("country_picker", selected=[])

        # 4. Update dropdown choices when Region changes
        @reactive.Effect
        @reactive.event(input.region_filter, ignore_init=True)
        def _handle_region_change():
            all_countries = country_list()
            current_region = input.region_filter()
            region_df = all_countries[all_countries['region'] == current_region] if current_region != "All" else all_countries
            new_choices = {
                row['iso2c']: f"{iso_to_flag_emoji(row['iso2c'])} {row['country']} ({row['iso2c']})"
                for _, row in region_df.iterrows()
            }
            valid_selected = [iso for iso in selected_countries.get() if iso in new_choices]
            selected_countries.set(valid_selected)
            ui.update_selectize("country_picker", choices=new_choices, selected=valid_selected)

        @output
        @render.ui
        def dynamic_explorer_kpis():
            """NZ Trade Dashboard style dynamic KPI cards reacting to user selection."""
            data = filtered_data()
            y_start, y_end = input.years()
            chem = input.chemical_category()
            mode = input.display_mode_input()

            if data.empty:
                return ui.div()

            if mode == "compare_individuals":
                by_country = data.groupby(['iso2c', 'country'])['total_percentage'].agg(['mean', 'max']).reset_index()
                by_country = by_country.sort_values('mean', ascending=False)
                top_row = by_country.iloc[0]
                leader_flag = iso_to_flag_emoji(top_row['iso2c'])
                leader_title = f"Top Selected ({y_start}–{y_end}): {top_row['country']}"
                leader_val = f"{top_row['mean']:.2f}% avg (Peak: {top_row['max']:.1f}%)"

                latest_yr = int(data['year'].max())
                latest_sum = data[data['year'] == latest_yr]['total_percentage'].sum()
                second_title = f"Combined Share in {latest_yr} ({len(by_country)} Selected)"
                second_val = f"{latest_sum:.2f}%"
            else:
                by_collab = data.groupby('plot_group')['total_percentage'].agg(['mean', 'max']).reset_index()
                by_collab = by_collab.sort_values('mean', ascending=False)
                top_row = by_collab.iloc[0]
                leader_flag = "🤝"
                leader_title = f"Top Collaboration ({y_start}–{y_end})"
                leader_val = f"{top_row['plot_group']} ({top_row['mean']:.2f}%)"
                second_title = "Active Collaboration Networks Found"
                second_val = f"{len(by_collab)} Networks"

            return ui.layout_column_wrap(
                ui.div(
                    ui.div(f"{leader_flag} {leader_title}", style="font-size: 0.82rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(leader_val, style="font-size: 1.35rem; font-weight: 700;"),
                    style="background: #0d9488; color: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                ui.div(
                    ui.div(f"📈 {second_title}", style="font-size: 0.82rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(second_val, style="font-size: 1.35rem; font-weight: 700;"),
                    style="background: #2563eb; color: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                ui.div(
                    ui.div("🧪 Active Subspace & Timeframe", style="font-size: 0.82rem; text-transform: uppercase; opacity: 0.9;"),
                    ui.div(f"{chem} ({y_start}–{y_end})", style="font-size: 1.35rem; font-weight: 700;"),
                    style="background: #4f46e5; color: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
                ),
                width=1/3,
                style="margin-bottom: 12px;"
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
            return f"{count} selected out of {available_count} countries in {current_region}"

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
            current_mode = input.display_mode_input()
            selected = selected_countries.get()

            if current_mode == "find_collaborations" and len(selected) < 2:
                return figure_to_html(create_empty_plot(
                    "🤝 Select at least 2 countries on the map or dropdown to find joint collaborations."
                ))

            data = filtered_data()
            if data.empty:
                if current_mode == "find_collaborations":
                    return figure_to_html(create_empty_plot("No collaborations found for the selected countries and filters."))
                return figure_to_html(create_empty_plot("Select one or more countries from the map or search box to view trends."))

            return figure_to_html(create_trends_plot(
                data=data,
                selected_countries=selected,
                mode=current_mode,
                show_flags=bool(input.show_flags_on_trends())
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
                elif mode == "cartogram_static":
                    return figure_to_html(
                        create_cartogram_plot(
                            choropleth_data,
                            main_title=f"Area-Deformed Cartogram ({y_start}–{y_end}) — {chem_cat}",
                            animated=False
                        )
                    )
                elif mode == "cartogram_anim":
                    return figure_to_html(
                        create_cartogram_plot(
                            choropleth_data,
                            main_title=f"Animated Deforming Cartogram ({y_start}–{y_end}) — {chem_cat}",
                            animated=True
                        )
                    )
                else:
                    return figure_to_html(create_contribution_choropleth(choropleth_data))
            except Exception as e:
                return figure_to_html(create_empty_plot(f"Error creating map: {str(e)}"))

        @output
        @render.data_frame
        def summary_table():
            data = filtered_data()
            if data.empty:
                msg = (
                    "Select at least 2 countries to find collaborations."
                    if input.display_mode_input() == "find_collaborations" and len(selected_countries.get()) < 2
                    else "No data available for current selections."
                )
                return pd.DataFrame({"Message": [msg]})
            return create_summary_dataframe(data, input.display_mode_input())

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
                        style="text-align: center; color: #666; font-size: 0.9em;"
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
            

    return App(app_ui, server, static_assets=www_dir)


app = create_app()

if __name__ == "__main__":
    app.run(port=8000)