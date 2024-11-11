import faicons as fa
import plotly.express as px
import pandas as pd

# Load data and compute static values
from shared import app_dir, new_df
from shinywidgets import render_plotly
from script import xgb

from shiny import reactive, render, App
from shiny.express import input, ui

new_df['zipcode'] = new_df['zipcode'].astype(str)
price_rng = (min(new_df.price), max(new_df.price))
bed_rng = (min(new_df.bedrooms), max(new_df.bedrooms))
bath_rng = (min(new_df.bathrooms), max(new_df.bathrooms))

# Add page title and sidebar
ui.page_opts(title="Housing Prices", fillable=True)

with ui.sidebar(open="desktop"):
    ui.input_slider(
        "price",
        "Price",
        min=price_rng[0],
        max=price_rng[1],
        value=price_rng,
        pre="$",
    )
    ui.input_slider(
        "bedrooms",
        "Bedrooms",
        min=bed_rng[0],
        max=bed_rng[1],
        value=bed_rng,
        
    )
    ui.input_slider(
        "bathrooms",
        "Bathrooms",
        min=bath_rng[0],
        max=bath_rng[1],
        value=bath_rng,
        
    )
    ui.input_checkbox_group(
        "homeType",
        "Home Type",
        ["CONDO", "SINGLE_FAMILY", "TOWNHOUSE", "MULTI_FAMILY"],
        selected=["CONDO", "SINGLE_FAMILY", "TOWNHOUSE", "MULTI_FAMILY"],
        inline=True,
    )
    ui.input_text_area(
        "zipcode", 
        "Zipcode",
        "",
        placeholder="Enter Zipcode",
        
        )
    
    ui.input_action_button("reset", "Reset filter")

# Add main content
ICONS = {
    "user": fa.icon_svg("user", "regular"),
    "wallet": fa.icon_svg("wallet"),
    "currency-dollar": fa.icon_svg("dollar-sign"),
    "ellipsis": fa.icon_svg("ellipsis"),
}

with ui.layout_columns(fill=False):
    with ui.value_box(showcase=ICONS["user"]):
        "Total Listings"

        @render.express
        def total_tippers():
            housing_data().shape[0]

    with ui.value_box(showcase=ICONS["currency-dollar"]):
        "Median Price"

        @render.express
        def average_tip():
            d = housing_data()
            if d.shape[0] > 0:
                
                f"{d.price.median():.2f}"

    with ui.value_box(showcase=ICONS["wallet"]):
        "Average sqft"

        @render.express
        def average_bill():
            d = housing_data()
            if d.shape[0] > 0:
                bill = d.sqft.mean()
                f"{bill:.2f}"


with ui.layout_columns(col_widths=[6, 6, 12]):
    with ui.card(full_screen=True):
        ui.card_header("Housing data")

        @render.data_frame
        def table():
            return render.DataGrid(housing_data())
        


    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "sqft vs price"
            with ui.popover(title="Add a color variable", placement="top"):
                ICONS["ellipsis"]
                ui.input_radio_buttons(
                    "scatter_color",
                    None,
                    ["homeType", "mortgageRates","bathrooms","sqft","bedrooms","yearBuilt","photoCount","price","school_count","closest_school_distance","crime_rate","Median_Household_Income","zipcode"],
                    inline=True,
                )

        @render_plotly
        def scatterplot():
            color = input.scatter_color()
            return px.scatter(
                housing_data(),
                x="sqft",
                y="price",
                color=None if color == "none" else color,
                trendline="lowess",
            )

    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "SQFT per Dollar"
            with ui.popover(title="Add a color variable"):
                ICONS["ellipsis"]
                ui.input_radio_buttons(
                    "tip_perc_y",
                    "Split by:",
                    ["mortgageRates","bathrooms","homeType","sqft","bedrooms","yearBuilt","photoCount","price","school_count","closest_school_distance","crime_rate","Median_Household_Income", 'zipcode'],
                    selected="homeType",
                    inline=True,
                )

        @render_plotly
        def housing_sqft():
            from ridgeplot import ridgeplot

            dat = housing_data()
            dat["percent"] = dat.sqft  / dat.price
            yvar = input.tip_perc_y()
            uvals = dat[yvar].unique()

            samples = [[dat.percent[dat[yvar] == val]] for val in uvals]

            plt = ridgeplot(
                samples=samples,
                labels=uvals,
                bandwidth=0.01,
                colorscale="viridis",
                colormode="row-index",
            )

            plt.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                )
            )

            return plt


ui.include_css(app_dir / "styles.css")

# --------------------------------------------------------
# Reactive calculations and effects
# --------------------------------------------------------




@reactive.calc
def housing_data():
    price = input.price()
    idx1 = new_df.price.between(price[0], price[1])
    idx2 = new_df.homeType.isin(input.homeType())
    bedrooms = input.bedrooms()
    idx3 = new_df.bedrooms.between(bedrooms[0], bedrooms[1])
    bathrooms = input.bathrooms()
    idx4 = new_df.bathrooms.between(bathrooms[0], bathrooms[1])
    zip = input.zipcode()
    if zip == "":
        idx5 = new_df.zipcode.notnull()
    else:
        idx5 = new_df.zipcode == zip
    return new_df[idx1 & idx2 & idx3 & idx4 & idx5]


@reactive.effect
@reactive.event(input.reset)
def _():
    ui.update_slider("price", value=price_rng)
    ui.update_slider("bedrooms", value=bed_rng)
    ui.update_slider("bathrooms", value=bath_rng)
    ui.update_checkbox_group("homeType", selected=["CONDO", "SINGLE_FAMILY", "TOWNHOUSE", "MULTI_FAMILY"])


