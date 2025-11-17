# import libraries
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gp

def load_data():

    df = pd.read_csv('data\historical_data.csv')
    gdf = gp.read_file('data\Province.shp')
    return df, gdf

def data_visualisation(data, map):

    col1, col2 = st.columns(2)

    # MiWay colour :)
    base_color = "#df004c"

    df = data.copy()
    gdf = map

    # total claims
    df["total_claims"] = df["pastclaims"] + df["claims"]

    # map gender 1/0 to labels
    df["gender_label"] = df["gender"].map({1: "Male", 0: "Female"})

    # aggregate by gender label
    claims_by_gender = df.groupby("gender_label")["total_claims"].sum()

    # aggregate by car colour label
    claims_by_car_colour = df.groupby("carcolour")["total_claims"].sum()

    # Province mapping
    province_map = {
        "MP":  "Mpumalanga",
        "GP":  "Gauteng",
        "WC":  "Western Cape",
        "LIM": "Limpopo",
        "EC":  "Eastern Cape",
        "FS":  "Free State",
        "NW":  "North West",
        "KZN": "KwaZulu-Natal",
        "NC":  "Northern Cape",
    }

    df["province_full"] = df["province"].map(province_map)

    claims_per_province = (df.groupby("province_full")["total_claims"].sum().reset_index())

    merged = gdf.merge(
        claims_per_province,
        left_on="PROVINCE",
        right_on="province_full",
        how="left"
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_blues", ["#ff7ea9", base_color]
    )
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 7))
        merged.plot(
            column="total_claims",
            cmap=cmap,
            linewidth=0.8,
            edgecolor="black",
            legend=True,
            legend_kwds={
                "label": "Number of Claims",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.02,
            },
            ax=ax
        )
        ax.set_title(
            "Number of Claims by Province",
            fontsize=14,
            color=base_color,
            loc="left",
            fontweight="bold",
        )
        ax.axis("off")
        st.pyplot(fig) 

    sns.set_style("whitegrid")

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x=claims_by_gender.index, y=claims_by_gender.values, color=base_color, ax=ax)
        ax.set_title(
            "Number of Claims by Gender", 
            fontsize=14,
            color=base_color,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel(
            "Gender",
            fontsize=12,
            color=base_color,
            fontweight="bold"
        )
        ax.set_ylabel("Total Claims", fontsize=12, color=base_color, fontweight="bold")
        plt.xticks(color="#e23e75", fontsize=10)
        plt.yticks(color="#e23e75", fontsize=10)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.barplot(x=claims_by_car_colour.index, y=claims_by_car_colour.values, color=base_color, ax=ax)
        ax.set_title(
            "Number of Claims by Car Colour",
            fontsize=14,
            color=base_color,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel("Car Colour", fontsize=12, color=base_color, fontweight="bold")
        ax.set_ylabel("Total Claims", fontsize=12, color=base_color, fontweight="bold")
        plt.xticks(rotation=45, color="#e23e75", fontsize=10)
        plt.yticks(color="#e23e75", fontsize=10)
        st.pyplot(fig)

def model_evaluation():

    text = st.write("Model Evaluation")
    return text

def claims_frequency_prediction():
    
    text = st.write("Claims Frequency Prediction")
    return text

def sidebar():

    with st.sidebar:

        mode = st.radio("Section", ["Data Visualisation", "Model Evaluation", "Claims Frequency Prediction"])

        return mode
        

def main():

    st.set_page_config(
        page_title = 'Claims Frequency Prediction with Poission Regression',
        layout = 'wide',
        initial_sidebar_state = 'expanded'
    )

    mode = sidebar()

    historical_data, za_map = load_data()

    match mode:
            
            case "Data Visualisation":
                data_visualisation(historical_data, za_map)

            case "Model Evaluation":
                model_evaluation()

            case "Claims Frequency Prediction":
                claims_frequency_prediction()

            case _:
                text = st.write("Error")
                return text


if __name__ == "__main__":
    main()