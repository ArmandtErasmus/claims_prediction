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
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

def load_data():

    df = pd.read_csv('data\historical_data.csv')
    gdf = gp.read_file('data\Province.shp')
    return df, gdf

def data_visualisation(data, map):

    #  colour :) and some page customisations
    base_color = "#df004c"
    st.markdown("<h2 style='color: #df004c;'>Data Visualisation and Analysis</h2>", unsafe_allow_html=True)

    # Importing the data
    df = data.copy()
    gdf = map

    # Total claims
    df["total_claims"] = df["pastclaims"] + df["claims"]

    # Map gender 1/0 to labels
    df["gender_label"] = df["gender"].map({1: "Male", 0: "Female"})

    # Aggregate by gender
    claims_by_gender = df.groupby("gender_label")["total_claims"].sum()

    # Aggregate by car colour
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

    claims_per_province = df.groupby("province_full")["total_claims"].sum().reset_index()

    merged = gdf.merge(
        claims_per_province,
        left_on="PROVINCE",
        right_on="province_full",
        how="left"
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_blues", ["#ff7ea9", base_color])
    sns.set_style("whitegrid")

    plot_info = [
        {
            "title": "Number of Claims by Province",
            "plot_func": lambda ax: merged.plot(
                column="total_claims",
                cmap=cmap,
                linewidth=0.8,
                edgecolor="black",
                legend=True,
                legend_kwds={"label": "Number of Claims",
                            "orientation": "horizontal",
                            "shrink": 0.6,
                            "pad": 0.02},
                ax=ax
            ),
            "description": "It is clear from the chart that the Western Cape has the most claims, followed by Gauteng. "
                        "This makes sense because these two provinces are urban and therefore more active."
        },
        {
            "title": "Number of Claims by Gender",
            "plot_func": lambda ax: sns.barplot(
                x=claims_by_gender.index,
                y=claims_by_gender.values,
                color=base_color,
                ax=ax
            ),
            "description": "The number of claims by gender appears evenly split between females and males, "
                        "with females having submitted slightly more claims than males."
        },
        {
            "title": "Number of Claims by Car Colour",
            "plot_func": lambda ax: sns.barplot(
                x=claims_by_car_colour.index,
                y=claims_by_car_colour.values,
                color=base_color,
                ax=ax
            ),
            "description": "It is clear that white cars have the most claims, followed by red cars, and then black cars, "
                        "with yellow cars having the least. This does not necessarily imply higher risk for white cars; "
                        "it is because white cars are more common."
        }
    ]

    for info in plot_info:
        col1, col2 = st.columns(2, vertical_alignment="top")
        
        with col1:
            fig, ax = plt.subplots(figsize=(7, 7))
            info["plot_func"](ax)
            ax.set_title(
                info["title"],
                fontsize=14,
                color=base_color,
                fontweight="bold",
                loc="left",
                pad=15
            )
            if "Gender" in info["title"]:
                ax.set_xlabel("Gender", fontsize=12, color=base_color, fontweight="bold")
                ax.set_ylabel("Total Claims", fontsize=12, color=base_color, fontweight="bold")
                plt.xticks(color="black", fontsize=10)
                plt.yticks(color="black", fontsize=10)
            elif "Car Colour" in info["title"]:
                ax.set_xlabel("Car Colour", fontsize=12, color=base_color, fontweight="bold")
                ax.set_ylabel("Total Claims", fontsize=12, color=base_color, fontweight="bold")
                plt.xticks(rotation=45, color="black", fontsize=10)
                plt.yticks(color="black", fontsize=10)
            elif "Province" in info["title"]:
                ax.axis("off")
            st.pyplot(fig)
            st.write("---")
        
        with col2:
            col2.subheader(info["title"])
            st.write(info["description"])
            st.write("---")

def model_evaluation(data):

    st.header("Model Evaluation")

    df = data.copy()

    feature_options = ["province", "carcolour", "hp", "gender", "age", "lic_years", "pastclaims", "exp_years"]

    st.write("Select the features to include in the GLM model:")
    selected_features = st.multiselect(
        "Choose at least one feature:",
        feature_options,
        default=[],
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one feature to train the model.")
        return

    ohe_columns = ["province", "carcolour"]
    df_encoded = pd.get_dummies(df, columns=ohe_columns, drop_first=True)
    df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes('bool').columns})

    model_features = []
    for feat in selected_features:
        if feat in ohe_columns:
            model_features.extend([col for col in df_encoded.columns if col.startswith(f"{feat}_")])
        else:
            model_features.append(feat)

    st.write("### Model Features Used")
    st.code(", ".join(model_features))

    y = df_encoded["claims"]
    X = df_encoded[model_features]
    X = sm.add_constant(X)

    offset = np.log(df_encoded["exp_years"])

    X_infl = sm.add_constant(X)

    col_zip, col_poi = st.columns(2)

    with col_zip:
        st.subheader("Zero-Inflated Poisson (ZIP)")

        try:
            zip_model = ZeroInflatedPoisson(
                endog=y,
                exog=X,
                exog_infl=X_infl,
                inflation='logit'
            ).fit(maxiter=200, method="bfgs")

            st.success("ZIP model trained successfully!")

            st.write("### ZIP Metrics")
            st.write(f"**AIC:** {zip_model.aic:.3f}")
            st.write(f"**BIC:** {zip_model.bic:.3f}")
            st.write(f"**Log-Likelihood:** {zip_model.llf:.3f}")

            null_model_llf = ZeroInflatedPoisson(
                endog=y,
                exog=np.ones((len(y), 1)),
                exog_infl=np.ones((len(y), 1)),
                inflation='logit'
            ).fit(maxiter=200, method="bfgs", disp=0).llf
            pseudo_r2 = 1 - zip_model.llf / null_model_llf
            st.write(f"**Pseudo R²:** {pseudo_r2:.3f}")

            df_encoded["pred_zip"] = zip_model.predict(X, exog_infl=X_infl)

            st.write("### ZIP Coefficients (no SE reported)")
            zip_params = pd.DataFrame({
                "coef": zip_model.params,
            })
            st.dataframe(zip_params)

        except Exception as e:
            st.error(f"ZIP failed: {e}")

    with col_poi:
        st.subheader("Standard Poisson GLM")

        try:
            formula = "claims ~ " + " + ".join(model_features)

            glm_model = smf.glm(
                formula=formula,
                data=df_encoded,
                family=sm.families.Poisson(),
                offset=offset
            ).fit()

            st.success("Poisson model trained successfully!")

            st.write("### Poisson Summary")
            st.markdown(glm_model.summary().as_html(), unsafe_allow_html=True)

            df_encoded["pred_poi"] = glm_model.predict(df_encoded)

        except Exception as e:
            st.error(f"Poisson failed: {e}")

    st.subheader("Lift Chart (Model Comparison)")

    try:
        df_encoded["decile"] = pd.qcut(df_encoded["pred_zip"], 10, labels=False) + 1

        lift = df_encoded.groupby("decile").agg(
            avg_zip=("pred_zip", "mean"),
            avg_poi=("pred_poi", "mean"),
            avg_actual=("claims", "mean"),
        ).sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
        ax.plot(lift.index, lift["avg_actual"], marker="o", label="Actual")
        ax.plot(lift.index, lift["avg_zip"], marker="o", label="ZIP Predicted")
        ax.plot(lift.index, lift["avg_poi"], marker="o", label="Poisson Predicted")
        ax.set_xlabel("Decile (10 = highest predicted risk)")
        ax.set_ylabel("Mean Claims")
        ax.set_title("Lift Chart")
        ax.legend()

        st.pyplot(fig, use_container_width=False)

    except Exception as e:
        st.error(f"Lift chart failed: {e}")

def claims_frequency_prediction():
    
    text = st.write("Claims Frequency Prediction")
    return text

def sidebar():

    with st.sidebar:

        mode = st.radio("Section", ["Data Visualisation and Analysis", "Model Creation and Evaluation", "Claims Frequency Prediction"])

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
            
            case "Data Visualisation and Analysis":
                data_visualisation(historical_data, za_map)

            case "Model Creation and Evaluation":
                model_evaluation(historical_data)

            case "Claims Frequency Prediction":
                claims_frequency_prediction()

            case _:
                text = st.write("Error")
                return text


if __name__ == "__main__":
    main()