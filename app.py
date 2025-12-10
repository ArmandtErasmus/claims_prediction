# import libraries
import streamlit as st 
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import geopandas as gp
import plotly.graph_objects as go

def load_data():

    df = pd.read_csv('data/historical_data.csv')
    #gdf = gp.read_file('data\Province.shp')
    return df#, gdf

def data_visualisation(data, map=None):

    #  colour :) and some page customisations
    base_color = "#df004c"
    st.markdown("<h2 style='color: #df004c;'>Data Visualisation and Analysis</h2>", unsafe_allow_html=True)

    # Importing the data
    df = data.copy()
    #gdf = map

    df2 = df.copy()
    df2["totalclaims"] = df2["pastclaims"] + df2["claims"]
    df2 = df2[["carcolour", "totalclaims"]]

    # car colour specific data (messy for now)
    total_white_cars = df2[df2["carcolour"] == "White"].shape[0]
    total_blue_cars = df2[df2["carcolour"] == "Blue"].shape[0]
    total_green_cars = df2[df2["carcolour"] == "Green"].shape[0]
    total_black_cars = df2[df2["carcolour"] == "Black"].shape[0]
    total_yellow_cars = df2[df2["carcolour"] == "Yellow"].shape[0]
    total_red_cars = df2[df2["carcolour"] == "Red"].shape[0]
    total_silver_cars = df2[df2["carcolour"] == "Silver"].shape[0]

    total_white_claims = df2.loc[df2["carcolour"] == "White", "totalclaims"].sum()
    total_blue_claims = df2.loc[df2["carcolour"] == "Blue", "totalclaims"].sum()
    total_green_claims = df2.loc[df2["carcolour"] == "Green", "totalclaims"].sum()
    total_black_claims = df2.loc[df2["carcolour"] == "Black", "totalclaims"].sum()
    total_yellow_claims = df2.loc[df2["carcolour"] == "Yellow", "totalclaims"].sum()
    total_red_claims = df2.loc[df2["carcolour"] == "Red", "totalclaims"].sum()
    total_silver_claims = df2.loc[df2["carcolour"] == "Silver", "totalclaims"].sum()

    white_rate = total_white_claims / total_white_cars
    blue_rate = total_blue_claims / total_blue_cars
    green_rate = total_green_claims / total_green_cars
    black_rate = total_black_claims / total_black_cars
    yellow_rate = total_yellow_claims / total_yellow_cars
    red_rate = total_red_claims / total_red_cars
    silver_rate = total_silver_claims / total_silver_cars

    car_counts = {
        "White": total_white_cars,
        "Blue": total_blue_cars,
        "Green": total_green_cars,
        "Black": total_black_cars,
        "Yellow": total_yellow_cars,
        "Red": total_red_cars,
        "Silver": total_silver_cars
    }

    claims = {
        "White": total_white_claims,
        "Blue": total_blue_claims,
        "Green": total_green_claims,
        "Black": total_black_claims,
        "Yellow": total_yellow_claims,
        "Red": total_red_claims,
        "Silver": total_silver_claims
    }

    # claims by car colour
    df_claims = pd.DataFrame({
        "carcolour" : list(claims.keys()),
        "claims" : list(claims.values()),
        "num_cars": list(car_counts.values())
    })

    # rates by car colour
    rates = {
        "White": white_rate,
        "Blue": blue_rate,
        "Green": green_rate,
        "Black": black_rate,
        "Yellow": yellow_rate,
        "Red": red_rate,
        "Silver": silver_rate
    }

    df_rates = pd.DataFrame({
        "carcolour": list(rates.keys()),
        "claim_rate": list(rates.values())
    })

    # overlay total cars by car colour and total claims by car colour
    df_overlay = pd.DataFrame({
        "carcolour": ["White", "Blue", "Green", "Black", "Yellow", "Red", "Silver"],
        "total_cars": [
            total_white_cars, total_blue_cars, total_green_cars, total_black_cars,
            total_yellow_cars, total_red_cars, total_silver_cars
        ],
        "total_claims": [
            total_white_claims, total_blue_claims, total_green_claims, total_black_claims,
            total_yellow_claims, total_red_claims, total_silver_claims
        ]
    })


    df_rates["claim_rate"] = df_rates["claim_rate"] * 1000 # per 1000 cars :D

    # Total claims
    df["total_claims"] = df["pastclaims"] + df["claims"]

    # Map gender 1/0 to labels
    df["gender_label"] = df["gender"].map({1: "Male", 0: "Female"})

    # Aggregate by gender
    claims_by_gender = df.groupby("gender_label")["total_claims"].sum()

    # Aggregate by car colour
    claims_by_car_colour = df.groupby("carcolour")["total_claims"].sum()



    # get claims counts data :)
    counts = df['claims'].value_counts().sort_index()
    zoom_counts = counts[counts.index.isin([3,4])]
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

    #merged = gdf.merge(
    #    claims_per_province,
    #    left_on="PROVINCE",
    #    right_on="province_full",
    #    how="left"
    #)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_blues", ["#ff7ea9", base_color])
    sns.set_style("whitegrid")

    plot_info = [
        {
            "title": "Number of Claims by Province",
            "type": "image", 
            "plot_func": lambda: st.image("data/map.png", use_column_width=True),
            "description": (
                "It is clear from the chart that the Western Cape has the most claims, "
                "followed by Gauteng. This makes sense because these two provinces are urban and therefore more active."
            )
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
        },
        {
            "title": "Distribution of the Number of Claims",
            "plot_func": lambda ax: (
                
                sns.barplot(x=counts.index, y=counts.values, color=base_color, ax=ax),
                
                (lambda axins: (
                    axins.bar(counts[counts.index.isin([3,4])].index,
                            counts[counts.index.isin([3,4])].values,
                            color=base_color),
                    axins.set_title("Zoom: 3-4 claims", fontsize=8),
                    axins.set_xticks([3, 4]),  
                    axins.set_xticklabels([3, 4]), 
                    axins.set_ylim(0, counts[counts.index.isin([3,4])].max()*1.2)
                ))(inset_axes(ax, width="40%", height="40%", loc="upper right"))
            ),
            "description": "It is clear that the data is zero-inflated, with zero claims submitted dominating the remaining claim counts."
        }
        
    ]

    for info in plot_info:
        col1, col2 = st.columns(2, vertical_alignment="top")
        
        with col1:
             
            if info.get("type") == "image":
                st.image("data/map.png", use_column_width=True)
            
            
            else:
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
                elif "Distribution" in info["title"]:
                    ax.set_xlabel("Number of Claims", fontsize=12, color=base_color, fontweight="bold")
                    ax.set_ylabel("Frequency", fontsize=12, color=base_color, fontweight="bold")
                    plt.xticks(color="black", fontsize=10)
                    plt.yticks(color="black", fontsize=10)
                st.pyplot(fig)
                st.write("---")
        
        with col2:
            st.write("---")
            col2.subheader(info["title"])
            st.write(info["description"])
            if info["title"] == "Number of Claims by Car Colour":
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=df_overlay["carcolour"],
                    y=df_overlay["total_claims"],
                    name="Amount of Claims",
                    marker_color="#801a3c",
                    text=df_overlay["total_claims"],
                    textposition="outside"  
                ))

                fig.add_trace(go.Bar(
                    x=df_overlay["carcolour"],
                    y=df_overlay["total_cars"],
                    name="Amount of Vehicles",
                    marker_color="#df004c",
                    text=df_overlay["total_cars"],
                    textposition="inside"
                ))

                fig.update_traces(texttemplate='%{text:.0f}')

                fig.update_layout(
                    barmode="overlay",       
                    xaxis_tickangle=0,
                    xaxis=dict(
                        title=dict(
                            text='Vehicle Colour', 
                            font=dict(size=14)  
                        ),
                        tickfont=dict(size=14)
                    ),
                    yaxis=dict(
                        title=dict(
                            text='Amount of Vehicles and Observed Claims',
                            font=dict(size=14) 
                        ),
                        tickfont=dict(size=14)
                    ),
                    template="plotly_white",
                    title_text='Total Amount of Claims in Relation to the Amount of Vehicles Across Vehicle Colours',
                    title_font=dict(size=16),
                    title_x=0.5,
                    title_xanchor='center',
                    height=600,
                    margin=dict(t=50)
                )

                st.plotly_chart(fig, use_container_width=True)
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

    col_zip, col_poi = st.columns(2, border=True)

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

            st.write("### ZIP Coefficients")
            zip_params = pd.DataFrame({
                "coef": zip_model.params,
            })
            st.dataframe(zip_params)

            st.session_state["zip_model"] = zip_model
            st.session_state["model_features"] = model_features
            st.session_state["selected_features"] = selected_features
            st.session_state["ohe_columns"] = ohe_columns
            st.session_state["cols_exog"] = X.columns.tolist()
            st.session_state["cols_exog_infl"] = X_infl.columns.tolist()

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

            st.session_state["poi_model"] = glm_model
            st.session_state["model_features"] = model_features
            st.session_state["selected_features"] = selected_features
            st.session_state["ohe_columns"] = ohe_columns

        except Exception as e:
            st.error(f"Poisson failed: {e}")

    st.subheader("Lift Chart")

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
    
    st.header("Claims Frequency Prediction")

    if "zip_model" not in st.session_state or "poi_model" not in st.session_state:
        st.warning("Please train a model first in 'Model Creation and Evaluation'.")
        return

    zip_model = st.session_state["zip_model"]
    poi_model = st.session_state["poi_model"]
    model_features = st.session_state["model_features"]
    selected_features = st.session_state["selected_features"]
    ohe_columns = st.session_state["ohe_columns"]

    st.subheader("Enter feature values")

    input_data = {}

    for feat in selected_features:

        if feat in ohe_columns:
            
            levels = sorted([c.split(f"{feat}_")[1] for c in model_features if c.startswith(feat + "_")])
            choice = st.selectbox(f"{feat}", levels)
            input_data[feat] = choice

        else:
            
            val = st.number_input(f"{feat}", value=0.0)
            input_data[feat] = val

    row_exog = {}
    row_infl = {}

    for col in st.session_state["cols_exog"]:
        if col == "const":
            row_exog[col] = 1.0
        elif "_" in col:  
            base, level = col.split("_")
            row_exog[col] = 1 if input_data.get(base) == level else 0
        else:
            row_exog[col] = input_data[col]

    for col in st.session_state["cols_exog_infl"]:
        if col == "const":
            row_infl[col] = 1.0
        elif "_" in col:
            base, level = col.split("_")
            row_infl[col] = 1 if input_data.get(base) == level else 0
        else:
            row_infl[col] = input_data[col]

    X_new = pd.DataFrame([row_exog])
    X_new_infl = pd.DataFrame([row_infl])

    exp_years = st.number_input("Exposure (exp_years)", value=1.0, min_value=0.0001)
    offset = np.log(exp_years)

    if st.button("Predict"):

        pred_zip = zip_model.predict(X_new, exog_infl=X_new_infl)
        pred_poi = poi_model.predict(X_new, offset=offset)

        st.write("---")

        st.subheader("Interpretation")

        def interpret_prediction(pred, model_name):
            yearly = float(pred)
            prob_zero = np.exp(-yearly)

            st.write(f"### {model_name} Model")
            st.write(f"- Expected number of claims **per year**: `{yearly:.4f}`")
            st.write(f"- Chance of **at least one claim**: `{(1 - prob_zero)*100:.1f}%`")
            st.write(f"- Chance of **zero claims**: `{prob_zero*100:.1f}%`")
            #st.write("---")

        col1, col2 = st.columns(2, border=True)

        with col1:

            interpret_prediction(pred_zip, "Zero-Inflated Poisson (ZIP)")

        with col2:
            interpret_prediction(pred_poi, "Standard Poisson")

        st.write("---")

        st.subheader("Risk Category")

        def risk_category(lambda_pred):
            if lambda_pred < 0.05:
                return "Very Low Risk Client", "🟢"
            elif lambda_pred < 0.10:
                return "Low Risk Client", "🟡"
            elif lambda_pred < 0.20:
                return "Medium Risk Client", "🟠"
            else:
                return "High Risk Client", "🔴"

        zip_cat, zip_icon = risk_category(float(pred_zip))
        poi_cat, poi_icon = risk_category(float(pred_poi))

        st.write(f"**ZIP Model:** {zip_icon} {zip_cat}")
        st.write(f"**Poisson Model:** {poi_icon} {poi_cat}")

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

    historical_data = load_data() #,za_map

    match mode:
            
            case "Data Visualisation and Analysis":
                data_visualisation(historical_data)

            case "Model Creation and Evaluation":
                model_evaluation(historical_data)

            case "Claims Frequency Prediction":
                claims_frequency_prediction()

            case _:
                text = st.write("Error")
                return text


if __name__ == "__main__":
    main()