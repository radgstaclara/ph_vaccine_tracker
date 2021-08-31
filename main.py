import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES


@st.cache
def load_data():
    # Load the data
    vaccination = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    )

    # Save the source
    data_source = "https://github.com/owid/covid-19-data"

    # Filter country
    country_filter = "PHL"
    vaccination = vaccination[vaccination["iso_code"] == country_filter]

    # Filter columns
    columns = ["date", "people_vaccinated", "people_fully_vaccinated"]
    vaccination = vaccination[columns]

    # Transform date
    vaccination["date"] = pd.to_datetime(vaccination["date"])

    # Apply forward-fill for dates with no data
    vaccination['people_vaccinated'] = vaccination['people_vaccinated'].ffill()
    vaccination['people_fully_vaccinated'] = vaccination['people_fully_vaccinated'].ffill()

    # Replace NA with 0
    vaccination = vaccination.fillna(0)

    return vaccination, data_source


@st.cache
def initialize_population_statistics():
    # Store population statistics
    population = 111265058
    herd_immunity_factor = 0.7
    target_vaccination = round(population * herd_immunity_factor)
    return population, herd_immunity_factor, target_vaccination


@st.cache
def get_latest_numbers(vaccination):
    # Get latest numbers
    recent_date = max(vaccination["date"])
    recent_data = vaccination[vaccination["date"] == recent_date]
    recent_people_vaccinated = int(recent_data["people_vaccinated"])
    recent_people_fully_vaccinated = int(recent_data["people_fully_vaccinated"])
    return recent_date, recent_people_vaccinated, recent_people_fully_vaccinated


def descriptive_analytics():
    # Load the data
    vaccination, data_source = load_data()
    recent_date, recent_people_vaccinated, recent_people_fully_vaccinated = get_latest_numbers(vaccination)

    # Write the title and the subheader
    st.title(":syringe: PH Vaccination Rate: Where We Are")
    st.subheader(
        f"This visualization aims to provide real-time data on the current state of vaccination in the Philippines. "
        f"Data was updated last {recent_date.strftime('%b %d, %Y')} from Our World in Data."
    )

    # Plot the data
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.plot(
        vaccination['date'],
        vaccination['people_vaccinated'] / 1000000,
        label="Total Number of Vaccinated (1st dose)"
    )
    ax.plot(
        vaccination['date'],
        vaccination['people_fully_vaccinated'] / 1000000,
        label="Total Number of Fully Vaccinated (2nd dose)"
    )
    ax.legend(loc="upper left")
    ax.set_ylabel("Vaccinated Inviduals (in millions)")
    st.pyplot(fig)
    st.caption(f"Source: COVID-19 Dataset from Our World in Data ({data_source}).")

    # Print recent numbers
    st.markdown(
        f"As of **{recent_date.strftime('%b %d, %Y')}**, we have successfully vaccinated **{recent_people_vaccinated:,}** "
        f"individuals wherein **{recent_people_fully_vaccinated:,}** have been fully vaccinated."
    )


def predictive_analytics():
    # Load the data
    vaccination, data_source = load_data()
    population, herd_immunity_factor, target_vaccination = initialize_population_statistics()

    # Initialize forecast horizon
    forecast_horizon = 1000

    # Generate predict_start and predict_end date
    train_end = max(vaccination["date"])
    predict_start = train_end + pd.Timedelta(days=1)
    predict_end = predict_start + pd.Timedelta(days=forecast_horizon - 1)
    prediction_date_range = pd.date_range(start=predict_start, end=predict_end)

    # Prepare the data
    vaccination = vaccination[["date", "people_fully_vaccinated"]]
    vaccination.set_index('date', inplace=True)

    # Drop 0 values
    vaccination = vaccination[vaccination["people_fully_vaccinated"] != 0]

    # Initialize the model
    model = HWES(
        endog=vaccination["people_fully_vaccinated"],
        trend='add',
        seasonal='mul',
        freq="D"
    )

    # Fit the model
    model_fit = model.fit(
        smoothing_level=0.4210526,
        smoothing_trend=0.0526316,
        smoothing_seasonal=0.5789474
    )

    # Forecast
    prediction_values = model_fit.forecast(forecast_horizon)

    # Create dataframe for predictions
    predictions_dictionary = {
        'date': prediction_date_range,
        'people_fully_vaccinated': prediction_values
    }
    predictions = pd.DataFrame(predictions_dictionary)

    # Make date as index
    predictions.set_index("date", inplace=True)

    # Filter predictions
    vaccinated_filter = predictions["people_fully_vaccinated"] < target_vaccination
    predictions = predictions[vaccinated_filter]

    # Target
    herd_immunity_date = max(predictions.index) + pd.Timedelta(days=1)

    # Write the title and subheader
    st.title(":syringe: PH Vaccination Rate: Forecast")
    st.markdown(
        f"Using **Holt-Winters Exponential Smoothing**, we tried to predict the date when the Philippines will "
        f"achieve herd immunity based on its current vaccination rate."
    )

    # Plot the data
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.plot(
        vaccination.index,
        vaccination["people_fully_vaccinated"] / 1000000,
        color="tab:blue",
        label="Fully vaccinated individuals (actual)"
    )
    ax.plot(
        predictions.index,
        predictions["people_fully_vaccinated"] / 1000000,
        color="tab:orange",
        label="Fully vaccinated individuals (forecasted)"
    )
    ax.legend(loc="upper left")
    ax.set_ylabel("Vaccinated Inviduals (in millions)")
    ax.set_title(
        f"We expect to reach target herd immunity by "
        f"{herd_immunity_date.strftime('%b %d, %Y')}."
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.caption(f"Source: COVID-19 Dataset from Our World in Data ({data_source}).")

    st.markdown(
        f"With a population of **{population:,}**, herd immunity can be achieved "
        f"once **{target_vaccination:,}** individuals are vaccinated (that is, 70% of the total population)."
    )


def methodology():
    # Write the title and subheader
    st.title(":syringe: PH Vaccination Rate: Methodology")
    st.markdown(
        f":memo: **Note:** this web app is still under development. Kindly reach out to aaronstaclara@gmail.com "
        f"for feedback."
    )
    st.markdown(
        ":one: The data is sourced from Our World in Data. We expect that this web app will be able to "
        "reflect the current vaccination rate in the Philippines by displaying the current number of vaccinated individuals."
    )
    st.markdown(
        ":two: In forecasting, we utilized the Holt-Winters Exponential Smoothing statistical model. Note that"
        " model parameters used were the default values of the model. To test the model's performance, tuning "
        "will be implemented in future iterations."
    )

    st.markdown(
        "**#OustDuterte** "
        "**#DuqueResign**"
    )


list_of_pages = [
    "Where We Are",
    "Forecast",
    "Methodology"
]

st.sidebar.title(':scroll: Main Pages')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "Where We Are":
    descriptive_analytics()

elif selection == "Forecast":
    predictive_analytics()

elif selection == "Methodology":
    methodology()
