import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


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


def initialize_population_statistics():
    # Store population statistics
    population = 111265058
    herd_immunity_factor = 0.7
    target_vaccination = round(population * herd_immunity_factor)
    return population, herd_immunity_factor, target_vaccination


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
        f"Data was last updated on {recent_date.strftime('%b %d, %Y')} from Our World in Data."
    )

    # Print recent numbers
    st.markdown(
        f"As of **{recent_date.strftime('%b %d, %Y')}**, we have successfully vaccinated **{recent_people_vaccinated:,}** "
        f"individuals wherein **{recent_people_fully_vaccinated:,}** have been fully vaccinated."
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
    ax.set_title("Vaccinated Individuals (in millions)")
    st.pyplot(fig)
    st.caption(f"Source: COVID-19 Dataset from Our World in Data ({data_source}).")


def predictive_analytics():
    # Load the data
    vaccination, data_source = load_data()
    recent_date, recent_people_vaccinated, recent_people_fully_vaccinated = get_latest_numbers(vaccination)
    population, herd_immunity_factor, target_vaccination = initialize_population_statistics()

    # Write the title and subheader
    st.title(":syringe: PH Vaccination Rate: Where We Are Headed")
    st.subheader(
        f"Using the Facebook Prophet forecasting API, we tried to predict the date when the Philippines will "
        f"achieve herd immunity. With a population of **{population:,}**, herd immunity can be achieved once "
        f"**{target_vaccination:,}** individuals are vaccinated."
    )


def methodology():
    # Write the title and subheader
    st.title(":syringe: PH Vaccination Rate: Methodology")
    st.subheader(
        f"Forecasting: https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd"
    )

list_of_pages = [
    "Where We Are",
    "Where We Are Headed",
    "Methodology"
]

st.sidebar.title('Section')
selection = st.sidebar.radio("Go to", list_of_pages)

if selection == "Where We Are":
    descriptive_analytics()

elif selection == "Where We Are Headed":
    predictive_analytics()

elif selection == "Methodology":
    methodology()
