# --- Gerekli kütüphaneler ---
import os
import re
from ast import literal_eval
import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Rastgeleliğin tekrarlanabilir olması için
np.random.seed(42)

# --- ENV Ayarları ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# --- Başlık ---
st.title("Bulaşıcı Hastalıklar Analizi & Eyalet Benzerliği")

# --- CSV dosyalarını oku ---
raw_df1 = pd.read_csv("data_1.csv", delimiter=';', engine='python')
raw_df2 = pd.read_csv("data_2.csv", delimiter=';', engine='python')

# --- Tarih formatlarını düzelt ---
raw_df1["Week Ending Date"] = raw_df1["Week Ending Date"].str.replace(".", "/", regex=False)
raw_df1["Week Ending Date"] = pd.to_datetime(raw_df1["Week Ending Date"], format="%m/%d/%Y", errors="coerce")

raw_df2["Week Ending Date"] = raw_df2["Week Ending Date"].str.replace(".", "/", regex=False)
raw_df2["Week Ending Date"] = pd.to_datetime(raw_df2["Week Ending Date"], format="%d/%m/%Y", errors="coerce")

# --- Sütunları filtrele ---
columns_to_hold_df1 = ['Jurisdiction of Occurrence', 'Week Ending Date', 'Septicemia (A40-A41)', 'Influenza and pneumonia (J10-J18)']
columns_to_hold_df2 = ['Jurisdiction of Occurrence', 'Week Ending Date', 'Septicemia (A40-A41)', 'Influenza and pneumonia (J09-J18)']

filtered_raw_df1 = raw_df1[columns_to_hold_df1]
filtered_raw_df2 = raw_df2[columns_to_hold_df2]

filtered_raw_df1.columns = filtered_raw_df1.columns.str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
filtered_raw_df2.columns = filtered_raw_df1.columns.str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
filtered_raw_df1 = filtered_raw_df1[filtered_raw_df2.columns]

# --- Birleştir ---
merged_df = pd.concat([filtered_raw_df1, filtered_raw_df2], axis=0, ignore_index=True)
copy_merged = merged_df.copy()
copy_merged.set_index('Week Ending Date', inplace=True)

# --- Sayısallaştır ---
infectious_diseases = ['Septicemia', 'Influenza and pneumonia']
for col in infectious_diseases:
    copy_merged[col] = pd.to_numeric(copy_merged[col], errors='coerce')

# --- Eyalet listesi ---
states_list = copy_merged['Jurisdiction of Occurrence'].dropna().unique().tolist()
states_list = [state for state in states_list if state != "United States"]

# --- Eyalet seçimi ---
state = st.selectbox("Bir eyalet seçin:", sorted(states_list))
st.write(f"Seçilen eyalet: {state}")

# --- Takvimle tarih seçimi ---
min_date = datetime.date(2024, 1, 1)
selected_date = st.date_input("Tarih seçin:", value=min_date, min_value=min_date)
st.write("Seçilen tarih:", selected_date.strftime("%d-%m-%Y"))

# --- LLM modeli ---
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)

def extract_list_from_response(response: str) -> list[str]:
    try:
        code_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_blocks:
            return literal_eval(code_blocks[0])
        list_text = re.search(r"\[.*?\]", response)
        if list_text:
            return literal_eval(list_text.group(0))
    except Exception as e:
        print("Liste çıkarma hatası:", e)
    return []

@st.cache_data(show_spinner=True)
def get_similar_states_via_llm(user_state: str, states: list[str], date: str) -> list[str]:
    prompt = f"""
Aşağıda listesi verilen ABD eyaletleri içinde, {user_state} eyaletine {date} tarihinde
iklim, coğrafya ve kültürel yapı açısından en çok benzeyen 3 tanesini sırala.

Bu tarihteki mevsimsel koşulları da dikkate al.

Eyalet listesi: {states}

Yalnızca Python listesi formatında 3 eyalet döndür: örneğin ["Arizona", "Nevada", "New Mexico"]
"""
    try:
        response = llm.invoke(prompt).content
        return extract_list_from_response(response)
    except Exception as e:
        print("Benzer eyalet tahmini hatası:", e)
        return []

# --- Benzer eyaletleri al ---
states = get_similar_states_via_llm(state, states_list, selected_date)
states_copy = states
states = [state] + states  # ilk eyalet kullanıcı seçimi

# --- Verileri süz ---
filtered_df_1 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[0]]
filtered_df_2 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[1]]
filtered_df_3 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[2]]
filtered_df_4 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[3]]

dfs = [filtered_df_1, filtered_df_2, filtered_df_3, filtered_df_4]

# --- Eksik verileri doldurma fonksiyonu ---
def generate_imputed_value(katsayi):
    if katsayi >= 0.7:
        return np.random.randint(1, 3)
    elif 0.4 <= katsayi < 0.7:
        return np.random.randint(3, 6)
    else:
        return np.random.randint(6, 10)

def fill_missing_across_states(dfs, group_col, cols_to_fill):
    combined_df = pd.concat(dfs, ignore_index=True)
    filled_df_list = []

    for df in dfs:
        filled_df = df.copy()
        for col in cols_to_fill:
            nan_counts = combined_df.groupby(group_col)[col].apply(lambda x: x.isna().sum())
            total_nan = nan_counts.sum()
            for state in df[group_col].unique():
                state_mask = (df[group_col] == state)
                state_nan_count = df.loc[state_mask, col].isna().sum()
                if total_nan != 0:
                    katsayi = 1 - (nan_counts[state] / total_nan)
                    filled_values = [generate_imputed_value(katsayi) for _ in range(state_nan_count)]
                    filled_df.loc[state_mask & df[col].isna(), col] = filled_values
        filled_df_list.append(filled_df)

    return filled_df_list

# --- Aykırı değer smoothing fonksiyonu ---
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

def smooth_outliers_in_dfs_for_multiple_diseases(dfs, diseases, sigma=2, z_threshold=2):
    smoothed_dfs = []

    # Iterate through each DataFrame
    for df_idx, dataframe in enumerate(dfs):
        df = dataframe.copy()
        jurisdiction = df['Jurisdiction of Occurrence'].iloc[0] if 'Jurisdiction of Occurrence' in df.columns else f"Dataset {df_idx}"
        print(f"Processing: {jurisdiction}")

        # Iterate through each disease
        for disease_name in diseases:
            if disease_name not in df.columns:
                print(f"  - {disease_name} column not found in {jurisdiction}")
                continue

            try:
                # Get the disease data and fill missing values
                counts = df[disease_name].copy().fillna(method='ffill').fillna(method='bfill').fillna(0)

                # Ensure there are enough data points for processing
                if len(counts) > 2:
                    # Calculate Z-scores
                    z_scores = zscore(counts)

                    # Detect outliers based on Z-score threshold
                    outliers = np.abs(z_scores) > z_threshold

                    # Apply Gaussian smoothing to the data
                    smoothed_counts = gaussian_filter1d(counts.values, sigma=sigma)

                    # Replace outliers with smoothed values
                    counts[outliers] = smoothed_counts[outliers]

                    # Update the DataFrame with smoothed values
                    df[disease_name] = counts

                    print(f"    ✓ Success: {sum(outliers)} outliers smoothed")
                else:
                    print(f"    ! Not enough data for {disease_name} in {jurisdiction}")

            except Exception as e:
                print(f"    ! Error processing {disease_name} in {jurisdiction}: {e}")

        # Append the processed DataFrame to the list
        smoothed_dfs.append(df)

    return smoothed_dfs


group_col = 'Jurisdiction of Occurrence'
cols_to_fill = ['Septicemia', 'Influenza and pneumonia']

# --- Eksik verileri doldur ---
filled_dfs = fill_missing_across_states(dfs, group_col, cols_to_fill)

# --- Ayrı ayrı al ---
filled_df_1, filled_df_2, filled_df_3, filled_df_4 = filled_dfs

# --- Smoothing uygula ---
smoothed_dfs = smooth_outliers_in_dfs_for_multiple_diseases(filled_dfs, cols_to_fill)

def forecast_cases_for_smooth_dfs(location, date, diseases, smooth_dfs, exog_column=None):
    """
    Make forecasts based on smoothed DataFrames for a given location and diseases.

    Parameters:
        location (str): The location (state) for which to forecast.
        date (str): The target forecast date.
        diseases (list): List of diseases to forecast.
        smooth_dfs (list): List of smoothed DataFrames.
        exog_column (str, optional): External regressor column name.

    Returns:
        dict: A dictionary with forecast results for each disease in the specified location.
    """
    result = {location: {}}

    try:
        # Find the relevant data for the given location in smoothed DataFrames
        location_data = None
        for df in smooth_dfs:
            if location in df['Jurisdiction of Occurrence'].unique():
                location_data = df[df['Jurisdiction of Occurrence'] == location]
                break

        if location_data is None:
            raise ValueError(f"No data found for {location}.")

        # Ensure the DataFrame has a time index
        if not isinstance(location_data.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame does not have a time index.")

        # Convert date to Timestamp
        date = pd.Timestamp(date)
        last_date = location_data.index[-1]

        # Skip forecasting if the target date is in the past or is already in the data
        if date <= last_date:
            print("Selected date is in the past or within the data range. Forecasting not needed.")
            return result

        # Calculate the number of weeks ahead to forecast
        delta_weeks = ((date - last_date).days) // 7
        if delta_weeks < 1:
            delta_weeks = 1  # Ensure at least one week of forecast

        # Forecast each disease
        for disease in diseases:
            if disease not in location_data.columns:
                result[location][disease] = "No data"
                continue

            # Get the disease data and drop NaN values
            y = location_data[disease].dropna()

            if len(y) > 2:
                if exog_column and exog_column in location_data.columns:
                    # Use external data if available
                    X = location_data[exog_column].dropna()
                    common_index = y.index.intersection(X.index)
                    y = y.loc[common_index]
                    X = X.loc[common_index]

                    # Fit SARIMAX model with exogenous variables
                    model = SARIMAX(y, order=(2, 1, 2), seasonal_order=(1, 1, 1, 52), exog=X)
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.forecast(steps=delta_weeks, exog=None)  # Exogenous variables are not forecasted
                else:
                    # Fit SARIMAX model without exogenous variables
                    model = SARIMAX(y, order=(1, 1, 1))
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.forecast(steps=delta_weeks)

                # Get the forecasted value for the last step
                forecast_value = round(forecast.iloc[-1])
                result[location][disease] = int(forecast_value)
            else:
                result[location][disease] = "Insufficient data"

    except Exception as e:
        print(f"Error occurred: {type(e).__name__} - {e}")
        result[location] = "Forecasting failed"

    return result

# Forecast for all states
forecast_results_all_states = []
diseases = ['Septicemia', 'Influenza and pneumonia']

for state in states:
    forecast_results_smooth = forecast_cases_for_smooth_dfs(
        location=state,
        date=selected_date,
        diseases=diseases,
        smooth_dfs=smoothed_dfs
    )

    forecast_results_all_states.append(forecast_results_smooth)

    print(f"\nForecast Results (Smoothed Data) - {state}:")
    print(forecast_results_smooth)

# Function to process CSV and calculate death rates
def calculate_death_rate(csv_path, forecast_results_all_states):
    try:
        # Load the CSV file from the provided path
        df = pd.read_csv(csv_path)

        # Convert 'Population' to integer
        df['Population'] = df['Population'].astype(int)

        # Convert to dict: State -> Population
        population_data = dict(zip(df['State'], df['Population']))

        results_with_death_rate = []

        # Iterate through forecast results and calculate death rates
        for state_data in forecast_results_all_states:
            for state, disease_data in state_data.items():
                if state in population_data:
                    total_deaths = disease_data.get('Septicemia', 0) + disease_data.get('Influenza and pneumonia', 0)
                    population = population_data[state]
                    death_rate = (total_deaths / population) * 1000000  # Death rate per 1M
                    disease_data['Death Rate per 1M'] = round(death_rate, 2)
                    results_with_death_rate.append({state: disease_data})
                else:
                    st.write(f"Population data not found for: {state}")
                    
        return results_with_death_rate
    except Exception as e:
        st.write(f"Error processing CSV or calculating death rates: {str(e)}")
        return []
    
csv_path = "state_populations.csv"
    
    # Call function to calculate death rates
results_with_death_rate = calculate_death_rate(csv_path, forecast_results_all_states)

if results_with_death_rate:
    st.write("Forecast Results with Death Rates:")
    st.write(results_with_death_rate)
else:
    st.write("No results available.")

# Güvenli eyalet seçimi fonksiyonu
def choose_safest_state_via_llm(date: str, death_data_list: list) -> str:
    # Liste içindeki dictionary'leri tek bir dictionary'ye dönüştür
    death_data = {}
    for item in death_data_list:
        death_data.update(item)

    # Normalize edilmiş (küçük harfli) eyalet verisi
    normalized_death_data = {
        state.strip().lower(): data for state, data in death_data.items()
    }

    # Gerekli alanları içeren eyaletleri filtrele
    filtered_data = {
        state: normalized_death_data[state]
        for state in normalized_death_data
        if (
            'Septicemia' in normalized_death_data[state] and
            'Influenza and pneumonia' in normalized_death_data[state] and
            'Death Rate per 1M' in normalized_death_data[state]
        )
    }

    if not filtered_data:
        return "❗ Uygun ölüm verisi bulunamadı. Lütfen verileri kontrol edin."

    # En güvenli eyaleti belirle (en düşük 1M başına ölüm oranı)
    safest_state = min(filtered_data.items(), key=lambda x: x[1]['Death Rate per 1M'])[0].title()

    # Metin olarak veri gösterimi
    death_info_str = "\n".join([ 
        f"{state.title()}:\n"
        f"  - Septicemia: {data['Septicemia']} ölüm\n"
        f"  - Influenza and pneumonia: {data['Influenza and pneumonia']} ölüm\n"
        f"  - 1M kişi başına ölüm oranı: {data['Death Rate per 1M']}"
        for state, data in filtered_data.items()
    ])

    # LLM'e gönderilecek prompt
    prompt = f"""
Aşağıda {date} tarihi için bazı ABD eyaletlerinde iki hastalık (Septicemia ve Influenza and pneumonia) nedeniyle tahmin edilen ölüm verileri yer alıyor.

Amacın, seyahat için **en güvenli eyaleti** belirlemek.

Ölüm verileri:

{death_info_str}

Verilere göre, **1 milyon kişi başına ölüm oranı** en düşük olan eyalet: **{safest_state}**.

Lütfen kullanıcıya şu şekilde açıklama yap:
- Her eyaletteki her iki hastalık için ölüm sayılarını ve genel ölüm oranını göster.
- **{safest_state}** eyaletini neden en güvenli seçenek olarak önerdiğini açıkla.
- **{safest_state}** eyaletini neden en güvenli seçenek olarak önerdiğini açıkla.
- Eğer {state.title()} değilse:
  - {state.title()} ile {safest_state} arasındaki benzerlikleri belirt (iklim, coğrafya, kültür gibi).
- Kullanıcıya sohbet eder gibi, sıcak ve kısa bir öneride bulun.
"""

    # LLM çağrısı
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        print("Karar LLM hatası:", e)
        return "⚠️ LLM üzerinden karar verilemedi."

try:
        # En güvenli eyaleti seç
        safest_state_info = choose_safest_state_via_llm(
            date=str(selected_date), 
            death_data_list=results_with_death_rate
        )

        # Sonucu kullanıcıya göster
        st.write(safest_state_info)

except Exception as e:
    st.write(f"Veri işlenirken hata oluştu: {str(e)}")
