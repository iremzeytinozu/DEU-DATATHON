import streamlit as st
from langchain.chat_models import ChatOpenAI
from ast import literal_eval
import os
from dotenv import load_dotenv
import datetime
import re

# Ortam değişkenlerinden API key al
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM modelini başlat (gpt-4.1-nano)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)

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


def choose_safest_state_via_llm(origin_state: str, date: str, death_data: dict, similar_states: list) -> str:
    filtered_death_data = {state: death_data[state] for state in similar_states if state in death_data}
    
    prompt = f"""
Aşağıda {date} tarihi için bazı ABD eyaletlerinde tahmin edilen ölüm sayıları verilmiştir.
Lütfen {origin_state} eyaletini de dahil ederek listedeki tüm eyaletleri karşılaştır.

Amacın, seyahat için en güvenli eyaleti seçmektir.

Ölüm verileri (eyalet: ölüm sayısı): {filtered_death_data}

Kurallar:
- Tahmini ölüm sayısı en düşük olan eyaleti seç.
- Eğer {origin_state} en güvenliyse, onu öner ve nedenlerini söyle.
- Eğer başka bir eyalet daha güvenliyse, onu öner ve kullanıcıya şu şekilde açıkla:
  - Neden bu eyalet daha güvenli?
  - Hangi yönlerden {origin_state} eyaletine benziyor? (iklim, coğrafya, kültür gibi)
  - Kullanıcıya dostça, sohbet eder gibi açıkla. Kısa ama sıcak bir öneri yap.

Yalnızca açıklayıcı bir metin döndür.
"""
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        print("Karar LLM hatası:", e)
        return "Karar verilemedi."

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

# -------------------- UI --------------------

st.set_page_config(page_title="Turizm Chatbot", page_icon="🧳", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧭 Turizm Yardımcı Chatbot</h1>", unsafe_allow_html=True)
st.markdown("#### 👋 Hoş geldiniz! Seyahatiniz için en benzer ve en güvenli eyaletleri birlikte keşfedelim.")

st.markdown("---")

# Kullanıcı girişleri
states_list = ["Texas", "Arizona", "Nevada", "California", "Florida", "New York", "Michigan", "Georgia", "Washington", "Colorado"]

col1, col2 = st.columns(2)
with col1:
    user_state = st.selectbox("Gideceğiniz Eyaleti Seçin", options=states_list)
with col2:
    date = st.date_input("Seyahat Tarihinizi Seçin")

st.markdown("---")

# Benzer eyaletleri bul ve en güvenli eyaleti aynı anda öner
if user_state and date:
    if st.button("🔍 Benzer Eyaletleri ve En Güvenli Eyaleti Bul"):
        # Benzer eyaletleri bul
        similar_states = get_similar_states_via_llm(user_state, states_list, str(date))
        st.success(f"**{user_state}** eyaletine iklim, kültür ve coğrafya açısından en çok benzeyen 3 eyalet:")
        st.write(similar_states)
        
        # Ölüm verileri (örnek)
        death_data = {
            "Texas": 10,
            "Arizona": 4,
            "Nevada": 5,
            "California": 12,
            "Florida": 8,
            "New York": 6,
            "Michigan": 3,
            "Georgia": 7,
            "Washington": 2,
            "Colorado": 4
        }

        # En güvenli eyalet önerisini bul
        safest_state_info = choose_safest_state_via_llm(user_state, str(date), death_data, similar_states)
        st.markdown("### 🏖️ En Güvenli Eyalet Önerisi")
        st.info(safest_state_info)

else:
    st.warning("Lütfen bir eyalet seçin ve tarih girin.")
