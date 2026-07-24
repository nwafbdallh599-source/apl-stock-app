import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. ضبط إعدادات الصفحة لتناسب شاشات الجوال والكمبيوتر
st.set_page_config(
    page_title="توقع أسعار أسهم أبل",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. العنوان والرأس
st.title("📈 مشروع التنبؤ بأسعار أسهم أبل (AAPL)")
st.write("مرحباً بك! يمكنك استخدام هذا التطبيق للتحليل والتنبؤ بأسعار الأسهم بكل سهولة من الجوال أو الكمبيوتر.")

st.markdown("---")

# 3. تحميل وقراءة البيانات
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("aapl_data.csv")
        return data
    except Exception as e:
        return None

df = load_data()

# 4. عرض البيانات والرسم البياني
if df is not None:
    st.subheader("📊 نظرة عامة على البيانات")
    st.dataframe(df.tail(10), use_container_width=True)

    st.markdown("---")

    st.subheader("📉 رسم بياني للتوقعات")
    try:
        st.image("aapl_prediction_plot.png", caption="توقعات أسعار الأسهم", use_column_width=True)
    except Exception:
        st.info("يمكنك إرفاق الرسم البياني aapl_prediction_plot.png لعرضه هنا.")

else:
    st.warning("لم يتم العثور على ملف البيانات aapl_data.csv. يرجى التأكد من وجود الملف في نفس المجلد.")

st.markdown("---")
st.caption("تم تطوير هذا التطبيق كجزء من مشروع تعلم الآلة.")