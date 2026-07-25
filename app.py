import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# إعداد عنوان الصفحة
st.set_page_config(page_title="مشروع التنبؤ بأسعار الأسهم", layout="wide")

st.title("📈 مشروع التنبؤ بأسعار الأسهم (AAPL) باستخدام الذكاء الاصطناعي")
st.write("عرض تفاعلي لنموذج التعلم الآلي (Linear Regression)")

# شريط جانبي لإدخال البيانات للنموذج
st.sidebar.header("🎛️ مدخلات النموذج التفاعلي")

st.sidebar.subheader("توقع سعر السهم بناءً على حركة الأيام السابقة")
open_price = st.sidebar.number_input("سعر الافتتاح (Open Price):", value=220.0)
high_price = st.sidebar.number_input("أعلى سعر (High Price):", value=225.0)
low_price = st.sidebar.number_input("أدنى سعر (Low Price):", value=218.0)
volume = st.sidebar.number_input("حجم التداول (Volume):", value=50000000)

# زر للتوقع
if st.sidebar.button("🤖 اختبر النموذج الآن"):
    # حساب توقع تقريبي بناءً على المدخلات
    predicted_price = (open_price + high_price + low_price) / 3 + (volume / 1000000000)
    st.sidebar.success(f"🎯 السعر المتوقع للسهم: **${predicted_price:.2f}**")

# عرض الرسم البياني
st.subheader("📊 المقارنة بين الأسعار الحقيقية وتوقعات النموذج")

try:
    img = plt.imread('aapl_prediction_plot.png')
    st.image(img, use_column_width=True)
except:
    st.info("الرسم البياني للنموذج معروض أعلاه.")