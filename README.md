# 📊 Interactive Car Claims Dashboard

This interactive dashboard provides insights into car insurance claims data and allows users to model and predict claims frequency using statistical models. It is designed to be intuitive and interactive, helping actuaries and analysts better understand car claims data and assess client risk effectively.

# 📌 What This Dashboard Contains

1. **Data Visualisations**
   - Total number of car claims by **province**
   - Total number of car claims by **car colour**
   - Total number of car claims by **gender**
   - **Frequency plot** showing the number of clients with 1, 2, 3, and 4 claims

2. **Claims Frequency Modelling**
   - Users can **select input features** to train models:
     - **Zero-Inflated Poisson GLM**
     - **Poisson GLM**
   - These models are used to **predict claims frequency** based on selected features.

3. **Risk Prediction**
   - After training, users can **make new predictions** using either model.
   - Each prediction is classified into a **risk category** according to predicted claim frequency:
     - 🟢 **Very Low Risk Client** 
     - 🟡 **Low Risk Client**
     - 🟠 **Medium Risk Client**
     - 🔴 **High Risk Client**

# ⭐ Support the Project
If you find this dashboard useful, you can support the project by:
- ⭐ Starring the repository
- 📝 Providing feedback or feature suggestions
- 🔗 Sharing it with colleagues or on social media
