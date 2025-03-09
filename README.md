# 📌 Credit Card Routing for Online Purchase via Predictive Modelling

## 📖 Overview

In the fast-paced world of online retail, efficient credit card transaction processing is critical for a seamless customer experience and financial success. This project aims to develop an **automated credit card routing system** using **predictive modelling** to:

✅ **Increase payment success rates 💳**

✅ **Reduce transaction costs 💰**

✅ **Optimize PSP selection 🤖**

### 🖼️ CRISP-DM Process Model:

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to ensure a structured and effective approach.

![CRISP-DM Reference Model.png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/CRISP-DM%20Reference%20Model.png)

## 🛠️ Project Structure

**📂 Data** - Contains historical transaction data for training models.

**📂 Code** - Python scripts for data preprocessing, model training, and evaluation.

**📂 Reports** - Detailed analysis, performance metrics, and findings.

**📂 Images** - Visual representations of model results and insights.


## 🎯 Objectives

**📌 Automate Credit Card Routing** - Replacing manual rule-based routing with AI-driven decisions.

**📌 Increase Payment Success Rate** - Predict transaction success likelihood.

**📌 Minimize Transaction Fees** - Optimize PSP selection to reduce costs.

**📌 Ensure Model Interpretability** - Provide transparent decision-making insights.


## 🏗️ Methodology

The project employs a two-model strategy:

**🔹 Model 1: Success Prediction**

🔸 Predicts whether a transaction will succeed.

🔸 Uses features like amount, country, 3D security, and PSP.

🔸 Output: Probability of transaction success.



**🔹 Model 2: PSP Selection**

🔸 Determines the best payment service provider.

🔸 Considers success probability (from Model 1) and transaction fees.

🔸 Output: Optimized PSP selection.



## 🔢 Data Understanding

The dataset includes:

**✅ Transaction timestamp ⏳**

**✅ Country 🌍**

**✅ Transaction amount 💵**

**✅ Success status ✅❌**

**✅ PSP (Payment Service Provider) 🏦**

**✅ 3D Secure authentication 🔐**

**✅ Card type (Visa, Master, Diners) 💳**


### 📊 Correlation Analysis:

![Correlation analysis.png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/Correlation%20analysis.png)

## ⚙️ Modeling & Feature Importance

The models were trained using multiple ML techniques:

**✅ K-Nearest Neighbor (KNN)**

**✅ Logistic Regression**

**✅ Support Vector Machines (SVM)**

**✅ Random Forest Classification** _(Selected Model)___

**📌 Feature Importance for Model 1 (Random Forest Classification) :**

![Features Importance in Model 1.png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/Features%20Importance%20in%20Model%201.png)

**📌 Feature Importance for Model 2 (Random Forest Classification):**

![Features Importance in Model 2.png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/Features%20Importance%20in%20Model%202.png)


## 📊 Model Performance

**📌 Performance of Model 1 (Random Forest Classification):**

![Model 1 performance output.png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/Model%201%20performance%20output.png)

**📌 Performance of Model 2 (Random Forest Classification):**

![Model 2 performance output .png](https://github.com/Kaushik-Puttaswamy/Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling/blob/dev/Images/Model%202%20performance%20output%20.png)

## 🚀 Deployment & Business Impact

**🔹 Aligns with business goals** - Reduces failed transactions & customer dissatisfaction.

**🔹 Enhances automation** - Reduces reliance on manual routing.

**🔹 Data-driven decisions** - Increases financial efficiency.

**🔹 Stakeholder-friendly** - Provides interpretable insights.

## 📌 Conclusion

This project successfully demonstrates how machine learning can optimize credit card routing in online retail, leading to higher success rates and lower transaction fees. By leveraging predictive modeling, businesses can enhance financial efficiency and customer satisfaction.

## 💡 Future Work:

🚀 Implement real-time transaction monitoring.

📊 Explore deep learning techniques for improved predictions.

📈 Fine-tune cost-benefit analysis for PSP selection.

#### 🔗 Developed by: [Kaushik Puttaswamy](https://www.linkedin.com/in/kaushik-puttaswamy-data-analyst/) – Case Study: Model Engineering
