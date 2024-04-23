# Credit-Card-Routing-for-Online-Purchase-via-Predictive-Modelling

# Case description:
Over the past year they have encountered a high failure rate of online credit card payments. The company loses a lot of money due to failed transactions and customers become increasingly unsatisfied with the 
online shop. Such online credit card payments are performed via so-called payment service providers, referred to as “PSPs” by the business stakeholders. Your company has contracts with four different PSPs and pays transaction fees for every single payment. The current routing logic is manual and rule-based. Business decision-makers, however, hope that with predictive modelling and with your help, a smarter way of routing a PSP to a transaction is possible. 

# Project Aim:
Help the business to automate the credit card routing via a predictive model. Such a model should increase the payment success rate by finding the best possible PSP for each transaction and at the same time keep the transaction fees low.

# Project Plan:
We have two primary objectives based on the project goal: improving payment success rates and minimizing transaction fees. The proposed project plan method entails developing two distinct mod-els to address each of these goals:
1) Two-model approach:
   
a) Model 1: Success prediction:
•	Predicts success probability while focusing primarily on increasing success rates.
•	The success probability from Model 1 is used as input for the second model.

b) Model 2: PSP selection:
•	Choosing the best payment service provider (PSP) based on success probabilities and trans-action fees.
•	Allows for dynamic decision-making by assessing success probabilities and costs.

3)	Implementation steps:

a)	Train Model 1:
•	Using transaction-related features (excluding fees) with success as the target variable.
•	Objective: Predict success probabilities.

b)	 Predict success probabilities:
•	Appling Model 1 to obtain success probabilities.

c)	Train Model 2:
•	Utilizing success probabilities from Model 1, transaction fees, and relevant features with PSP as the target variable.
•	Objective: Predicting the optimal PSP, considering success probabilities and fees.

d)	Routing decision:
•	Using success probabilities from Model 1 and Model 2 predictions to determine the best PSP.
•	Leverage fee comparison features in case of ties.

3)	Benefits:

a)	Specialization:
•	Model 1 optimizes success rates.
•	Model 2 minimizes transaction fees, considering success probabilities.

b)	Interpretability:
•	Separate models provide transparency for each decision-making aspect.




