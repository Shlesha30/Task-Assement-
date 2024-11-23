import pandas as pd
import sqlite3

sales_data = pd.read_excel("sales_data.xlsx")

conn = sqlite3.connect("inventory_db.sqlite")
products = pd.read_sql_query("SELECT * FROM products", conn)
inventory = pd.read_sql_query("SELECT * FROM inventory", conn) 

customer_behavior = pd.read_csv("customer_behavior.csv")

sales_products = sales_data.merge(products, on="ProductCategory")


sales_inventory = sales_products.merge(inventory, on=["ProductID", "Region"])

final_data = sales_inventory.merge(customer_behavior, on="Region")

from sklearn.impute import KNNImputer
imputer = KNNImputer()
numerical_columns = final_data.select_dtypes(include=["float64", "int64"]).columns
final_data[numerical_columns] = imputer.fit_transform(final_data[numerical_columns])

sales_data['DiscountImpact'] = sales_data['Revenue'] * sales_data['DiscountApplied'] / 100
revenue_loss = sales_data.groupby('Region')['DiscountImpact'].sum()

from statsmodels.tsa.statespace.sarimax import SARIMAX

inventory['Date'] = pd.to_datetime(inventory['Date'])
inventory = inventory.set_index('Date')
model = SARIMAX(inventory['StockLevel'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
inventory['OptimalReorderPoint'] = inventory['StockLevel'] - inventory['LeadTime'] * inventory['ReorderPoint']
supplier_analysis = products.groupby('Supplier').agg({'Price': 'mean', 'ProductionCost': 'mean'})
from sklearn.model_selection import train_test_split
from sklearn.ensemble import XGBRegressor

X = sales_data[['Region', 'ProductCategory', 'UnitsSold', 'DiscountApplied', 'AverageSpend']]
y = sales_data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBRegressor()
model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

X = customer_behavior[['SatisfactionScore', 'PurchaseFrequency', 'ReferralCount']]
y = customer_behavior['ChurnRisk']

model = RandomForestClassifier()
model.fit(X, y)

import plotly.express as px
import dash
from dash import dcc, html

fig = px.line(sales_data, x="Date", y="Revenue", color="Region")


app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
report_data = {
    "Region": revenue_loss.index,
    "Revenue Loss": revenue_loss.values,
    "Inventory Bottlenecks": forecast,
    "Churn Rates": customer_behavior['ChurnRisk'].value_counts()
}

report = pd.DataFrame(report_data)
report.to_excel("report.xlsx", index=False)

import schedule
import time

def fetch_and_update():
    
    sales_data = pd.read_excel("sales_data.xlsx")
    customer_behavior = pd.read_csv("customer_behavior.csv")
   
    model.fit(X_train, y_train)report.to_excel("report.xlsx", index=False)

schedule.every().day.at("00:00").do(fetch_and_update)

while True:
    schedule.run_pending()
    time.sleep(1)


