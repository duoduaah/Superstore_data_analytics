from operator import mul
from turtle import width
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as po
import plotly.graph_objs as pg
import plotly.graph_objects as go
import streamlit as st
from millify import millify


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import month_plot, quarter_plot     
from pmdarima import auto_arima   



# Set webpage to wide
st.set_page_config(layout="wide")

# Read csv file 
df = pd.read_csv("Sample_Superstore.csv", encoding='latin1')
data = df.copy()



# ______________________________________________ HELPER FUNCTIONS __________________________________________________________
# Function to get sum of a numeric column
@st.cache
def get_sum(df, column_name):
    return df[column_name].sum()


@st.cache   
def unique_column_items_stats(data, groupby_col, sortby_col ):
    
    """ returns data grouped by groupby_col and aggregate of Quantity,
    Discount, Sales and Profit column and sorted by sortby_col in descending order"""
    
    data_grouped = data.groupby(groupby_col, as_index=False).agg(
                 Quantity = ("Quantity", np.sum),
                Discount=("Discount", np.sum),
                Sales=("Sales", np.sum),
                Profit=("Profit", np.sum)).sort_values(sortby_col, ascending=True)
    
    return data_grouped


# Function to group-by, aggregate and return the percentage change between rows in another column
@st.cache
def get_pct_change(data, groupby_col, agg_col):

    # order_name = name of column to group by
    # agg_name = name of column to aggregate
    
    data_grouped = data.groupby(by=groupby_col)[agg_col]
    data_grouped = pd.DataFrame(data_grouped.sum()).reset_index()
    data_grouped["Pct_"+agg_col] = data_grouped[agg_col].pct_change().fillna(0)*100
    return data_grouped





# ____________________________________________ DATA PREPROCESSING AND FEATURE ENGINEERING ______________________________________
# Drop Row ID column
data = data.drop(['Row_ID'], axis=1)

# Set date columns as datetime
date_cols = ['Order_Date', 'Ship_Date']
data[date_cols] = data[date_cols].apply(pd.to_datetime, format="%m/%d/%Y")

# Cast postal code to object
data['Postal_Code'] = data['Postal_Code'].astype('object')

# Sort data by 'Order Date'
data = data.sort_values(by='Order_Date', ignore_index=True)

# Extract day and month from data
data['Order_Day'] = data['Order_Date'].dt.day_name()
data['Order_Month'] = data['Order_Date'].dt.month_name()
data['Order_Year'] = data['Order_Date'].dt.year
data = data.set_index('Order_Date')



# ____________________________________________ KEY INDICATORS ________________________________________________________________
unique_products = data["Product_ID"].nunique()
unique_customers = data["Customer_ID"].nunique()
total_sales = get_sum(data, "Sales")
total_products_sold = get_sum(data, "Quantity")
total_discounts = get_sum(data, "Discount")
net_profit = get_sum(data,"Profit")



# ________________________________________________ Group sales per month _______________________________________________________
months_df = data[["Sales"]].resample('M').sum()


# Cumulative monthly sales over all 4 years plot
months_fig = px.line(months_df, x=months_df.index, y="Sales", title="Monthly sales over the 4-year period", 
                    color_discrete_sequence=['darkblue'])
months_fig.update_traces(hovertemplate=None)
months_fig.update_xaxes(title_text='Month and Year')
months_fig.update_yaxes(title_text='Sales ($)')


# Monthly plot per year 
year_fig = px.line(months_df, x = months_df.index.month, y="Sales", color=months_df.index.year,
             labels = dict(x = "Month", y='Sales ($)', color="Year"), title = "Comparison of the monthly sales for 2014, 2015, 2016, and 2017", 
             markers=True, color_discrete_sequence=['Midnight blue', 'maroon','pink', 'cadetblue'])
year_fig.update_layout( hovermode='x unified',
                        xaxis = dict(
                            tickmode = 'array',
                            tickvals = np.arange(1,13),
                            ticktext = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 
                                                'September', 'October', 'November', 'December']
                                    ))



# ________________________________________________ FORECASTING ______________________________________________________
arima = SARIMAX(months_df, order=(2,1,0), seasonal_order=(1,0,0,12), enforce_invertibility=False).fit(disp=0)
predictions = pd.DataFrame(arima.predict(start = len(months_df), end=len(months_df)+11, typ='level', dynamic=False))

# Plot
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=months_df.index, y=months_df["Sales"], mode='lines', name='Original sales'))
forecast_fig.add_trace(go.Scatter(x= predictions.index, y = predictions['predicted_mean'], mode='lines',
                        name='Predicted sales'))
forecast_fig.update_xaxes(title_text='Month and Year')
forecast_fig.update_yaxes(title_text='Sales ($)')
forecast_fig.update_layout(title = 'Previous sales per month for 2014 - 2017 (in blue) and forecasted sales (in red) for 2018)',
autosize=False, width=1200, height=600)


# ________________________________________________  Overall monthly and quarter plots _____________________________________
# Month plot
month_plot_fig = month_plot(months_df)
month_plot_fig.tight_layout()
month_plot_fig.set_figheight(2)
month_plot_fig.set_figwidth(5)

# Quarter plot
quarter_df = data[["Sales"]].resample('Q').sum()
quarter_plot_fig = quarter_plot(quarter_df)
quarter_plot_fig.tight_layout()
quarter_plot_fig.set_figheight(2)
quarter_plot_fig.set_figwidth(5)



# ________________________________________________ GEO-PLOT ______________________________________________________

state_codes = {
    'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME'}

data['State_code'] = data['State'].apply(lambda x : state_codes[x])

geo_grouped = data.groupby(['State_code'])['Sales']
geo_grouped = pd.DataFrame(geo_grouped.sum()).reset_index()

geo_data = dict(type='choropleth', locations=geo_grouped["State_code"], locationmode="USA-states",
                z=geo_grouped['Sales'], colorscale="pubu", colorbar={'title': 'Total Sales'})

geo_layout = dict(title="4-year geographical sales distribution of USA", width = 1000, height=600,
                    geo=dict(scope='usa', showlakes=True, lakecolor = 'rgb(0, 200, 250)'))

geo_fig = pg.Figure(data=geo_data, layout=geo_layout)



# ________________________________________________ PIE - PLOT ______________________________________________________
#pie_fig = px.pie(data, values="Sales", names="Product Sub-Category", 
#            title="Sales by product category for the 4-year period", color_discrete_sequence=px.colors.sequential.Brwnyl)


# ______________________________ PERCENT - CHANGE ----------------

pct_sales = get_pct_change(data, "Order_Year", "Sales")
pct_profit = get_pct_change(data, "Order_Year", "Profit")
pct_quantity = get_pct_change(data, "Order_Year", "Quantity")
pct_discount = get_pct_change(data, "Order_Year", "Discount")

pct_change = pd.DataFrame(pct_sales["Order_Year"])
pct_change["Sales"] = pct_sales["Pct_Sales"]
pct_change["Profit"] = pct_profit["Pct_Profit"]
pct_change["Quantity"] = pct_quantity["Pct_Quantity"]
pct_change["Discount"] = pct_discount["Pct_Discount"]



with st.sidebar:
    add_radio = st.radio("Type", ('General', '2014', '2015', '2016', '2017'))



if add_radio == "General":

    st.title("GENERAL OVERVIEW OF THE SALES, PROFITS AND OTHER METRICS USING A 4-YEAR PERIOD; 2014 - 2017 DATA FROM A SUPERSTORE")
    st.text(" ") 
    st.caption('Note: Use the sidebar for individual year analysis.')
    
    st.subheader("Overall values of the performance metrics.")
    
    # Key indicators 
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    
    with col5:
        st.metric(label="Unique Products", value=unique_products)

    with col6:
        st.metric(label="Number of orders", value=millify(total_products_sold, precision=2))
        
    with col4:
        st.metric(label="Active Customers", value=unique_customers)

    with col1:
        st.metric(label="Total Revenue", value="$ "+millify(total_sales, precision=2))

    with col3:
        st.metric(label="Total Discount", value="$ "+millify(total_discounts, precision=2))

    with col2:
        st.metric(label="Net Profit", value="$ "+millify(net_profit, precision=2))


    st.write(" ")
    st.write(" ")
    st.write(" ")
    # Figures
    # Single plot
    st.subheader("1. Visualization of the trends and the comparison of total sales over the 4-year period. ")
    
    st.write("We visualize the trends in sales over the years for some categories. The first plot below shows the monthly sales from \
        January 2014 to December 2017, where we see a seasonal characteristic in the sales. The second graph to the right gives a comparison of the monthly sales\
            for the years. We can identify the year with the highest or lowest sale for a particular month using this plot. For example, 2016 \
                records the highest sales for December.")

    

    

    col7, col8 = st.columns([1,1])
    

    with col7:
        st.plotly_chart(months_fig)

    with col8:
        st.plotly_chart(year_fig)


    # Monthly and Quaterly plots
    st.subheader('2. Seasonal monthly and quarterly plots')
    st.write("In addition, we can check the seasonal plot of monthly (left) and quarterly (right) sales shown below:")

    colmonth, colquart = st.columns([1,1])
    with colmonth:
        st.pyplot(month_plot_fig)
    
    with colquart:
        st.pyplot(quarter_plot_fig)
    
    st.write(' ')
    st.write(' ')
    st.subheader('3. Predicting into the future: Forecasting 2018 sales')
    st.write("Given data from previous 4 years, we can predict the sales for the next 12 months (i.e year 2018) using \
    AutoRegressive Moving Average (SARIMA) statistical model.")
    
    # Forecast plot
    col9, col10, col11 = st.columns([1, 10, 1])
    with col10:
        st.plotly_chart(forecast_fig)



    # _________________ NEXT SECTION _________________
    # GEO PLOT
    st.subheader('4. Sales distribution over US regions')
    #st.caption("Geographical plot")
    #with st.expander("See details"):
        
    st.write("We are interested in the regions of the United States which generates more revenue and the regions with the least revenue generated. The \
        geographic plot below gives us the answers. Clearly, California, New York and Texas generates a large percentage of the total revenue.")
    col12, col13, col14 = st.columns([1, 7, 1])
    with col13:
        st.plotly_chart(geo_fig)


    # Various plots depending on the indicator chosen
    st.header("5. Further Analysis and Plots")
    st.write("The graphs below show the influence of the various categories of the store's products on the total revenue generated, profit margin and \
        the number of units sold over the 4-year period")
    st.write('Select the performance indicator of choice and to view the corresponding plots')
    

    indicator = st.selectbox("Performance Indicators", ("Profit Margin", "Sales", "Quantity = Units sold"))
    categories = ["Category", "Sub_Category", "Order_Day", "Order_Month", "Region", "Ship_Mode"] 

    col15, col16 = st.columns([1.15, 1])
    col17, col18 = st.columns([1.15, 1])
    col19, col20 = st.columns([1.15, 1])
    cols = [col15, col16, col17, col18, col19, col20]

    if indicator == 'Profit Margin':

        i = 0
        for cat in categories:
            fig = px.histogram(data, x=cat, y="Profit", title=cat.upper(), color="Order_Year", color_discrete_sequence=px.colors.sequential.tempo)
            with cols[i]:
                st.plotly_chart(fig)
            
            i += 1

    if indicator == 'Sales':

        i = 0
        for cat in categories:
            fig = fig = px.histogram(data, x=cat, y="Sales", title=cat.upper(), color="Order_Year", color_discrete_sequence=px.colors.sequential.Brwnyl)
            with cols[i]:
                st.plotly_chart(fig)
            
            i += 1

    if indicator == 'Quantity = Units sold':

        i = 0
        for cat in categories:
            fig = fig = px.histogram(data, x=cat, y="Quantity", title=cat.upper(), color="Order_Year", color_discrete_sequence=px.colors.sequential.Teal)
            with cols[i]:
                st.plotly_chart(fig)
            
            i += 1


    # __________ LAST section with top products and top customers

    st.subheader('6. Top-10 Products and Top-10 Customers')
    st.write("Click the buttons below to reveal the top-10 products and the top-10 customers for the 4-year period ")
    col21, col22 = st.columns(2)

    with col21:
        if st.button("Top-10 Products"):
            top_10_products = unique_column_items_stats(data, "Product_ID", "Quantity").tail(10)
        
            fig_top_products = px.bar(top_10_products, x="Quantity", y="Product_ID", orientation='h')
            st.plotly_chart(fig_top_products)

    with col22:
        if st.button("Top-10 Customers"):
            top_10_customers = unique_column_items_stats(data, "Customer_Name", "Quantity").tail(10)
        
            fig_top_customers = px.bar(top_10_customers, x="Quantity", y="Customer_Name", orientation='h')
            st.plotly_chart(fig_top_customers)


    
    


# _______________________________________________________________________ PER YEAR ANALYSIS ____________________________________________________
years = [2014, 2015, 2016, 2017]
data2014 = data.query('Order_Year == 2014')
data2015 = data.query('Order_Year == 2015')
data2016 = data.query('Order_Year == 2016')
data2017 = data.query('Order_Year == 2017')

years_data_dict = {"2014":data2014, "2015":data2015, "2016":data2016, "2017":data2017}


# ______ Grouped data per year aggregated by sum ___________
data_year_grouped = data.groupby("Order_Year").agg(
                 Quantity = ("Quantity", np.sum),
                Discount =("Discount", np.sum),
                Sales=("Sales", np.sum),
                Profit=("Profit", np.sum)).sort_values("Quantity", ascending=False)



# _____________ Key indicators and their percent change from previous year __________________
# Percent_change
pct_change_years = pct_change.set_index("Order_Year")

# Unique customers
unique_2014 = data2014["Customer_ID"].nunique()
unique_2015 = data2015["Customer_ID"].nunique()
unique_2016 = data2016["Customer_ID"].nunique()
unique_2017 = data2017["Customer_ID"].nunique()



for year in years:

    presentkey = str(year)
    temp = list(years_data_dict)
    try:
        if year == 2014:
            previousyear = str(2014)
        else:
            previousyear = temp[temp.index(presentkey) - 1]
    except (ValueError, IndexError):
        previousyear = str(2014)


    if add_radio == str(year):

        st.title("Key metrics and the '%' change with respect to previous year.")
        st.text(" ") 

        col1, col2, col3, col4, col5 = st.columns(5)  

        with col1:
            st.metric(label="Number of orders", value=millify(data_year_grouped.loc[year]["Quantity"], precision=2), 
            delta="{:.2f}%".format(pct_change_years.loc[year]["Quantity"]))
            
        with col2:
            st.metric(label="Total Sales", value="$ "+millify(data_year_grouped.loc[year]["Sales"], precision=2),
            delta="{:.2f}%".format(pct_change_years.loc[year]["Sales"]))

        with col3:
            st.metric(label="Net Profit", value="$ "+millify(data_year_grouped.loc[year]["Profit"], precision=2),
            delta="{:.2f}%".format(pct_change_years.loc[year]["Profit"]))

        with col4:
            st.metric(label="Total Discount", value="$ "+millify(data_year_grouped.loc[year]["Discount"], precision=2),
            delta="{:.2f}%".format(pct_change_years.loc[year]["Discount"]))

        with col5:

            prev = years_data_dict[str(previousyear)]["Customer_ID"].nunique()
            current = years_data_dict[str(year)]["Customer_ID"].nunique()
            prct_change = ((current - prev) / prev) * 100

            st.metric(label="Active Customers", value=years_data_dict[str(year)]["Customer_ID"].nunique(), 
            delta="{:.2f}%".format(prct_change))


        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.subheader('1. Monthly sales and the top US regions with the most products sold.')
        col5, col6 = st.columns(2)

        # Monthly sales
        data_y = months_df[months_df.index.year == year]
        fig = px.line(data_y, x=data_y.index.month, markers=True, y=data_y['Sales'])
        fig.update_layout( title='Monthly sales over the year', hovermode='x unified', xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.arange(1,13),
                        ticktext = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 
                                            'September', 'October', 'November', 'December']
                                ))
 
        with col5:
            st.plotly_chart(fig)

        # Top 10 products
        #top_10 = unique_column_items_stats(years_data_dict[str(year)], "Product ID", "Quantity").tail(10)
        top_10_cities = unique_column_items_stats(years_data_dict[str(year)], "City", "Quantity").tail(10)
        #fig_top = px.bar(top_10_cities, x="Quantity", y="Product ID", orientation='h')
        fig_top_cities = px.bar(top_10_cities, x="Quantity", y="City", orientation='h')
        with col6:
            st.plotly_chart(fig_top_cities)


        st.write("")
        st.write("")
        st.header("2. Further Analysis and Plots")
        st.write("The graphs below show the influence of the various categories of the store's products on the total revenue generated, profit margin and \
        the number of units sold over the 4-year period.")
        st.write('Select the performance indicator of choice and to view the corresponding plots.')

        indicator = st.selectbox("Performance Indicators", ("Profit Margin", "Sales", "Quantity = Units sold"))
        categories = ["Category", "Sub_Category", "Order_Day", "Order_Month", "Region", "Ship_Mode"] 

        col8, col9 = st.columns([1.15, 1])
        col10, col11 = st.columns([1.15, 1])
        col12, col13 = st.columns([1.15, 1])
        cols = [col8, col9, col10, col11, col12, col13]

        

        if indicator == 'Profit Margin':

            i = 0
            for cat in categories:
                fig = px.pie(years_data_dict[str(year)], names=cat, values="Profit", title=cat.upper(), color_discrete_sequence=px.colors.sequential.Teal)
                with cols[i]:
                    st.plotly_chart(fig)
                
                i += 1

        if indicator == 'Sales':

            i = 0
            for cat in categories:
                fig = px.pie(years_data_dict[str(year)], names=cat, values="Sales", title = cat.upper(), color_discrete_sequence=px.colors.sequential.Brwnyl)
                with cols[i]:
                    st.plotly_chart(fig)
                
                i += 1

        if indicator == 'Quantity = Units sold':

            i = 0
            for cat in categories:
                fig = px.pie(years_data_dict[str(year)], names=cat, values="Quantity", title=cat.upper(), color_discrete_sequence=px.colors.sequential.tempo)
                with cols[i]:
                    st.plotly_chart(fig)
                
                i += 1

        st.write("")
        st.write("")
        st.write("")
        st.subheader('3. Top-10 Products and Top-10 Customers')
        st.write("Click the buttons below to reveal the top-10 products and the top-10 customers for the 4-year period ")
        col14, col15 = st.columns(2)

        with col14:
            if st.button("Top-10 Products"):
                top_10_products = unique_column_items_stats(years_data_dict[str(year)], "Product_ID", "Quantity").tail(10)
            
                fig_top_products = px.bar(top_10_products, x="Quantity", y="Product_ID", orientation='h')
                st.plotly_chart(fig_top_products)

        with col15:
            if st.button("Top-10 Customers"):
                top_10_customers = unique_column_items_stats(years_data_dict[str(year)], "Customer_Name", "Quantity").tail(10)
            
                fig_top_customers = px.bar(top_10_customers, x="Quantity", y="Customer_Name", orientation='h')
                st.plotly_chart(fig_top_customers)
