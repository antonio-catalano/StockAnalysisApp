# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:36:59 2019

@author: Admin
"""

import FundamentalAnalysis as fa
import streamlit as st
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from PIL import Image


from matplotlib.pyplot import rc
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from dateutil.parser import parse
from scipy.stats import iqr
from datetime import timedelta, date



def set_pub():
    rc('font', weight='bold')    # bold fonts are easier to see
    rc('grid', c='0.5', ls='-', lw=0.5)
    rc('figure', figsize = (10,8))
    plt.style.use('bmh')
    rc('lines', linewidth=1.3, color='b')

@st.cache(suppress_st_warning=True)
def loadData(ticker, start, end): 
     df_stockdata = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )['Adj Close']   
     df_stockdata.index = pd.to_datetime(df_stockdata.index)
     return df_stockdata


@st.cache(suppress_st_warning=True)
def summary_stats(ticker):
    df_summary = fa.summary(ticker)
    return df_summary

@st.cache(suppress_st_warning=True)
def ratio_indicators(ticker):
    df_ratios = fa.ratios(ticker)
    return df_ratios

def get_data_yahoo(ticker, start, end):
    data = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )
    return st.dataframe(data)

        

def plotData(ticker, start, end):
    
    df_stockdata = loadData(ticker, start, end)
    df_stockdata.index = pd.to_datetime(df_stockdata.index)
    
    
    set_pub()
    fig, ax = plt.subplots(2,1)

    
    ma1_checkbox = st.checkbox('Moving Average 1')
    
    ma2_checkbox = st.checkbox('Moving Average 2')
    
    ax[0].set_title('Adj Close Price %s' % ticker, fontdict = {'fontsize' : 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values,'g-',linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].grid(True)
    
    if ma1_checkbox:
        days1 = st.slider('Business Days to roll MA1', 5, 120, 30)
        ma1 = df_stockdata.rolling(days1).mean()
        ax[0].plot(ma1, 'b-', label = 'MA %s days'%days1)
        ax[0].legend(loc = 'best')
    if ma2_checkbox:
        days2 = st.slider('Business Days to roll MA2', 5, 120, 30)
        ma2 = df_stockdata.rolling(days2).mean()
        ax[0].plot(ma2, color = 'magenta', label = 'MA %s days'%days2)
        ax[0].legend(loc = 'best')
    
    ax[1].set_title('Daily Total Returns %s' % ticker, fontdict = {'fontsize' : 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:],'r-')
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot()
    
def rolling_sharpe(y):
    return np.sqrt(252) * (y.mean() / y.std())


def plot_std_ret(ticker, start, end):
    def standard_ret(df):
        ret = df.pct_change()[1:]
        mean = ret.values.mean()
        std = ret.values.std()
        return (ret - mean) / std
    
    fig, ax = plt.subplots(figsize=(9,4))
    df_stockdata = loadData(ticker, start, end)
    ax.plot(df_stockdata.index[1:], 
          standard_ret(df_stockdata).values)
    ax.set_title('Standardized daily total returns %s'%ticker, fontdict = {'fontsize' : 15})
    ax.set_xlim(ax.get_xlim()[0] - 10, ax.get_xlim()[1] + 10)
    plt.grid(True)
    st.pyplot()


def plot_trailing(ticker, start, end):
    ret = loadData(ticker, start, end).pct_change()[1:]
    days = st.slider('Business Days to roll', 5, 120, 30)
    trailing_median = ret.rolling(days).median()
    trailing_max = ret.rolling(days).max()
    trailing_min = ret.rolling(days).min()
    trailing_iqr = ret.rolling(days).apply(iqr)
    q3_rolling = ret.rolling(days).apply(lambda x: np.percentile(x,75))
    q1_rolling = ret.rolling(days).apply(lambda x: np.percentile(x,25))
    soglia_upper = trailing_iqr*1.5 + q3_rolling
    soglia_lower = q1_rolling - trailing_iqr*1.5 
    trailing_all = pd.concat([trailing_median, trailing_max, trailing_min,trailing_iqr
                              ,soglia_upper, soglia_lower],
                             axis = 1)
    trailing_all.columns = ['Median', 'Max', 'Min','IQR','Q3 + 1.5IQR','Q1 - 1.5IQR']
    fig, ax = plt.subplots(figsize = (9,5))
    trailing_all.plot(ax = ax)
    ax.set_title('Rolling nonParametric Statistics (%s days)'%days
                     , pad = 30, fontdict = {'fontsize' : 17})
    ax.set_xlim(ax.get_xlim()[0] - 15, ax.get_xlim()[1] + 15)
    ax.legend(bbox_to_anchor=(0,0.96,0.96,0.2), loc="lower left",
                mode="expand", borderaxespad = 1, ncol = 6)
    ax.set_xlabel('')
    plt.grid(True)
    st.pyplot()
        
    ii = trailing_all.dropna().reset_index().drop('Date', axis = 1)
    st.subheader('Rolling nonParametric Statistics (%s days)'%days)
    print(st.dataframe(trailing_all.dropna()))
        
    st.subheader('Interactive chart, {} rolling observations,\
                     from {} to {}'.format(len(ii), parse(str(trailing_all.dropna().index[0])).date(),
                     parse(str(trailing_all.dropna().index[-1])).date()))
    st.line_chart(ii, width=800, height=120)
    
    ret = loadData(ticker, start, end).pct_change()[1:]
    trail_aim = trailing_all[['Q3 + 1.5IQR', 'Q1 - 1.5IQR']]
    
    def daterange(start_date, end_date):
        days_ = days
        for n in range(0, int ((end_date - start_date).days), days_):
            yield start_date + timedelta(n)
                
    def outliers():
        lista = []
        thresholds = []
        
        for i in range(len(ret)-days):
            ret_ = ret.iloc[i:days+i]
            trail_ = trail_aim.iloc[days+i]
            right_ret = np.where(ret_ > trail_['Q3 + 1.5IQR'], 1, 0).sum()
            left_ret = np.where(ret_ < trail_['Q1 - 1.5IQR'], 1, 0).sum()
            lista.append((right_ret, left_ret, (i,days+i)))
            trail_['Q1 - 1.5IQR'] = round(float(trail_['Q1 - 1.5IQR']),4)
            trail_['Q3 + 1.5IQR'] = round(float(trail_['Q3 + 1.5IQR']),4)
            thresholds.append((trail_['Q1 - 1.5IQR'], trail_['Q3 + 1.5IQR']))
        df = pd.DataFrame(np.random.randn(len(ret)-days,2))
        df.index = [elem[2] for elem in lista]
         
        df.columns = ['Left tail', 'Right tail']
        df['Right tail'] = [elem[0] for elem in lista]
        df['Left tail'] = [elem[1] for elem in lista]
        df['Thresholds'] = [t for t in thresholds]
        
        df2 = pd.DataFrame()
        df2['uno'] = [i[0] for i in df.index]
        df2['due'] = [i[1] for i in df.index]
        
      
        new = pd.DataFrame()
        new['A'] = [(str(ret.index[i].date()), str(ret.index[i+days].date())) for i in range(len(ret)-days)]
        new.index = new['A']
        
        df.index = new.index
        
       
        return df
    
    st.subheader('Number outliers for each datarange (%s business days)'%days)
    df_outliers = outliers()
    st.dataframe(df_outliers)
    st.subheader('Sorted by number of positive outliers (decreasing order)')
    st.dataframe(df_outliers.sort_values(by = 'Right tail', ascending = False))
    st.subheader('Sorted by number of negative outliers (decreasing order)')
    st.dataframe(df_outliers.sort_values(by = 'Left tail',ascending = False))
   
   
    
def rolling_sharpe_plot(ticker, start, end):
    data_ = loadData(ticker, start, end)
    ret = data_.pct_change()[1:]
    start_sp = data_.index[0].strftime('%Y-%m-%d')
    sp500 = pdr.get_data_yahoo('^SP500TR', start= start_sp, end = str(end) )
    sp500_ret = sp500['Close'].pct_change()[1:]
        
    days2 = st.slider('Business Days to roll', 5, 130, 20)
    rs_sp500 = sp500_ret.rolling(days2).apply(rolling_sharpe)
    rs = ret.rolling(days2).apply(rolling_sharpe)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(rs.index, rs.values, 'b-', label = 'Rolling Sharpe %s'%ticker)
    ax.plot(rs.index, rs_sp500, 'r-', label = 'Rolling Sharpe S&P500 (TR)')
    ax.set_title('Rolling Sharpe ratio (%s days, annualized)'%days2, fontdict = {'fontsize' : 15})
    ax.set_xlim(ax.get_xlim()[0] - 15, ax.get_xlim()[1] + 15)
    ax.legend(loc = 'best')
    plt.grid(True)
    st.pyplot()
    

''' # Adjusted close prices and total returns
   ### (stock prices from *yahoo finance*) '''

sp500_list = pd.read_csv('SP500_list.csv')

ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp500_list['Symbol'], index = 30).upper()
pivot_sector = True
checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \
                            Deselect to come back to the S&P 500 index stock list')
if checkbox_noSP:
    ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()
    pivot_sector = False


start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2018-01-01')
end = st.text_input('Enter the end date in yyyy-mm-dd format:', '2019-01-01')




try:
    start = parse(start).date()
    #print('The start date is valid')
    control_date1 = True
except ValueError:
    st.error('Invalid Start date')
    control_date1 = False
 
    
try:
    end = parse(end).date()
    #print('The end date is valid')
    control_date2 = True
except ValueError:
    st.error('Invalid End date')
    control_date2 = False

def check_dates():
    return control_date1 & control_date2


if start <= datetime(1970,1,1,0,0).date():
    st.error('Please insert a date posterior to 1st January 1970')
    pivot_date = False
else:
    pivot_date = True
    
if check_dates() and pivot_date == True:
    
        
    if len(loadData(ticker, start, end)) > 0: # if the ticker is invalid the function returns an empty series
        
     
        image = Image.open('imageforapp2.jpg')



        st.sidebar.image(image, caption='',

                 use_column_width=True)
        st.sidebar.header('A stock analysis app')
        st.sidebar.subheader('Choose the option to visualize')
        
        ticker_meta = yf.Ticker(ticker)
        
        series_info  = pd.Series(ticker_meta.info,index = reversed(list(ticker_meta.info.keys())))
        series_info = series_info.loc[['symbol', 'shortName', 'financialCurrency','exchange', 
                          'fullExchangeName', 'exchangeTimezoneName', 'marketCap', 'quoteType']]
        if pivot_sector:
            sector = sp500_list[sp500_list['Symbol'] == ticker]['Sector']
            sector = sector.values[0]
            series_info['sector'] = sector
        
           
        series_info.name = 'Stock'            
        st.dataframe(series_info)
        
        
        
        
        principal_graphs_checkbox = st.sidebar.checkbox('Stock prices and total returns', value = True)
        if principal_graphs_checkbox:
            plotData(ticker, start, end)
        
        std_ret_checkbox = st.sidebar.checkbox('Standardized daily total returns') 
        if std_ret_checkbox:
            plot_std_ret(ticker, start, end)
        
        trailing_checkbox = st.sidebar.checkbox('Outlier analysis')
        if trailing_checkbox:  
            ''' ## Outlier analysis '''
            
            st.markdown('''In order to conduct an outlier analysis we use nonparametrics statistics in a rolling mode. 
            In financial markets the mean and standard deviation are deceptive, because the returns distribution
            is not well behaved. The typical distribution of stock returns is skewed and leptokurtic, so 
            i prefer to use the median as location metric and the interquartile range as dispersion metric.
            Interquartile range (**IQR**) is defined as *Q3 - Q1*, where *Q3* is the third quartile and *Q1* the first quartile.
            Each return observation above the upper threshold *Q3 + 1.5IQR* can be considered an outlier in the right 
            tail of the distribution, and each observation below the lower threshold *Q1 - 1.5IQR* can be considered
            an outlier in the left tail of the distribution.''')
            
            st.subheader('Interquartile range : Q3 - Q1')
            st.subheader('Upper threshold : Q3 + 1.5IQR')
            st.subheader('Lower threshold : Q1 - 1.5IQR')
            st.write('')
            plot_trailing(ticker, start, end)
                     
        fundamental_checkbox = st.sidebar.checkbox('Fundamental Analysis')
        if fundamental_checkbox:
            ''' ## Fundamental analysis '''
            st.title('Summary')
            st.dataframe(summary_stats(ticker))
        
            st.title('Ratios and indicators')
            st.dataframe(ratio_indicators(ticker))
            
        rs_checkbox = st.sidebar.checkbox('Rolling Sharpe ratio vs Rolling Sharpe ratio S&P500, (annualized)')
        if rs_checkbox: 
            ''' # Rolling Sharpe Ratio '''
            ''' We compare the rolling sharpe ratio (RSR) of the stock with the rolling sharpe ratio of S&P500 (TR).
            We calculate the RSR by fixing the risk free rate equal to 0.
            Hence *RSR = rolling_returns_mean / rolling_returns_std*.
            ''' 
            rolling_sharpe_plot(ticker, start, end)
    
        historical_prices_checkbox = st.sidebar.checkbox('Historical prices and volumes')
        if historical_prices_checkbox:
            st.title('Historical prices and volumes')
            get_data_yahoo(ticker, start, end)
        
        
    else:
        st.error('Invalid ticker')
    

    




