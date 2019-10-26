StockAnalysisApp
================

A stock analysis app with streamlit.  
You select the ticker of the stock and the app makes a series of analysis by
using the price chart, returns, ratios, fundamental metrics, quantitative
metrics and outlier analysis.

 

Demo
----

 

![StockAnalysisApp Demo](demo/sample.gif)

 

**Requirements**

Python 3.6 version or superior

 

**How to run this demo**

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install streamlit 
pip install numpy pandas matplotlib scipy # if you don't use Anaconda
pip install FundamentalAnalysis
pip install yfinance
pip install Pillow # the successor of PIL, it's backwards compatible with PIL

git clone https://github.com/antonio-catalano/StockAnalysisApp.git
# cd into the project root folder
cd StockAnalysisApp
streamlit run app.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

**Run online**

<https://essential-stock-analysis.herokuapp.com/>
