# Midwest Trading Competition 2018: Case 3  
 
## Case Overview  
 
The goal of this case is to practice predictive analytics & portfolio construction  
on simulated stock returns data. Given a historical series of features known to be  
predictive of stock returns, your task is to create buy/sell signals for stocks in  
order to form the highest sharpe portfolio.  

## Data 

See the following link for data files: 

https://drive.google.com/file/d/1KSJpJU_kyC7J03y9TsMVpZourGFS9eub/view?usp=sharing 


## Dependencies 

Install dependencies using "pip install -r requirements.txt" 

You are welcome to add other dependencies to the requirements.txt file in your submission. 

## Directory Overview  
 
The following important files are included in the directory:  

- portfolio.py : a portfolio interface that you are tasked with implementing (see  
                 docstring on file for more details)  
- sample_strategy.py : a sample implementation of the portfolio interface that would  
                       be an acceptable submission.  
- stock_data/ : a directory where you should add the provided "ticker_data.csv" and  
                "factor_data.csv" files.  

## Key Pointers/Hints  
  
### Basic  
 
- Prior to implementing a strategy, it is a good idea to do some research with the  
  raw data files, and search for any relationships that you can find. 
- When doing research, pay close attention to the industry that a stock is in.  
- Each stock, identified by the ticker column, has a different relationship to the   
  provided features. The best performing strategies will train models that adjust  
  parameters/weights based on the ticker that they are predicting returns for.   
- Many of the functions mapping features to stock returns are quite simple; be wary  
  of overfitting, because there is a lot of noise in the data.  
  
### Advanced  
 
- The provided features have a strong relationship with some tickers, and a weaker  
  relationship with others. It is a good idea to allocate more weightage to tickers  
  that have a stronger relationship to the features by returning larger signal values  
  for these tickers.  
  
- The covariance of assets in a portfolio has a strong impact on its sharpe ratio.  
  Assets with returns series that are uncorrelated form portfolios with higher  
  sharpe ratios than assets with returns series that are correlated. Keep this  
  in mind when deciding the signal strength to assign to each ticker. 
