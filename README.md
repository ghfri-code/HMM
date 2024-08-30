# Stock Market Forecasting Using Hidden Markov Model

Forecasting stock price or financial markets has been one of the biggest challenges to the AI community. A Hidden Markov Model (HMM) is a finite state machine which has some fixed
number of states. It provides a probabilistic framework for modelling a time series of multivariate
observations. Hidden Markov models were introduced in the beginning of the 1970’s as a tool in
speech recognition. This model based on statistical methods has become increasingly popular in
the last several years due to its strong mathematical structure and theoretical basis for use in a wide
range of applications.
In related paper, made use of the well-established Hidden Markov Model (HMM) technique to
forecast stock price for some of the airlines. You should use given datasets (four Corporations)
instead of paper airline datasets and implement proposed method of paper on them. The dataset files (Corporation1-4) contain daily stock prices (almost 2500 working days) for 4
different corporations in order- Close, Open, High, Low. (Features = Close price, Open price, High
price, Low price) attention that, the data at the end of row datasets is related to the first days,
the first rows are related to the recent days . You should keep aside the recent 200 observations
for testing (the first 200 rows of each dataset) and used rest of the observations for training the
model. You should predict the prices for the past 200 days, starting from the 200th day (200 recent
days) separated for each feature for all datasets and then plot actual price (test set) and predicted
price (for each feature separated - horizontal axes=200 days, vertical axis = price) like figure2 in
paper. Report the MAPE for each dataset. ignore the ANN method from paper and only
implemented the proposed method in paper.
