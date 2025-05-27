import pandas as pd

year = '2019'  # please specify the year, our experiments use 2019
ca_his = pd.read_hdf('ca_his_raw_' + year + '.h5')

### please comment this line if you don't want to do resampling
ca_his = ca_his.resample('5T').mean().round(0)
###

ca_his = ca_his.fillna(0)
print('check null value number', ca_his.isnull().any().sum())

# only take first 4 months
ca_his = ca_his.iloc[:288*30*4]

ca_his.to_hdf('ca_his_' + year + '.h5', key='t', mode='w')