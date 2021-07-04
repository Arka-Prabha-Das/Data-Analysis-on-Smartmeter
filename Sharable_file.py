#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import file

life = pd.read_csv("C:\\Users\\user\\Desktop\\Book1.csv")
life

#Getting sum of each column

z = np.array(life.columns)
arr = []
for el in z[1:]:
    arr.append(sum(life[el]))
    
arr    

#creating dummy data for dates

z = ['1-7-2018','2-7-2018','3-7-2018','4-7-2018','5-7-2018','6-7-2018','7-7-2018']
brr = pd.to_datetime(z)
brr


#converting data for dates and consumption (arr) to dataframe

y = list(zip(brr,arr))
df = pd.DataFrame(y,columns=['Date','Consumption'],index = ['','','','','','',''])
df


#changing default index of dataframe to date

df.set_index('Date',inplace=True)


#setting seaborn for plotting

sns.set(rc={'figure.figsize':(11, 4)})

#plotting the dataframe through seaborn

df['Consumption'].plot(linewidth=0.5);

#matplotlib presentation of same graph

plt.xlabel("Date")
plt.ylabel("Consumption")
plt.title("production graph")
plt.plot(df)

#k-plot

df.plot(style='k.')
plt.show()

#time-series multiplicative plot

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model='multiplicative',freq=1)
result.plot()
plt.show()

#day-wise consumption plot in matplotlib

y1 = life['Days']
y2 = life['Monday']
y3 = life['Tuesday']
y4 = life['Wednesday']

fig,ax1 = plt.subplots(constrained_layout = True)
ax2 = ax1.twinx()
ax3 = ax1.twinx()

curve1 = ax1.plot(y1,y2,color='r')
curve2 = ax2.plot(y1,y3,color='b')
curve3 = ax1.plot(y1,y3,color='g')
plt.plot()
plt.show()
