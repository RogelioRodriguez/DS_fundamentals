import pandas as pd
data=pd.read_csv("craigslistVehicles.csv")
#full_data=pd.read_csv("craigslistVehiclesFull.csv")

# View Data
pd.set_option('display.max_columns', 100)  # The maximum amount of columns to display
data.head(3)
data.info()
data.shape
data.columns
data.describe()

# Understand missing values
data.isnull().any()
data.isnull().sum()/data.shape[0]  # Percent of null values
pd.notnull(data).sum()/data.shape[0]  # Percent that are full

# Explore variables
data.make  # or data['make']
data.make.unique()
data.city.unique().size
data.make.value_counts()  # counts each unique value # How to avoid cutting off output
data.type.value_counts()/data.type.notnull().sum()  # Percent of each vehicle type

########### GRAPHS ###########
data.year.plot(kind='hist')
data.year.hist()
data.year.hist(bins=50)  # Run last two simultaneously for overlaying graph
data.type.value_counts().plot(kind='bar')

########### MANIPULATION #########
dh=data.head(100)
# To change the column name
dh.rename(index=str,columns={'city':'new_city'}, inplace=True)

# Column Creation
dh['age'] = 2019 - dh['year']
dh['curr_year'] = 2019
dh['price_per_mile'] = data['price']/data['odometer']
# When setting values in a pandas object,
# care must be taken to avoid what is called chained indexing.
# dh.insert(2,'test_col',dh.type)

# Drop Column
dh_no_url=dh.drop('url',axis=1) # axis=1 means dropping columns, axis=0 means dropping rows

# Column Selection
dh[['url','city','price']]
dh.loc[33:66,["url","city","price"]]
dh.iloc[:,-1] #Last row

# ROW SELECTION/ FILTERING
data[0:100]
data[data.type=="SUV"]
data.loc[data.year==2000,['type','year']]
data.loc[data.type=="SUV",][['type','year','price','odometer']]

new_cars = dh[dh.age<=10]
new_cars_price = dh[(dh.age<=10) & (dh.price>=5000)]
d=dh.loc[dh.age<=10,:]
new_cars.equals(d) # shows that these are equal

# apply function
def double(x):
    return 2*x
dh.insert(dh.columns.get_loc('price')+1, 'price*2', dh.price.apply(double)) # to insert index after 'price'
dh.insert(dh.columns.get_loc('price*2')+1, 'price*3', dh.price.apply(lambda x: x*3))
dh.insert(list(dh.columns.values).index('price*3')+1, 'is_expensive', dh.price.apply(lambda x: 'expensive' if x>10000 else 'cheap'))
dh.is_expensive.value_counts()
dh["new_and_cheap"] = dh.apply(lambda x: 'yes' if x['price']<10000 and x['age']<10 else 'no', axis=1)

# Move a column to a specific index
cols = list(dh.columns.values) # Converts ndarry to list
dc = cols.pop(cols.index("new_and_cheap"))
cols.insert(cols.index('is_expensive')+1, dc)
dh=dh[cols]

# Binning - use to reduce the effects of minor observation errors
data['price_quantile'] = pd.qcut(data.price,5)
data['price_quantile'].value_counts()

pd.qcut(data.year,5).value_counts() # same number in each bin
pd.cut(data.year,5).value_counts() # equally spaced
year = pd.cut(data.year,[1899,1990,2000,2010,2020], labels=['up to 90','90 to 00','00 to 10','10 to 20'])
odometer = pd.qcut(data.odometer,4,labels=['low','medium','high','very high'])

# Each column replaced with new columns for every catergory with 0 or 1 indicating whether that row has it or not
dummie_vars = pd.get_dummies(data[['price','year','fuel','transmission','type']]).head(1000)

# as_index false: first two columns are what it's grouped by
data.groupby('type')[['price']].mean()
data.groupby('fuel')[['price','year','odometer']].mean()
data.groupby(['type','fuel'], as_index=False)[['price','year','odometer']].mean()
    #You may want to set as_index False when grouping by more than one

# Pivot tables - If two elements in groupby and only one column then pivot table is ideal
pd.pivot_table(data,values='price',index='type',columns='fuel',aggfunc='mean')
    #.sort(values), aggfunc can also be 'count', .plot()
pt = pd.pivot_table(data,'price',['type','fuel'],[year,odometer])
data.type.unique()
data.fuel.unique()
year.unique().categories
odometer.unique().categories
pt.loc[('SUV','electric'),('10 to 20','medium')]

# Merge df
d1 = data[['url','city']]
d2 = data[['url','year']]
df_joined = pd.merge(d1,d2,on='url')

# Append df
samp1 = data.sample(100,random_state=1)
samp2 = data.sample(100,random_state=2)
appDf = samp1.append(samp2)

############# DATA CLEANING ###################

# Remove duplicates
data.drop_duplicates(inplace=True) # inplace=True affects that df, does not return new one

# Delete columns with too many null values
cd1 = data.dropna(thresh=len(data)*.6,axis=1) # drops columns
# See what cols were deleted
set(data.columns)-set(cd.columns)
cd2 = data.dropna(thresh=21,axis=0) # drops rows
# ?Best way to clean data

# Fill Na
data.odometer.fillna(data.odometer.median()).isnull().any()

# convert text to all lower or upper for easier analysis
data.desc.head()
data.desc[[(type(i) is not str) for i in data.desc]] #ret elems that are not str
data.desc.apply(lambda x: x.lower() if type(x) is str else x) #lowers if str else leaves as is
# Or
data.desc.astype(str).apply(lambda x: x.lower()) #this changes null values to string, less ideal

# CHANGE ENTRIES TO INTEGER, CONVERT OTHER TO NaN, FILL NULL WITH MEAN
data.cylinders.value_counts() # does not include null values
data.cylinders.isnull().sum()
# to str, lowercase, del cylinder,
dc = data.cylinders.apply(lambda x: str(x).lower().replace('cylinders','').strip())
dc.value_counts() #includes nan as str
dc2 = pd.to_numeric(dc,errors='coerce')
dc2.value_counts() #converts the strings nan and other Nan
dc2.isnull().sum() #includes the nan and other
dc3 = dc2.fillna(dc2.mean()) # Is it okay to fill so many values with mean
dc3[dc2.isnull()].value_counts() #where the prev NaNs used to be is now mean

#VISUALIZATION
data.boxplot("odometer")
data.boxplot("price")
data.hist('odometer')
numeric = data._get_numeric_data()

#Import modules
from scipy import stats
import numpy as np
data_outliers = data[(data.price < data.price.quantile(.995)) & (data.price > data.price.quantile(.005))]
data_outliers.boxplot('price')
data_outliers.hist('price')

#Some type of models where scaling your data can improve speed and quality
import sklearn.preprocessing as sp
scalar = sp.MinMaxScaler()
# Will need to pass array of arrays to fit
# .values makes it ndarray, reshape makes it array of arrays, -1 means whatever number makes it fit
scalar.fit(dc3.values.reshape(-1,1))
ft_d = scalar.transform((dc3.values.reshape(-1,1)))

np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})
pd.set_option("display.precision", 4)
unique, counts = np.unique(ft_d, return_counts=True)
pd.DataFrame({"Unique":unique,"Counts":counts})
