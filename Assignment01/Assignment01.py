import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_original = pd.read_csv('landslide_data_original.csv')
df_missing = pd.read_csv('landslide_data_miss.csv')


def statistics(n,attribute,type):
    # n = df_original['temperature'].to_numpy()
    sum = max = total = 0
    min = n[0]
    for i in n:
        total+=1
        sum+=i
        if i>max:
            max = i
        if i < min:
            min = i
    mean = sum/total
    std = (( np.sum ((mean - n) * (mean - n)) ) ** 0.5) / total
    print(f"The statistical measures of {attribute} attribute in {type} data are: mean= {mean:.2f}, maximum= {max:.2f}, minimum= {min:.2f}, STD= {std:.2f}")


#Q1A
n = df_original['temperature'].to_numpy()
sum = max = total = 0
min = n[0]
for i in n:
    total+=1
    sum+=i
    if i>max:
        max = i
    if i < min:
        min = i
mean = sum/total
std = (( np.sum ((mean - n) * (mean - n)) ) ** 0.5) / total
print(f"The statistical measures of Temperature attribute are: mean= {mean:.2f}, maximum= {max:.2f}, minimum= {min:.2f}, STD= {std:.2f}")


#Q1B
def pearsonCorrelation(x,y):
    meanX = np.mean(x)
    meanY = np.mean(y)
    return np.sum((x - meanX) * (y - meanY)) / np.sqrt(np.sum((x - meanX)**2) * np.sum((y - meanY)**2))

#Initializing the correlation matrix
attributes = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
correlation_matrix = pd.DataFrame(columns=attributes, index=attributes)

for y_att in attributes:
        for x_att in attributes:
            x = df_original[x_att].to_numpy()
            y = df_original[y_att].to_numpy()
            correlation_matrix.loc[x_att, y_att] = pearsonCorrelation(x, y)
    
print(correlation_matrix)
l_avg = correlation_matrix['lightavg'].to_numpy()
# To find values that lie between 0.6 and 1
redundant_attributes = [att for att, corr in correlation_matrix.iterrows() if 0.6 <= corr['lightavg'] < 1]
print("Redundant Attributes wrt lightavg: ", redundant_attributes)

#Q1C
t12Humidity = df_original[df_original['stationid'] == 't12']['humidity'].to_numpy()
minHumidity = np.min(t12Humidity)
# print(minHumidity)
maxHumidity = np.max(t12Humidity)
bin_size = 5
bins = np.arange(minHumidity, maxHumidity+bin_size, bin_size)

# print(bins)
heights = np.zeros_like(bins)
for h in t12Humidity:
    heights[int((h - minHumidity) // bin_size)] += 1
# print(heights)

plt.bar(bins+bin_size/2, heights, width=bin_size, edgecolor='yellow')
plt.xticks(bins)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Histogram of humidity for station t12')
plt.show()

#----------------------------------------------------------------------------------------


# Q2A
df_missing.dropna(subset=['stationid'], inplace=True) #drop rows with missing station_id
df_missing.dropna(subset=attributes, thresh=int(2*len(attributes)/3), inplace=True) #drop rows with missing values in more than 1/3 of the attributes'


#Q2B
for j in attributes:
    values = df_missing[j].values
    x = len(values)
    idx = -1
    for i in range(x):
        if np.isnan(values[i]):
            idx = i
            count = 0
            while np.isnan(values[i]):
                count+=1
                i+=1
            ans = (abs ( values[idx-1] - values[i] ) ) / (count + 1)
            for k in range(count):
                if k == 0:
                    if values[idx-1] > values[i]:
                        values[idx] = values[i] + ans
                    else:
                        values[idx] = values[i] - ans
                else:
                    values[idx+k] = values [idx+k-1] + ans
    df_missing[j] = values


for i in attributes:
    n1 = df_original[i].to_numpy()
    n2 = df_missing[i].to_numpy()
    statistics(n1,i,'original')
    statistics(n2,i,'missing')
    print()

#Q2C
rmse_list = []

for j in attributes:
    val_original = df_original[['dates',j]].to_numpy()
    val_missing = df_missing[['dates', j]].to_numpy()
    rmse = 0
    for i in range(len(val_original)):
        for j in range(len(val_missing)):
            if val_original[i][0] == val_missing[j][0]:
                rmse += (val_original[i][1] - val_missing[j][1])**2
    rmse = np.sqrt(rmse/len(val_missing))
    rmse_list.append(rmse)
print(rmse_list)

plt.bar(attributes, rmse_list)
plt.show()

#------------------------------------------------------------------------------------------------------

#Q3
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))  # Adjust figsize for better size
axes = axes.flatten()
for i, j in enumerate(attributes):
    axes[i].boxplot(df_missing[j].dropna())  # Plot boxplot for each attribute
    axes[i].set_xlabel(j)  # Set x-label
    axes[i].set_ylabel('Values')  # Set y-label
    axes[i].set_title(f'Boxplot of {j}')  # Set title
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()


for j in attributes:
    val = list(df_missing[j].to_numpy())
    length = len(val)
    temp = val.copy()
    temp.sort()
    # print(temp)
    median = temp[length//2]
    Q1 = temp[length//4]
    Q3 = temp[3*length//4]
    lower_lim = Q1 - 1.5*(Q3-Q1)
    upper_lim = Q3 + 1.5*(Q3-Q1)
    for i in range(len(val)):
        if val[i] < lower_lim or val[i] > upper_lim:
            val[i] = median
    df_missing[j] = val
        
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
axes = axes.flatten()
for i, j in enumerate(attributes):
    axes[i].boxplot(df_missing[j].dropna()) 
    axes[i].set_xlabel(j)
    axes[i].set_ylabel('Values')
    axes[i].set_title(f'Boxplot of {j}')
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------

#Q4
for j in attributes:
    val = df_missing[j].values
    minima = np.min(val)
    maxima = np.max(val)
    val = (val - minima) / (maxima - minima) * 7 + 5
    df_missing[j] = val
    statistics(val,j,'normalized')
