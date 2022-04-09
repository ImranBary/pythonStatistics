#To load the data set USArrests as a pandas DataFrame (df)
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
sp="  "#variable for creating space 
df = sm.datasets.get_rdataset("USArrests").data
'''to select 10 rows (from the 2-12 row)
and the two columns Murder and Assault then store the
resultant data frame into df variable
'''
df=df[["Murder","Assault"]].iloc[2:12]
print (df)
print (sp)
import sys
print ("Estimated memory usage:",sys.getsizeof(df))
print (sp)
murders= df["Murder"]
print (murders)
print (sp)
assaults= df["Assault"]
print (assaults)
print (sp)
states= df.index
print (states)

fig=plt.figure(dpi=150)
fig.set_figheight(25)
fig.set_figwidth(25)
ax2=fig.add_subplot(2,2,1)
ax3=fig.add_subplot(2,2,2)
ax4=fig.add_subplot(2,2,3)
ax5=fig.add_subplot(2,2,4)

fig, ax1 = plt.subplots()
ax1.scatter(murders,assaults)
ax1.grid()
ax1.set_title("Graph of No. of Murders and Assaults in each State")
ax1.set_xlabel("No. of Murders (x)"),ax1.set_ylabel("No. of Assaults (y)")


for i, txt in enumerate(states):
    ax1.annotate(txt,(murders[i],assaults[i]))
    
ax2.bar(states,murders,color="green")#murders is the x varaible 
ax2.set_title("Bar chart for No. of Murders in given States")


ax3.boxplot(assaults,vert=False)#assaults is the y variable 
ax3.grid()
ax3.set_title("Box plot for No. of Assaults")
'''
X variable is Murders, indipenedent analysis
'''
print (1500*sp)
print (murders.describe())
q75, q25 = np.percentile(df['Murder'], [75 ,25])
iqr = q75 - q25
print ("interquartile range is:",iqr)

x=df["Murder"].min()
y=(df["Murder"].max()-x)/4
li=[str(x)+"<=X<"+str(x+y),str(x+y)+"<=X<"+str(x+2*y),str(x+2*y)+"<=X<"+str(x+3*y),str(x+3*y)+"<=X<"+str(x+4*y)]
freq=pd.DataFrame({"Ranges":li,"f":[0.,0.,0.,0.]})
for i in range (10):
    if df["Murder"].iloc[i]>x and df["Murder"].iloc[i]<x+y:
        freq["f"].iloc[0] +=1
    elif df["Murder"].iloc[i]>=x+y and df["Murder"].iloc[i]<x+2*y:
        freq["f"].iloc[1] +=1
    elif df["Murder"].iloc[i]>=x+2*y and df["Murder"].iloc[i]<x+3*y:
        freq["f"].iloc[2] +=1
    else:
        freq["f"].iloc[3] +=1
print (freq)
print (sp)
ranges= freq["Ranges"]

frequency= freq["f"]

print ("cumulative frequency:",50*sp,np.cumsum(frequency))
print (sp)



r_freq = pd.Series(frequency).value_counts()       
print("relative frequency:",50*sp,r_freq / len(frequency))
#still gotta do percentage relative frequency


patches, texts = ax4.pie(frequency,shadow=True,startangle=90)
ax4.legend(patches,ranges,loc="best")
ax4.axis("equal")
ax4.set_title("Pie chart for Murders Grouped Frequency distribution")

ax5.bar(ranges,frequency)
ax5.set_title("Bar graph for Murders Grouped Frequency distribution")

'''
Dependant variable Y analysis- Assaults
'''
print (1500*sp)
print (assaults.describe())
q75, q25 = np.percentile(df['Assault'], [75 ,25])
iqr = q75 - q25
print ("interquartile range is:",iqr)


'''
Both independent and dependant variables analysis
'''
fig2=plt.figure(dpi=150)
ax_1=fig2.add_subplot(1,1,1)
print (1500*sp)
print("Correlation between Murders and Assualts is:")
print (df.corr())
model = LinearRegression(fit_intercept=True)

model.fit(murders[:, np.newaxis], assaults)

xfit = np.linspace(0, 20, 1000)
yfit = model.predict(xfit[:, np.newaxis])

ax_1.scatter(murders, assaults)
ax_1.plot(xfit, yfit);
for i, txt in enumerate(states):
    ax_1.annotate(txt,(murders[i],assaults[i]))
ax_1.grid()



