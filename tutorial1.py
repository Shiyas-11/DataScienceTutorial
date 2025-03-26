#imports
import numpy as np
import pandas as pd
import csv

file_name = "./Advertising.csv"

#initialise numpy arrays
tv=np.array([])
radio=np.array([])
newsp=np.array([])
sales=np.array([])

# open the csv file and process the information to the arrays

with open(file_name, mode='r') as file:
    csv_reader = csv.reader(file)    
    header = next(csv_reader)
    for row in csv_reader:
        tv=np.append(tv,float(row[1]))
        radio=np.append(radio,float(row[2]))
        newsp=np.append(newsp,float(row[3]))
        sales=np.append(sales,float(row[4]))

length=len(tv)
assert len(tv)==len(radio)==len(newsp)==len(sales)


def function(x):
    x_=np.mean(x)
    sales_=np.mean(sales)

    beta1=(np.sum((x-x_)*(sales-sales_)))/ (np.sum((x-x_)**2))
    beta1=round(beta1,5)


    beta0=sales_-beta1*x_
    beta0=round(beta0,5)

    # print(x_,sales_,beta1,beta0)

    sales_pred=beta0+x*beta1

    RSS=np.sum((sales-sales_pred)**2)

    RSE=np.sqrt((1/(length-2))*RSS)

    TSS=np.sum((sales-sales_)**2)

    R2=(TSS-RSS)/TSS
    p=1
    FS=((TSS-RSS)/p)/(RSS/(length-p-1))

    print("RSS",RSS,"\nRSE",RSE,"\nTSS",TSS,"\nR2",R2,"\nFS",FS)

if __name__=="__main__":
    function(tv)
    function(radio)
    function(newsp)



