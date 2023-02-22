#18/1/2023 Read Savannahs data, calculate derivative and output derivative with concentration

import pandas as pd

#----------------------------------------------------------------------------------------
def Derivatives_fun(filename):
    #Calculate differences between succesive rows
    Differences_Total=raw_data.diff()
    Differences_Concentrations=Differences_Total.iloc[:,2:]
    
    #Time differences between succesive concentrations
    Time_Differences=Differences_Total['Time(min)']

    #Numerical Derivative
    derivatives=Differences_Concentrations.div(Differences_Total['Time(min)'], axis=0)

    #Rename columns of derivatives
    original_col_names=derivatives.columns.values.tolist()
    new_col_names={column:'d_'+column for column in original_col_names}
    derivatives=derivatives.rename(columns=new_col_names)

    #Concatenated dataset
    full_dataset=pd.concat([filename,derivatives],axis=1)
    #Only derivatives
    derivatives=derivatives.drop([0])

    return full_dataset,derivatives
#----------------------------------------------------------------------------------------


#Read file
path='/home/sergio/work/Github/machine_scientist_ecology/data/microbial_growth/'
name='microbial_growth.csv'
raw_data=pd.read_csv(path+name)

#Get dataset with derivatives
Full_Dataset,Only_Derivatives_Dataset=Derivatives_fun(raw_data)

#Save to csv
Full_Dataset.to_csv(path+'microbial_growth_full.csv')
Only_Derivatives_Dataset.to_csv(path+'microbial_growth_derivatives.csv')



