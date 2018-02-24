#/bin/python
'''
This script is going to transform data downloaded from UCI repo into structrured
dataframe.
Take care of how the original data break for new lines.
Should look carefully how many lines assemble into one complete row.
'''
import pickle
import pandas as pd
import os

def assemble(raw_data, output_path, attr_explanation='name.txt'):
    name_list = []
    result_data = []
    complete_row = []
    with open(attr_explanation, 'r') as file:
        for line in file:
            name_list.append(line.split()[0])
    with open(raw_data, 'r') as file:
        for line in file:
            temp_list = [float(value) for value in line.split()]
            complete_row.extend(temp_list)
            # here is the number of value in the second line of the complete row of data
            if len(temp_list) == 3:
                result_data.append(complete_row)
                complete_row = []
    df = pd.DataFrame(result_data, columns=name_list)
    with open(output_path + '/' + os.path.basename(raw_data) + '.pickle', 'wb') as file:
        pickle.dump(df, file)

if __name__=='__main__':
    assemble('housing_dataset', 'dataset')
