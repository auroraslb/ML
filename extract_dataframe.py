'''
Extract data from numpy-dataset.
Write to txt file for visualization.
'''
import numpy as np

df_x = np.load('Xtrain_Regression_Part1.npy')
df_y = np.load('Ytrain_Regression_Part1.npy')

def write_to_file(df_x, df_y):
    #write to file:
    f = open("data_as_txt.txt", "w")
    f.write('X-values:\n')
    for row in df_x:
        np.savetxt(f, row)
        f.write('\n')

    f.write('\n\nY-values:\n')
    for row in df_y:
        np.savetxt(f, row)
        f.write('\n')

    f.close()

write_to_file(df_x, df_y)

