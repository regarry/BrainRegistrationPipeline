import pandas as pd
import numpy as np
from scipy.io import savemat

# Create some 2D DataFrames
#df1 = pd.DataFrame(np.arange(0, 4).reshape(-1,2))
#df2 = pd.DataFrame(np.arange(4, 10).reshape(-1,2))
#df3 = pd.DataFrame(np.arange(10, 18).reshape(-1,2))
df1 = np.arange(0, 4).reshape(-1,2)
df2 = np.arange(4, 10).reshape(-1,2)
df3 = np.arange(10, 18).reshape(-1,2)
# Combine the DataFrames into a list
dfs = []
dfs.append(df1)
dfs.append(df2)
dfs.append(df3)

# Now dfs is a list of 2D DataFrames, which is similar to a 3D array
print(dfs)
print(dfs[0][0])
arr = np.array(dfs,dtype=object)
np.save('arr.npy', arr)
print(arr[1][0,0])
savemat('my_array.mat', {'my_array': arr})
loaded_arr = np.load('arr.npy',allow_pickle=True)

print(loaded_arr)
print(loaded_arr[1][0,0])
