Base code
---

Data generating
---
I tried to write a generic data generator. It takes csv file as it input, loading into memory as pandas.dataframe, and process the dummy variable. It can future be specifed the indices to be dropped out. The detailed explanation and pipeline are the following: 

* How to do meta processing?     
As in this case, data samples are essentially pairs. So the job meta-processing of it is to discover matched indices and unmatched indices.

    cd ./meta
    ./get_uniq.sh
    python get_pair_indices.py

What is worth noting is that, get_uniq.sh adopts mundane tool `awk` to process the csv file, which is rather fast! (fuck pandas lol)
get_pair_indices.py finds the matching indices based on the processing done by `get_uniq.sh`       
*Important*: the file path may needs manually modification.

* How to get data batches?        
Infected with Stochastic optimization means, from the `base/data.py`, one interface is provided as the generation of mini-sample batches.
For the fact that majority of elements are zeros and the sake of fast manipulations, returned values are not suprisingly, `scipy.sparse.csr_matrix`.

* TODO
scikit learn can`t be used in this way.
