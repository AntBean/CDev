#!/bin/bash

# dev_train_basic.csv
idx="1 2 3 4 5 7 8"
for i in `echo $idx`
do
    awk '{split ($0,a,","); print a['"$i"']}' ../../DataSample/dev_train_basic.csv | uniq 
    # Next, pipe and seirielize to a python list, passing to the next stage python handle
done
