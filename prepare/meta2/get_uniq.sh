#!/bin/bash

awk '{split ($0,a,","); print a['"$1"']}' $2 | uniq  | grep -v '^\-1'
