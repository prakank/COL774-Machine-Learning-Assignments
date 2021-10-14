#!/bin/bash
if [ "$1" == "1" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python3 naive_bayes.py $2 $3 $4
elif [ "$1" == "2" ] && [ "$2"!="" ] && [ "$3" != "" ] && [ "$4" != "" ] && [ "$5" != "" ]; then
	python3 svm.py $2 $3 $4 $5
else
	echo "Supply appropriate and correct arguments"
	exit
fi
