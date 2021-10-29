#!/bin/bash
if [ "$1" == "1" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ] && [ "$5" != "" ]; then
	python3 decision_trees.py $2 $3 $4 $5
elif [ "$1" == "2" ] && [ "$2"!="" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python3 neural_networks.py $2 $3 $4
else
	echo "Supply appropriate and correct arguments"
	exit
fi
