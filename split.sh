#!/bin/bash

n_train=43100
n_validation=9236

count=0
while read -r line
do
	image=${line%%\ *}
	line=${line#*\ }
	income=${line%%\ *}
	line=${line#*\ }
	poverty=${line%%\ *}
	line=${line#*\ }
	crime=${line%%\ *}
	line=${line#*\ }
	income_cat=${line%%\ *}
	line=${line#*\ }
	poverty_cat=${line%%\ *}
	line=${line#*\ }
	crime_cat=${line%%\ *}

	income_cat=$((income_cat-1))
	poverty_cat=$((poverty_cat-1))
	crime_cat=$((crime_cat-1))

	
	
	if ((count<n_train))
	then 		
		echo $image >> "train_image.txt"	
		echo $income_cat $poverty_cat $crime_cat >> "train_value.txt"
	elif ((count<n_train+n_validation))
	then
		echo $image >> "validate_image.txt"	
		echo $income_cat $poverty_cat $crime_cat >> "validate_value.txt"
	else
		echo $image $income_cat $poverty_cat $crime_cat >> "classify.txt"		
	fi
	count=$((count+1))
done < "data_shuf.txt"
