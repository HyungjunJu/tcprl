#!/usr/bin/env bash

for ((c=$1; c < $2; c++))
do
	gnuplot -c draw_pol.p policy$c.txt result_policy$c.png
done
mv *.png png/
