#!/usr/bin/env bash

for ((d=$3, c=$1; c < $2; c++))
do
	gnuplot -c draw_step.p result$d-$c.txt result_step$d-$c.png
	gnuplot -c draw_time.p result$d-$c.txt result_time$d-$c.png
done
mv *.png png/
