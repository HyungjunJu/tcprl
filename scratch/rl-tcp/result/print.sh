#!/usr/bin/env bash

for ((c=$1; c < $2; c++))
do
	gnuplot -c draw_step.p result$c.txt result_step$c.png
	gnuplot -c draw_time.p result$c.txt result_time$c.png
done
mv *.png png/
