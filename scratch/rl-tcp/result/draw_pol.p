set terminal png size 1000,800
set output ARG2
set title 'policy_step'
plot ARG1 using 1:2 axis x1y1 with l title "1", ARG1 using 1:3 axis x1y2 with l title "2", ARG1 using 1:4 axis x1y2 with l title "3", ARG1 using 1:5 axis x1y2 with l title "4", ARG1 using 1:6 axis x1y2 with l title "5"

