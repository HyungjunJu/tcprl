set terminal png size 1000,800
set output ARG2
set title 'result-time'
set y2tics
set ytics nomirror
plot ARG1 using 2:3 axis x1y1 with l title "cwnd", ARG1 using 2:4 axis x1y2 with l title "Rtt"

