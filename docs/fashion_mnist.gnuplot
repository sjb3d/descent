set key above
set xlabel "Epoch"
set ylabel "Loss"
set y2label "Accuracy"
set ytics nomirror
set y2tics
set xtics nomirror
set tics out
set y2range [0.7 : 1.0]
set output ARG2
set terminal svg enhanced background rgb 'white'
plot ARG1 using 1:2 with lines axes x1y1 title "training loss",\
     ARG1 using 1:3 with lines axes x1y1 title "test loss",\
     ARG1 using 1:4 with lines axes x1y2 title "training accuracy",\
     ARG1 using 1:5 with lines axes x1y2 title "test accuracy"
