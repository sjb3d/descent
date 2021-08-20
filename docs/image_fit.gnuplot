set xlabel "Epoch"
set ylabel "Loss"
set tics out
set output ARG1
set terminal svg enhanced background rgb 'white'
set logscale y
plot ARG2 with lines title "ReLU",\
     ARG3 with lines title "ReLU with Position Encoding",\
     ARG4 with lines title "SIREN"
