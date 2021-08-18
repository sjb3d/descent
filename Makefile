GNUPLOT=gnuplot
RESULTS_DIR=results

MNIST_APP=cargo run --release --example mnist -- --quiet -t 4
MNIST_PLOT=$(RESULTS_DIR)/mnist.gnuplot
MNIST_STATS=\
	$(RESULTS_DIR)/mnist_linear.csv \
	$(RESULTS_DIR)/mnist_single_layer.csv \
	$(RESULTS_DIR)/mnist_conv_net.csv \
	$(RESULTS_DIR)/mnist_conv_blur_net.csv
MNIST_GRAPHS=$(MNIST_STATS:%.csv=%.svg)

DIRS=$(RESULTS_DIR)

$(info $(shell mkdir -p $(DIRS)))

all: mnist
.PHONY: all clean mnist

clean:
	$(RM) $(MNIST_STATS) $(MNIST_GRAPHS)

mnist: $(MNIST_GRAPHS)

$(RESULTS_DIR)/mnist_%.svg : $(RESULTS_DIR)/mnist_%.csv $(MNIST_PLOT)
	$(GNUPLOT) -c $(MNIST_PLOT) "$<" "$@"

$(RESULTS_DIR)/mnist_linear.csv:
	$(MNIST_APP) --csv-file-name "$@" linear

$(RESULTS_DIR)/mnist_single_layer.csv:
	$(MNIST_APP) --csv-file-name "$@" single-layer

$(RESULTS_DIR)/mnist_conv_net.csv:
	$(MNIST_APP) --csv-file-name "$@" conv-net

$(RESULTS_DIR)/mnist_conv_blur_net.csv:
	$(MNIST_APP) --csv-file-name "$@" conv-blur-net
