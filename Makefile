GNUPLOT=gnuplot

TEMP_DIR=temp
DOCS_DIR=docs

FASHION_MNIST_APP=cargo run --release --example fashion_mnist -- --quiet -t 4
FASHION_MNIST_PLOT=$(DOCS_DIR)/fashion_mnist.gnuplot
FASHION_MNIST_STATS=\
	$(TEMP_DIR)/fashion_mnist_stats_linear.csv \
	$(TEMP_DIR)/fashion_mnist_stats_single_layer.csv \
	$(TEMP_DIR)/fashion_mnist_stats_conv_net.csv \
	$(TEMP_DIR)/fashion_mnist_stats_conv_blur_net.csv
FASHION_MNIST_GRAPHS=$(FASHION_MNIST_STATS:$(TEMP_DIR)/%.csv=$(DOCS_DIR)/%.svg)

DIRS=$(TEMP_DIR) $(DOCS_DIR)

$(info $(shell mkdir -p $(DIRS)))

all: fashion_mnist
.PHONY: all clean fashion_mnist

clean:
	$(RM) $(FASHION_MNIST_STATS) $(FASHION_MNIST_GRAPHS)

fashion_mnist: $(FASHION_MNIST_GRAPHS)

$(DOCS_DIR)/fashion_mnist_%.svg : $(TEMP_DIR)/fashion_mnist_%.csv $(FASHION_MNIST_PLOT)
	$(GNUPLOT) -c $(FASHION_MNIST_PLOT) "$<" "$@"

$(TEMP_DIR)/fashion_mnist_stats_linear.csv:
	$(FASHION_MNIST_APP) --csv-file-name "$@" linear

$(TEMP_DIR)/fashion_mnist_stats_single_layer.csv:
	$(FASHION_MNIST_APP) --csv-file-name "$@" single-layer

$(TEMP_DIR)/fashion_mnist_stats_conv_net.csv:
	$(FASHION_MNIST_APP) --csv-file-name "$@" conv-net

$(TEMP_DIR)/fashion_mnist_stats_conv_blur_net.csv:
	$(FASHION_MNIST_APP) --csv-file-name "$@" conv-blur-net
