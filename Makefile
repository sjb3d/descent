GNUPLOT=gnuplot

TEMP_DIR=temp
DOCS_DIR=docs

FASHION_MNIST_APP=cargo run --release --example fashion_mnist --
FASHION_MNIST_PLOT=$(DOCS_DIR)/fashion_mnist.gnuplot
FASHION_MNIST_STATS=\
	$(TEMP_DIR)/fashion_mnist_stats_linear.csv \
	$(TEMP_DIR)/fashion_mnist_stats_single-layer.csv \
	$(TEMP_DIR)/fashion_mnist_stats_conv-net.csv \
	$(TEMP_DIR)/fashion_mnist_stats_conv-blur-net.csv
FASHION_MNIST_GRAPHS=$(FASHION_MNIST_STATS:$(TEMP_DIR)/%.csv=$(DOCS_DIR)/%.svg)

IMAGE_FIT_APP=cargo run --release --example image_fit --
IMAGE_FIT_PLOT=$(DOCS_DIR)/image_fit.gnuplot
IMAGE_FIT_STATS=\
	$(TEMP_DIR)/image_fit_stats_relu.csv \
	$(TEMP_DIR)/image_fit_stats_relu-pe.csv \
	$(TEMP_DIR)/image_fit_stats_siren.csv
IMAGE_FIT_GRAPHS=$(DOCS_DIR)/image_fit_stats.svg

DIRS=$(TEMP_DIR) $(DOCS_DIR)

$(info $(shell mkdir -p $(DIRS)))

all: fashion_mnist image_fit
.PHONY: all clean clean_fashion_mnist fashion_mnist clean_image_fit image_fit

clean: clean_fashion_mnist clean_image_fit

clean_fashion_mnist:
	$(RM) $(FASHION_MNIST_STATS) $(FASHION_MNIST_GRAPHS)

fashion_mnist: $(FASHION_MNIST_GRAPHS)

clean_image_fit:
	$(RM) $(IMAGE_FIT_STATS) $(IMAGE_FIT_GRAPHS)

image_fit: $(IMAGE_FIT_GRAPHS)

$(DOCS_DIR)/fashion_mnist_%.svg : $(TEMP_DIR)/fashion_mnist_%.csv $(FASHION_MNIST_PLOT)
	$(GNUPLOT) -c $(FASHION_MNIST_PLOT) "$@" "$<"

$(TEMP_DIR)/fashion_mnist_stats_%.csv:
	$(FASHION_MNIST_APP) --quiet -t 4 --csv-file-name "$@" $*

$(TEMP_DIR)/image_fit_stats_%.csv:
	$(IMAGE_FIT_APP) --quiet --csv-file-name "$@" --image-prefix "$(DOCS_DIR)/image_fit_output_$*" $*

$(DOCS_DIR)/image_fit_stats.svg: $(IMAGE_FIT_STATS) $(IMAGE_FIT_PLOT)
	$(GNUPLOT) -c $(IMAGE_FIT_PLOT) "$@" $^
