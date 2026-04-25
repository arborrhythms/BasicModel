ifeq ($(OS),Windows_NT)
SHELL := C:/msys64/usr/bin/bash.exe
else
SHELL := /bin/bash
endif
ifeq ($(OS),Windows_NT)
VENV_BIN_DIR := Scripts
VENV_PYTHON  := .venv/$(VENV_BIN_DIR)/python.exe
VENV_PIP     := .venv/$(VENV_BIN_DIR)/pip.exe
VENV_ARGS    := --system-site-packages
PYTHON_BOOTSTRAP ?= py -3.12
else
VENV_BIN_DIR := bin
VENV_PYTHON  := .venv/$(VENV_BIN_DIR)/python
VENV_PIP     := .venv/$(VENV_BIN_DIR)/pip
VENV_ARGS    :=
PYTHON_BOOTSTRAP ?= python3
endif

VENV_PYTHON_FROM_BIN := ../$(VENV_PYTHON)
VENV_STAMP := .venv/.installed-$(VENV_BIN_DIR)

# --- PDF generation options (inlined from Make.mk) ----------------------------
PD_TEMPLATE := /bits/projects/custom_template.tex
PDFOPTS := --pdf-engine=xelatex \
          -V geometry:margin=1in \
          --template=$(PD_TEMPLATE) \
          -V header-includes="\usepackage{amsmath} \usepackage{amssymb} \usepackage{unicode-math} \hyphenpenalty=10000 \exhyphenpenalty=10000 \makeatletter \renewcommand\section{\@startsection{section}{1}{\z@}{-3.5ex}{2.3ex}{\normalfont\Large\bfseries\centering}} \makeatother"

# Ordered list of doc chapters for PDF generation
PDF_CHAPTERS := README.md  doc/Installation.md doc/Architecture.md doc/BasicModel.md doc/Spaces.md doc/Language.md doc/Mereology.md doc/Logic.md doc/Reasoning.md doc/Training.md doc/Ergodic.md doc/MachineMinds.md doc/Params.md
XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml
MODEL ?= data/BasicModel.xml
PYTHON := PYTHONPATH=bin $(VENV_PYTHON)

# SSH defaults for ArborMini.local
TRAIN_HOST ?=
TRAIN_USER ?= arogers
TRAIN_KEY  ?= ~/.ssh/id_ed25519_arbormini
TRAIN_DIR  ?= ~/WikiOracle/basicmodel

# The shared MAKE_PDF macro drops later options due to a broken line
# continuation, so override it locally with the reader extensions these docs use.
MAKE_PDF = pandoc $(PDFOPTS) \
		--from=gfm+smart+bracketed_spans+attributes \
		--metadata title="$(TITLE)" \
		--toc --toc-depth=3 \
		--resource-path=.:doc

.PHONY : all install xor tomatoes ergodic simple run compare test test_all bench doc clean \
         train train_micro train_remote train_micro_remote

all : xor

$(VENV_STAMP): requirements.txt
	rm -rf .venv
	$(PYTHON_BOOTSTRAP) -m venv $(VENV_ARGS) .venv
	"$(VENV_PYTHON)" -m pip install --upgrade pip setuptools wheel
	"$(VENV_PYTHON)" -m pip install -r requirements.txt
	@touch "$@"

install : $(VENV_STAMP)


run : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) Models.py $(XML1)

train : $(VENV_STAMP)
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log \
		$(if $(TRAIN_HOST),--host $(TRAIN_HOST) --user $(TRAIN_USER) --key-file $(TRAIN_KEY) --remote-dir $(TRAIN_DIR))

train_micro : $(VENV_STAMP)
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log \
		--max-docs 500 --num-shards 1 --num-epochs 1 --random-shards \
		$(if $(TRAIN_HOST),--host $(TRAIN_HOST) --user $(TRAIN_USER) --key-file $(TRAIN_KEY) --remote-dir $(TRAIN_DIR))

train_remote :
	$(MAKE) train TRAIN_HOST=arbormini.local

train_micro_remote :
	$(MAKE) train_micro TRAIN_HOST=arbormini.local

xor : data/MM_xor.xml
	$(MAKE) run XML1=$<

tomatoes : data/tomatoes.xml
	$(MAKE) run XML1=$<

ergodic : data/ergodic.xml
	$(MAKE) run XML1=$<

simple : data/simple.xml
	$(MAKE) run XML1=$<

mnist : data/mnist.xml
	$(MAKE) run XML1=$<

SigmaPi : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) SigmaPi.py

SymPercept : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) SymPercept.py

SPNN : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) SPNN.py


compare : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) BasicModel.py --compare $(XML1) $(XML2)

test : $(VENV_STAMP)
	BASICMODEL_DEVICE=cpu PYTHONPATH=bin $(VENV_PYTHON) test/test_report.py

test_all : $(VENV_STAMP)
	BASICMODEL_DEVICE=cpu PYTHONPATH=bin $(VENV_PYTHON) test/test_report.py

bench : $(VENV_STAMP)
	@echo "=== Baseline (no env tweaks) ==="
	PYTHONPATH=bin $(VENV_PYTHON) test/bench_training.py

clean :
	rm -f BasicModel.pdf
	rm -rf output/*

doc : BasicModel.pdf


TITLE := Basic Model
BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^
