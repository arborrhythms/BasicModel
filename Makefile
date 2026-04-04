SHELL := /bin/bash

# --- PDF generation options (inlined from Make.mk) ----------------------------
PD_TEMPLATE := /bits/projects/custom_template.tex
PDFOPTS := --pdf-engine=xelatex \
          -V geometry:margin=1in \
          --template=$(PD_TEMPLATE) \
          -V header-includes="\usepackage{amsmath} \usepackage{amssymb} \usepackage{unicode-math} \hyphenpenalty=10000 \exhyphenpenalty=10000 \makeatletter \renewcommand\section{\@startsection{section}{1}{\z@}{-3.5ex}{2.3ex}{\normalfont\Large\bfseries\centering}} \makeatother"

# Ordered list of doc chapters for PDF generation
PDF_CHAPTERS := README.md doc/Architecture.md doc/BasicModel.md doc/Spaces.md doc/Language.md doc/Logic.md doc/Syntax.md doc/Ergodic.md doc/Training.md doc/MachineMinds.md doc/Params.md doc/Installation.md
XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml
MODEL ?= data/BasicModel.xml
PYTHON := PYTHONPATH=bin .venv/bin/python

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

.PHONY : all install xor tomatoes ergodic simple run compare test test_all bench doc_pdf clean \
         train train_micro train_remote train_micro_remote

all : xor

install :
	deactivate 2>/dev/null || true
	rm -rf .venv
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -r requirements.txt


run :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $(XML1)

train :
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log \
		$(if $(TRAIN_HOST),--host $(TRAIN_HOST) --user $(TRAIN_USER) --key-file $(TRAIN_KEY) --remote-dir $(TRAIN_DIR))

train_micro :
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log \
		--max-docs 500 --num-shards 1 --random-shards \
		$(if $(TRAIN_HOST),--host $(TRAIN_HOST) --user $(TRAIN_USER) --key-file $(TRAIN_KEY) --remote-dir $(TRAIN_DIR))

train_remote :
	$(MAKE) train TRAIN_HOST=arbormini.local

train_micro_remote :
	$(MAKE) train_micro TRAIN_HOST=arbormini.local

xor : data/XOR_exact.xml
	make run XML1=$<

tomatoes : data/tomatoes.xml
	make run XML1=$<

ergodic : data/ergodic.xml
	make run XML1=$<

simple : data/simple.xml
	make run XML1=$<

mnist : data/mnist.xml
	make run XML1=$<

SigmaPi :
	cd bin && PYTHONPATH=. ../.venv/bin/python SigmaPi.py

SymPercept :
	cd bin && PYTHONPATH=. ../.venv/bin/python SymPercept.py

SPNN :
	cd bin && PYTHONPATH=. ../.venv/bin/python SPNN.py


compare :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py --compare $(XML1) $(XML2)

test :
	BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python test/test_report.py

test_all :
	BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python test/test_report.py

bench :
	@echo "=== Baseline (no env tweaks) ==="
	PYTHONPATH=bin .venv/bin/python test/bench_training.py

clean :
	rm -f BasicModel.pdf

doc_pdf : BasicModel.pdf


TITLE := Basic Model
BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^
