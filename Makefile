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
PYTHON_BOOTSTRAP ?= python3.12
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
PDF_CHAPTERS := README.md  doc/Installation.md doc/Architecture.md doc/BasicModel.md doc/Spaces.md doc/STM.md doc/Language.md doc/Mereology.md doc/Logic.md doc/Reasoning.md doc/Training.md doc/Ergodic.md doc/MachineMinds.md doc/Params.md
XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml
MODEL ?= data/MM_20M_legacy.xml
PYTHON := PYTHONPATH=bin $(VENV_PYTHON)

# The shared MAKE_PDF macro drops later options due to a broken line
# continuation, so override it locally with the reader extensions these docs use.
MAKE_PDF = pandoc $(PDFOPTS) \
		--from=gfm+smart+bracketed_spans+attributes \
		--metadata title="$(TITLE)" \
		--toc --toc-depth=3 \
		--resource-path=.:doc

.PHONY : all install xor tomatoes ergodic simple run compare test test_all bench doc clean \
         train train_micro bench_local bench_sync bench_remote bench_pull

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
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log

train_micro : $(VENV_STAMP)
	$(PYTHON) bin/train.py --model $(MODEL) --data text --log \
		--max-docs 1000000 --num-shards 10 --num-epochs 1 --batches 1440 --random-shards

xor : data/MM_xor.xml
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(MAKE) run XML1=$<

tomatoes : data/tomatoes.xml
	$(MAKE) run XML1=$<

ergodic : data/ergodic.xml
	$(MAKE) run XML1=$<

simple : data/simple.xml
	$(MAKE) run XML1=$<

mnist : data/mnist.xml
	$(MAKE) run XML1=$<

SigmaPi : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) etc/SigmaPi.py

SymPercept : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) etc/SymPercept.py

SPNN : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) etc/SPNN.py


compare : $(VENV_STAMP)
	cd bin && PYTHONPATH=. $(VENV_PYTHON_FROM_BIN) Models.py --report --compare $(XML1) $(XML2)

# `make test` runs the default (fast) suite; tests tagged slow (>30s wall) are
# skipped via the RUN_SLOW gate (see test/*.py `_RUN_SLOW`). `make test_all`
# sets RUN_SLOW=1 to also run them.
test : $(VENV_STAMP)
	BASICMODEL_DEVICE=cpu PYTHONPATH=bin $(VENV_PYTHON) test/test_report.py

test_all : $(VENV_STAMP)
	RUN_SLOW=1 BASICMODEL_DEVICE=cpu PYTHONPATH=bin $(VENV_PYTHON) test/test_report.py

bench : $(VENV_STAMP)
	@echo "=== Baseline (no env tweaks) ==="
	PYTHONPATH=bin $(VENV_PYTHON) test/bench_training.py

# --- Recon fidelity/timing bench (bin/recon_bench.py; doc/plans/2026-07-03-reconstruction-fidelity-execution.md Task 2) ---
# bench_local pins cpu+eager for reproducibility (MPS is seeded-nondeterministic; the iCloud path space breaks inductor).
# bench_remote runs on ArborStudio's native path; REMOTE_COMPILE=auto is the compile experiment (Task 8).
EPOCHS ?= 3
REMOTE_COMPILE ?= auto
BASICMODEL_DEVICE ?= cpu
ARBOR_USER ?= arogers
ARBOR_HOST ?= ArborStudio.local
ARBOR_KEY ?= ~/.ssh/id_ed25519_arborstudio
ARBOR_DEST ?= ~/WikiOracle/basicmodel
# ARBOR_TUNNEL=1 routes off-LAN via the wikiOracle.org reverse tunnel (parent repo `make ssh`): jump through bitnami@wikiOracle.org to the tunnel listener at 127.0.0.1:2222.
ARBOR_TUNNEL ?= 0
ifeq ($(ARBOR_TUNNEL),1)
ARBOR_HOST = 127.0.0.1
ARBOR_KEY = /bits/cloud/bin/arssh.pem
ARBOR_SSH_OPTS = -o ConnectTimeout=15 -i $(ARBOR_KEY) -o IdentitiesOnly=yes \
	-o HostKeyAlias=ArborStudio-via-wikioracle -o StrictHostKeyChecking=accept-new \
	-o 'ProxyCommand=ssh -i /bits/cloud/bin/wikiOracle.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -W %h:%p bitnami@wikiOracle.org' \
	-p 2222
else
ARBOR_SSH_OPTS = -o ConnectTimeout=10 -i $(ARBOR_KEY)
endif
ARBOR_SSH = ssh $(ARBOR_SSH_OPTS) $(ARBOR_USER)@$(ARBOR_HOST)
# Excludes: venvs/caches/git plus large regenerable artifacts (fineweb 7.8G, embeddings 5.1G, MNIST 80M, ckpt/kv dumps).
BENCH_EXCLUDES = --exclude .venv --exclude venv --exclude output --exclude .git \
	--exclude .claude --exclude .worktrees --exclude __pycache__ --exclude '*.pyc' \
	--exclude .DS_Store --exclude data/fineweb --exclude data/embeddings \
	--exclude data/MNIST --exclude data/mnist_test.csv \
	--exclude 'data/*.ckpt' --exclude 'data/*.kv'

# bench_* default MODEL to the grammar config (train keeps its legacy default); command-line MODEL=... still wins.
bench_local bench_remote : MODEL = data/MM_20M_grammar.xml

bench_local : $(VENV_STAMP)
	BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager $(PYTHON) bin/recon_bench.py $(MODEL) --epochs $(EPOCHS) --out output/

bench_sync :
	$(ARBOR_SSH) "mkdir -p $(ARBOR_DEST)"
	rsync -av --progress $(BENCH_EXCLUDES) -e "ssh $(ARBOR_SSH_OPTS)" \
		./ $(ARBOR_USER)@$(ARBOR_HOST):$(ARBOR_DEST)/

# Non-interactive ssh PATH omits /opt/homebrew/bin (python3.12 venv bootstrap lives there), so prefix it remotely.
bench_remote : bench_sync
	$(ARBOR_SSH) "export PATH=/opt/homebrew/bin:\$$PATH && cd $(ARBOR_DEST) && ([ -x .venv/bin/python ] || make install) && \
		BASICMODEL_DEVICE=$(BASICMODEL_DEVICE) MODEL_COMPILE=$(REMOTE_COMPILE) \
		PYTHONPATH=bin .venv/bin/python bin/recon_bench.py $(MODEL) --epochs $(EPOCHS) --out output/"

bench_pull :
	rsync -av -e "ssh $(ARBOR_SSH_OPTS)" \
		--include 'recon_*.json' --include '*.profile.txt' --exclude '*' \
		$(ARBOR_USER)@$(ARBOR_HOST):$(ARBOR_DEST)/output/ output/

clean :
	rm -f BasicModel.pdf
	rm -rf output/*

doc : BasicModel.pdf


TITLE := Basic Model
BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^
