include /bits/projects/Make.mk

# Ordered list of doc chapters for PDF generation
PDF_CHAPTERS := README.md doc/CognitiveScienceSociety.md doc/Architecture.md doc/BasicModel.md doc/MachineMinds.md doc/Params.md
XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml

# The shared MAKE_PDF macro drops later options due to a broken line
# continuation, so override it locally with the reader extensions these docs use.
MAKE_PDF = pandoc $(PDFOPTS) \
		--from=gfm+smart+bracketed_spans+attributes \
		--metadata title="$(TITLE)" \
		--toc --toc-depth=3 \
		--resource-path=.:doc

.PHONY : all xor tomatoes ergodic simple run compare test bench doc_pdf clean

all : xor


run :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $(XML1)

train : data/train.xml
	make run XML1=$<

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

bench :
	@echo "=== Baseline (no env tweaks) ==="
	PYTHONPATH=bin .venv/bin/python test/bench_training.py

clean :
	rm -f BasicModel.pdf

doc_pdf : BasicModel.pdf


TITLE := Basic Model
BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^
