TITLE := Basic Model
include /bits/projects/Make.mk

# Ordered list of doc chapters for PDF generation
PDF_CHAPTERS := README.md $(wildcard doc/*.md)

all : BasicModel.pdf

doc_pdf : BasicModel.pdf

BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^

xor : data/xor.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

tomatoes : data/tomatoes.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

ergodic : data/ergodic.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

simple : data/simple.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

run :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $(XML)

XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml
compare :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py --compare $(XML1) $(XML2)

test :
	PYTHONPATH=bin .venv/bin/python -m pytest test/test_basicmodel.py -v

clean :
	rm -f BasicModel.pdf

.PHONY : all xor tomatoes ergodic simple run compare test doc_pdf clean
