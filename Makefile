include /bits/projects/Make.mk

# Ordered list of doc chapters for PDF generation
PDF_CHAPTERS := README.md $(wildcard doc/*.md) todo.md
XML1 ?= data/simple.xml
XML2 ?= data/ergodic-only.xml

.PHONY : all xor tomatoes ergodic simple run compare test doc_pdf clean

all : xor


run :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $(XML1)

xor : data/xor.xml
	make run XML1=$<

tomatoes : data/tomatoes.xml
	make run XML1=$<

ergodic : data/ergodic.xml
	make run XML1=$<

simple : data/simple.xml
	make run XML1=$<

compare :
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py --compare $(XML1) $(XML2)

test :
	PYTHONPATH=bin .venv/bin/python -m pytest test/test_basicmodel.py -v

clean :
	rm -f BasicModel.pdf

doc_pdf : BasicModel.pdf


TITLE := Basic Model
BasicModel.pdf : $(PDF_CHAPTERS)
	$(MAKE_PDF) -o $@ $^
