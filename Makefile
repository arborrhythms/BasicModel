TITLE := Basic Model

all : BasicModel.pdf

BasicModel.pdf : README.md
	pandoc --metadata title="$(TITLE)" -V geometry:margin=1in -o $@ $<

xor : data/xor.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

tomatoes : data/tomatoes.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python BasicModel.py $<

ergodic : data/ergodic.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python Ergodic.py $<

simple : data/simple.xml
	cd bin && PYTHONPATH=. ../.venv/bin/python Ergodic.py $<

test :
	PYTHONPATH=bin .venv/bin/python -m pytest test/test_basicmodel.py -v

clean :
	rm -f BasicModel.pdf

.PHONY : all basic ergodic test clean
