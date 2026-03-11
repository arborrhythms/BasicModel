include ../Make.mk



all : BasicModel.pdf

TITLE := Basic Model
BasicModel.pdf : Readme.md
	$(MAKE_PDF) -o $@ $^
