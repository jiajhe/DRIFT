LATEXMK ?= latexmk

.PHONY: all clean

all: main.pdf

main.pdf: main.tex aaai2027.sty aaai2027.bst aaai2027.bib
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

clean:
	$(LATEXMK) -C main.tex
