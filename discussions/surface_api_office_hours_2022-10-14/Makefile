.PHONY: clean all

all: surfaces_draft.html

surfaces_draft.html: surfaces_draft.ipynb
	jupyter nbconvert --no-prompt --to html $<

surfaces_draft.ipynb: surfaces_draft.md
	jupytext --execute --to notebook $<

clean:
	rm -f surfaces_draft.ipynb surfaces_draft.html
