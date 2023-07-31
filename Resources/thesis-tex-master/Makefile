PRIMARIES:=thesis.pdf
BUILDDIR:=build

.PHONY: all $(PRIMARIES) $(patsubst %,$(BUILDDIR)/%,$(PRIMARIES))
all: $(PRIMARIES)

$(PRIMARIES): %.pdf: $(BUILDDIR)/%.pdf
	cp '$<' .

$(patsubst %,$(BUILDDIR)/%,$(PRIMARIES)): %.pdf:
	mkdir -p '$(dir $@)'
	latexmk -pdf -bibtex -synctex=0 -interaction=nonstopmode -output-directory='$(dir $@)' '$(patsubst $(BUILDDIR)/%.pdf,%.tex,$@)'

.PHONY: almostclean
almostclean:
	if [ -e '$(BUILDDIR)' ] ; then rm -r '$(BUILDDIR)' ; fi

.PHONY: clean
clean: almostclean
	for f in $(PRIMARIES) ; do \
		if [ -e "$$f" ] ; then rm "$$f" ; fi ;\
	done
