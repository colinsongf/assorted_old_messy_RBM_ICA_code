all : tags

tags :
	find . | egrep '\.py$$' | xargs etags

clean :
	rm -f ETAGS
