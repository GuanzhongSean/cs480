all: clean zip

zip:
	mkdir j76xiao
	pdflatex a3.tex
	cp -r a3.pdf a3.tex a3-vit.* adaboost*.py train_test_split.pkl README img/ plot* j76xiao
	zip -r a3.zip j76xiao/

clean:
	rm -rf j76xiao/ a3.zip a3.pdf
