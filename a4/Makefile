all: clean zip

zip:
	jupyter nbconvert --to pdf a4-wgan.ipynb
	jupyter nbconvert --execute --inplace ex2.ipynb
	jupyter nbconvert --to pdf ex2.ipynb
	mkdir j76xiao
	pdflatex a4.tex
	cp -r a4.pdf a4.tex a4-wgan* plot/ ex2.* j76xiao/
	zip -r a4.zip j76xiao/

clean:
	jupyter nbconvert --clear-output --inplace ex2.ipynb
	rm -rf j76xiao/ a4.zip a4.pdf a4-wgan.pdf ex2.pdf
