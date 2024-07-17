all: clean zip

zip:
	pdflatex a2.tex
	zip -r a2.zip a2.pdf a2.tex README plots/ plots_retrained/ cnn.py

clean:
	rm -f a2.zip
