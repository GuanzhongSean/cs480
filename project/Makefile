zip:
	rm -rf j76xiao/ project.zip
	mkdir -p j76xiao/
	jupyter nbconvert --to pdf 6PlantTraitsPredict.ipynb
	cd report && pdflatex project.tex
	cp 6PlantTraitsPredict.ipynb j76xiao/
	cp 6PlantTraitsPredict.pdf j76xiao/
	cp -r report/ j76xiao/
	zip -r project.zip j76xiao/

init:
	kaggle competitions download -c cs-480-2024-spring --force
	unzip -q cs-480-2024-spring.zip

convert:
	jupyter nbconvert --to script 6PlantTraitsPredict.ipynb

submit:
	kaggle competitions submit -c cs-480-2024-spring -f submission.csv -m "Final submission"
