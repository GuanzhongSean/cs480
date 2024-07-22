all: clean init

init:
	kaggle competitions download -c cs-480-2024-spring
	unzip cs-480-2024-spring.zip

submit:
	kaggle competitions submit -c cs-480-2024-spring -f submission.csv -m "From Makefile"

clean:
	rm -rf cs-480-2024-spring.zip data/ submission.csv
