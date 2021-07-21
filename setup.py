import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="TLAF",

	version="1.0",

	author="Tushar Sarkar , Nishant Rajadhyaksha",

	author_email="tushar.sarkar@somaiya.edu, n.rajadhyaksha@somaiya.edu",

	# #Small Description about module
	description="TLA is built using PyTorch, Transformers and several other State-of-the-Art machine learning techniques and it aims to expedite and structure the cumbersome process of collecting, labeling, and analyzing data from Twitter for a corpus of languages while providing detailed labeled datasets for all the languages.",

	long_description=long_description,
	long_description_content_type="text/markdown",

	url="https://github.com/tusharsarkar3/TLA",
	packages=setuptools.find_packages(),
    include_package_data=True,


		 install_requires= list(open("requirements.txt","r").read().splitlines()),


	license="MIT",

	# classifiers like program is suitable for python3, just leave as it is.
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
