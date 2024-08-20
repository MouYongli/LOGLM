from setuptools import find_packages, setup

setup(
    name = "LOGLM",
    version = "0.1.0",
    author = "Qihui Feng, Er Jin, Yongli Mou, Johannes Stegmaier , Stefan Decker and Gerhard Lakemeyer",
    author_email = "mou@dbis.rwth-aachen.de",
    description = ("On the translation from natural language to formal language via LLM prompting"),
    license = "MIT",
    url = "https://github.com/MouYongli/LOGLM",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Formal Language :: Natural Language Processing",
        "License :: OSI Approved :: MIT License",
    ],
)