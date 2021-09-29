import setuptools

with open("README.md","r") as fh:
    long_descriptions = fh.read()

def make_required_install_packages():
    return [
        "tensorflow",
        "optuna",
        "typing",
        "keras",
        "sklearn",
    ]
setuptools.setup(
    name="autoprototype",
    version="0.0",
    author="Ankan Ghosh",
    author_email = "ankan@ideas2it.com",
    description=" This is a module for Hyper-parameter tuning and rapid prototyping",
    long_description = long_descriptions,
    long_description_content_type = "text/markdown",
    url="https://github.com/Ideas2IT/auto-prototype.git",
    install_requires=["tensorflow",
                      "optuna",
                      "typing",
                      "keras",
                      "sklearn"
                      ],

    keywords=["RapidOpt",
              "Hyper parameter Optimizations",
              "Optuna",
              ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)