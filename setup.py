import setuptools

with open("README.md","r") as fh:
    long_descriptions = fh.read()


setuptools.setup(
    name="autoprototype",
    version="1.2.1",
    author="Ankan Ghosh",
    author_email = "ankan@ideas2it.com",
    description=" This is a module for Hyper-parameter"
                " tuning and rapid prototyping",
    long_description = long_descriptions,
    long_description_content_type = "text/markdown",
    url="https://github.com/Ideas2IT/auto-prototype.git",
    install_requires=["tensorflow==2.5.0",
                      "optuna",
                      ],

    keywords=["autoprototype","auto-prototype",
              "Hyper parameter Optimizations",
              "Optuna",
              ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
