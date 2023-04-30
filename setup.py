from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()

LICENSE: str = "MIT"


setup(
    name="langgym",
    version="0.0.1",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description="Train language-based agents on arbitrary environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    keywords=["machine-learning", "artificial-intelligence", "generative-agents"],
    data_files=[("", ["LICENSE"])],
    include_package_data=True,
    packages=find_packages(
        include=["langgym*"],
        exclude=["tests*"],
    ),
    package_data={"langgym": ["py.typed"]},
    python_requires=">=3.7",
    platforms="any",
    install_requires=[
        "faiss-cpu",
        "gymnasium",
        "openai",
        "pettingzoo[mpe]>=1.22.3",
        "Pillow",
        "pygame",
        # install from commit, until next release after 0.0.147 comes out
        "langchain @ git+https://github.com/hwchase17/langchain.git@0cf934c#egg=langchain",
        "streamlit",
        "tiktoken",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        f"License :: OSI Approved :: {LICENSE} Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    url="https://github.com/cosmicBboy/langgym/",
    project_urls={
        "Source Code": "https://github.com/cosmicBboy/langgym/",
        "Issue Tracker": "https://github.com/cosmicBboy/langgym/issues",
    },
)
