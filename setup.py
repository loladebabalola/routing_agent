from setuptools import setup, find_packages

setup(
    name="routing_agent",
    version="1.0.0",
    description="Intelligent local AI model routing framework",
    author="Lola Debabalola",
    author_email="lola@debabalola.com",
    packages=find_packages(),
    install_requires=[
        "PyYAML>=6.0",
        "click>=8.0",
        "rich>=13.0",
        "pytest>=7.0",
        "transformers>=4.0",
        "torch>=2.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "routing-agent=routing_agent.cli.chat:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)