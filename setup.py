from setuptools import setup, find_packages

setup(
    name='trend_analyzer_model',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'statsmodels',
        'pandas',
        'metrics @ git+https://github.com/BobryTeam/metrics.git@pip-deps',
    ],
    author='BobryTeam',
    author_email='sinntexxx@gmail.com',
    description='Trend Analyzer AI model',
    url='https://github.com/BobryTeam/trend-analyzer-model',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)