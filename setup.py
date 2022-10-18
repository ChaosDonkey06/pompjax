from setuptools import setup

setup(
    name             = 'pompjax',
    version          = '0.1.0',
    description      = 'Epidemiological partially observed Markov processes with JAX',
    url              = 'https://github.com/ChaosDonkey06/pompjax',
    author           = 'Jaime Cascante Vega',
    author_email     = 'jc5647@cumc.columbia.edu.co',
    license          = 'MIT License',
    packages         = ['pompjax'],
    install_requires =['mpi4py>=2.0',
                       'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: -  :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)