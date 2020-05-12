import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='relepo',
      version='0.1.3',
      description='A Reinforcement Learning test models for poker.',
      url='https://github.com/ilyakochik/relepo.git',
      author='Ilya Kochik',
      author_email='ilya.kochik@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
            # 'tensorflow',
            # 'tf-agents'
      ],
)