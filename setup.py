import setuptools

setuptools.setup(
      name='relepo',
      version='0.1.0',
      description='A Reinforcement Learning test model for poker.',
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