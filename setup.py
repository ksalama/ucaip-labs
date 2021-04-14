import setuptools

REQUIRED_PACKAGES = [
    #'tensorflow==2.3'
    'tensorflow-transform==0.26.0',
    'tensorflow-data-validation==0.26.0'
]

setuptools.setup(
    name='trainer',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'model_src': ['raw_schema/schema.pbtxt']}
)