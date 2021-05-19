import setuptools

REQUIRED_PACKAGES = [
    'tensorflow-transform==0.27.0',
    'tensorflow-data-validation==0.27.0'
]

setuptools.setup(
    name='executor',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'src': ['raw_schema/schema.pbtxt']}
)