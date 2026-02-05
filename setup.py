from setuptools import setup, find_packages

# Although pyproject.toml is the source of truth, explicit configuration
# here can help with tool compatibility and package discovery.
if __name__ == "__main__":
    setup(
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
    )
