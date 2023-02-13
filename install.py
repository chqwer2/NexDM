import os
import yaml

with open("StableDifussion/environment.yaml") as file_handle:
    environment_data = yaml.load(file_handle)

for dependency in environment_data["dependencies"]:
    package_name, package_version = dependency.split("=")
    os.system("pip install {}=={}".format(package_name, package_version))
