import yaml

# Read YAML file
with open('/mnt/vincent-pvc-rwm/Github/MedVision/src/medvision_bm/sft/config/model_info.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Access the data
print(data['model_info']['lingshu'])