import kagglehub

# Download latest version
path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")

print("Path to dataset files:", path)

# Load dataset
dataset = kagglehub.dataset_upload(path,path)

# Print dataset
#print(dataset)

# Print dataset info
print(dataset.info())

# Print dataset head
print(dataset.head())