import pandas as pd
import re


# Function to parse the tensor values and filter out 255
def parse_tensor(tensor_str):
  numbers = re.findall(r'\d+', tensor_str)
  numbers = [int(num) for num in numbers if int(num) != 255]
  return len(set(numbers))


# Read the data file
with open('unique_vals_epoch0.txt', 'r') as file:
  lines = file.readlines()

# Initialize a dictionary to hold our data
data = {'SIC': [], 'SOD': [], 'FLOE': []}

# Process each line
for line in lines:
  # Find the target name and the tensor data
  match = re.search(r'for chart (\w+):tensor\((.+?)\)', line)
  if match:
    target, tensor_str = match.groups()
    # Parse the tensor and count unique classes
    unique_count = parse_tensor(tensor_str)
    # Append the count to the appropriate target list
    if target in data:
      data[target].append(unique_count)

# Convert dictionary to DataFrame for easier processing
df = pd.DataFrame(data)

# Calculate the average number of unique classes per target
averages = df.mean()

# Print the results
print(averages)