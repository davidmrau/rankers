import sys
import jsonlines
from tqdm import tqdm
import json
f = sys.argv[1]
# Define the name of your input and output JSONL files
input_file = sys.argv[1]
output_file = f"{input_file}_sorted.jsonl"

for l in open(input_file):
    try:
        json.loads(l)
    except:
        print(l)
exit()
# Open the input and output JSONL files
with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
    # Use tqdm to create a progress bar for the iteration
    for line in tqdm(reader):
        # Get the dictionary under the key "plm"
        plm_dict = line["plm"]
        # Sort the dictionary by value in decreasing order
        sorted_list = sorted(plm_dict.items(), key=lambda x: x[1], reverse=True)
        # Replace the original dictionary with the sorted list
        line["plm"] = sorted_list
        # Write the updated line to the output file
        writer.write(line)
