import json
import csv
import os

input_file = r'c:\Travail\S6C01\data\yelp_academic_dataset_user4students.jsonl'
output_file = r'c:\Travail\S6C01\data\yelp_academic_dataset_user4students.csv'

def convert():
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as infile:
        first_line = infile.readline()
        if not first_line:
            return
        
        headers = [k for k in json.loads(first_line).keys() if k not in ('friends', 'latitude', 'longitude')]
        infile.seek(0)
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=headers, quoting=csv.QUOTE_MINIMAL, extrasaction='ignore')
            writer.writeheader()
            
            for line in infile:
                if line.strip():
                    writer.writerow(json.loads(line))

if __name__ == "__main__":
    print(f"Converting {input_file} to CSV...")
    convert()
    print("Done.")
