import json
import csv
import os

input_file = r'c:\Travail\S6C01\data\yelp_academic_dataset_business.json'
output_file = r'c:\Travail\S6C01\data\yelp_academic_dataset_business.csv'

def convert():
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as infile:
        first_line = infile.readline()
        if not first_line:
            return
        
        headers = list(json.loads(first_line).keys())
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
