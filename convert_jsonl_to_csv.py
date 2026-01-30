import os
import json
import csv

def convert_jsonl_to_csv(input_path, output_path):
    """
    Converts a JSONL file to a CSV file.
    Uses utf-8-sig encoding for better compatibility with Power BI/Excel (adds BOM).
    Uses the csv module to properly handle commas, quotes, and newlines in text.
    """
    print(f"Conversion de {input_path} vers {output_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            # On lit la première ligne pour obtenir les en-têtes (headers)
            first_line = infile.readline()
            if not first_line:
                print(f"Le fichier {input_path} est vide.")
                return
            
            headers = list(json.loads(first_line).keys())
            
            # On revient au début du fichier pour tout traiter
            infile.seek(0)
            
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                
                count = 0
                for line in infile:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    
                    try:
                        data = json.loads(stripped_line)
                        writer.writerow(data)
                        count += 1
                        if count % 100000 == 0:
                            print(f"{count} lignes traitées...")
                    except json.JSONDecodeError:
                        print(f"Erreur de lecture JSON sur une ligne. Passage...")
                
        print(f"Succès : {output_path} créé ({count} lignes).")
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")

def main():
    data_dir = r'c:\Travail\S6C01\data'
    
    if not os.path.exists(data_dir):
        print(f"Le dossier {data_dir} n'existe pas.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    if not files:
        print("Aucun fichier .jsonl trouvé dans le dossier data.")
        return

    for filename in files:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, filename.replace('.jsonl', '.csv'))
        convert_jsonl_to_csv(input_path, output_path)

if __name__ == "__main__":
    main()
