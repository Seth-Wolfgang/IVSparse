import csv
import json

def json_to_csv(json_file, csv_file):
    # Write the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header_written = False

        with open(json_file, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                if not header_written:
                    keys = list(data.keys())
                    writer.writerow(keys)  # Write the header row
                    header_written = True
                writer.writerow(data.values())  # Write each element

    print("Conversion completed successfully!")

# Example usage
json_file_path = 'ratings.json'
csv_file_path = 'bigratings.csv'
json_to_csv(json_file_path, csv_file_path)
