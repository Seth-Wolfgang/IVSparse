import csv


def convert_scientific_notation(value):
    try:
        # Split the scientific notation string into base and exponent
        base, exponent = value.split('e')
        base = float(base)
        exponent = int(exponent)
        
        # Calculate the converted value
        converted_value = base * (10 ** exponent)
        
        return converted_value
    except ValueError:
        return value  # Return t



def convert_to_coo(x_data_file, x_indices_file, x_indptr_file, coo_file):
    # Read data.csv
    with open(x_data_file, 'r') as file:
        data_reader = csv.reader(file)
        data = [row[0] for row in data_reader]
    
    data = [convert_scientific_notation(value) for value in data]

    # Read rows.csv
    with open(x_indices_file, 'r') as file:
        rows_reader = csv.reader(file)
        rows = [row[0] for row in rows_reader]

    rows = [int(convert_scientific_notation(value)) for value in rows]

    # Read cols.csv
    with open(x_indptr_file, 'r') as file:
        cols_reader = csv.reader(file)
        cols = [row[0] for row in cols_reader]

    cols = [int(convert_scientific_notation(value)) for value in cols]

    # Write to COO matrix file
    with open(coo_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['row', 'col', 'value'])  # Write the header row

        for col, start_idx in enumerate(cols):
            end_idx = cols[col + 1] if col + 1 < len(cols) else len(rows)
            for i in range(start_idx, end_idx):
                writer.writerow([i, rows[i], data[i]])

    print("Conversion completed successfully!")

# Example usage
path = "/home/sethwolfgang/matrices/1.0"
x_data_file_path = path + '/vals.csv'
x_indices_file_path = path + '/inner.csv'
x_indptr_file_path = path + '/outer.csv'
coo_file_path = '/home/sethwolfgang/vscode/CSF-Matrix/Benchmarking/helpful_scripts/output.csv'

convert_to_coo(x_data_file_path, x_indices_file_path, x_indptr_file_path, coo_file_path)
