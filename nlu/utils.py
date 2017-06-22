import pandas as pd
# avoid newline when dataframe is printed
pd.set_option('display.width', 0)

def print_matrix(matrix, rows_names, columns_names):
    num_rows, num_colums, = matrix.shape
    assert num_rows == len(rows_names)
    assert num_colums == len(columns_names)
    pretty_rows = []
    for row_index, row in enumerate(matrix):
        pretty_rows.append((rows_names[row_index], row)) 
    print(pd.DataFrame.from_items(pretty_rows, orient='index', columns=columns_names))