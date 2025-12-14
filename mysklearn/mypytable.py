from mysklearn import myutils

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)


    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        N = len(self.data)
        M = len(self.column_names)
        return N, M 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        if isinstance(col_identifier, str):
            try:
                col_index = self.column_names.index(col_identifier)
            except ValueError:
                raise ValueError('Column name not found')
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError('Column index out of range')
        else:
            raise ValueError('Invalid column identifier type')
        column = []
        for row in self.data:
            value = row[col_index]
            if include_missing_values or value != 'NA':
                column.append(value)
        return column 


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except (ValueError, TypeError):
                    pass 


    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        i = 0
        count = 0 
        while i < len(self.data):
            if count in row_indexes_to_drop:
                del self.data[i]
            else:
                pass 
                i += 1
            count += 1


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline = '') as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.column_names = rows[0]
            self.data = rows[1:]
        self.convert_to_numeric()
        return self


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)
            for row in self.data:
                if row == []:
                    pass 
                else:
                    writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        key_indices = []
        for name in key_column_names:
            key_indices.append(self.column_names.index(name))
        duplicate = []
        seen = []
        i = 0
        for row in self.data:
            key = []
            for index in key_indices:
                key.append(row[index])
            key_tuple = tuple(key)
            if key_tuple in seen:
                duplicate.append(i)
            else:
                seen.append(key_tuple)
            i += 1
        return duplicate 


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        new_data = []
        for row in self.data:
            has_missing = False
            for value in row:
                if value == 'NA':
                    has_missing = True
                    break
            if not has_missing:
                new_data.append(row) 
            else:
                pass
        self.data = new_data


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        values = []
        for row in self.data:
            if row[col_index] == 'NA':
                pass
            else:
                values.append(float(row[col_index]))
        if values:
            avg = sum(values)/len(values)
            for row in self.data:
                if row[col_index] == "NA":
                    row[col_index] = avg
                else:
                    pass 


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_data = []
        for col_name in col_names:
            col = self.get_column(col_name, include_missing_values = False)
            col = [float(x) for x in col]
            if col:
                min_val = min(col)
                max_val = max(col)
                mid_val = (min_val + max_val)/2
                avg_val = sum(col)/len(col)

                sorted_col = sorted(col)
                n = len(sorted_col)
                if n % 2 == 1:
                    median_val = sorted_col[n//2]
                else:
                    median_val = (sorted_col[n//2-1]+sorted_col[n//2])/2
                stats_data.append([col_name, min_val, max_val, mid_val, avg_val, median_val])
        new_col_names = ["attribute", "min", "max", "mid", "avg", "median"]
        return MyPyTable(new_col_names, stats_data) # TODO: fix this


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indices = []
        for name in key_column_names:
            self_key_indices.append(self.column_names.index(name))
        other_key_indices = []
        for name in key_column_names:
            other_key_indices.append(other_table.column_names.index(name))
        result_columns = self.column_names[:]
        for col in other_table.column_names:
            if col not in key_column_names:
                result_columns.append(col)
        result_data = []
        i = 0
        while i < len(self.data):
            row1 = self.data[i]
            j = 0
            while j < len(other_table.data):
                row2 = other_table.data[j]
                keys_match = True
                k = 0
                while k < len(key_column_names):
                    if row1[self_key_indices[k]] != row2[other_key_indices[k]]:
                        keys_match = False
                        break
                    k += 1
                if keys_match:
                    new_row = row1[:]
                    for l in range(len(other_table.column_names)):
                        if other_table.column_names[l] not in key_column_names:
                            new_row.append(row2[l])
                    result_data.append(new_row)
                j += 1
            i += 1
        return MyPyTable(result_columns, result_data) # TODO: fix this
    
    
    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """

        self_key_indices = []
        for name in key_column_names:
            self_key_indices.append(self.column_names.index(name))
        other_key_indices = []
        for name in key_column_names:
            other_key_indices.append(other_table.column_names.index(name))
        result_columns = self.column_names[:]
        for col in other_table.column_names:
            if col not in key_column_names:
                result_columns.append(col)
        result_data = []
        matched_other = [False] * len(other_table.data)
        i = 0
        while i < len(self.data):
            row1 = self.data[i]
            found = False
            j = 0
            while j < len(other_table.data):
                row2 = other_table.data[j]
                keys_match = True
                k = 0
                while k < len(key_column_names):
                    if row1[self_key_indices[k]] != row2[other_key_indices[k]]:
                        keys_match = False
                        break
                    k += 1
                if keys_match:
                    new_row = row1[:]
                    for l in range(len(other_table.column_names)):
                        if other_table.column_names[l] not in key_column_names:
                            new_row.append(row2[l])
                    result_data.append(new_row)
                    matched_other[j] = True
                    found = True
                j += 1
            if not found:
                new_row = row1[:]
                for col in other_table.column_names:
                    if col not in key_column_names:
                        new_row.append("NA")
                result_data.append(new_row)
            i += 1
        j = 0
        while j < len(other_table.data):
            if not matched_other[j]:
                row2 = other_table.data[j]
                na_row = []
                k = 0
                while k < len(self.column_names):
                    if self.column_names[k] in key_column_names:
                        idx = other_table.column_names.index(self.column_names[k])
                        na_row.append(row2[idx])
                    else:
                        na_row.append("NA")
                    k += 1
                l = 0
                while l < len(other_table.column_names):
                    if other_table.column_names[l] not in key_column_names:
                        na_row.append(row2[l])
                    l += 1
                result_data.append(na_row)
            j += 1
        return MyPyTable(result_columns, result_data) # TODO: fix this

    def copy(self):
        '''
        Return a deep copy of the current MyPyTable object.
        '''
        new_column_names = self.column_names[:]
        new_data = [row[:] for row in self.data]
        return MyPyTable(new_column_names, new_data)

    def project_table(self, col_names):
        # find indices of the desired columns
        keep_indices = []
        for name in col_names:
            keep_indices.append(self.column_names.index(name))
        # build new data with only those columns
        projected_data = []
        i = 0
        while i < len(self.data):
            row = self.data[i]
            new_row = []
            j = 0
            while j < len(keep_indices):
                idx = keep_indices[j]
                new_row.append(row[idx])
                j += 1
            projected_data.append(new_row)
            i += 1

        return MyPyTable(col_names, projected_data)