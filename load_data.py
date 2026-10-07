# Reads dataset-uci.xlsx directly
# The file is saved in the "Strict Open XML" format, which pandas / openpyxl can not always open.
# An xlsx file is a zip file with xml files inside, so we read the xml files ourselves.

import re
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd


def column_index(cell_ref):
    """Convert a cell reference like 'AB12' into a column number starting at 0"""
    letters = re.match(r'[A-Z]+', cell_ref).group()
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index - 1


def tag_name(tag):
    """Remove the namespace from an xml tag"""
    return tag.split('}')[-1]


def load_xlsx(path='dataset-uci.xlsx'):
    """Return the first sheet of an xlsx file as a DataFrame"""

    # First try the normal way
    try:
        df = pd.read_excel(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # Otherwise read the xml files inside the xlsx
    with zipfile.ZipFile(path) as z:
        names = z.namelist()

        # Text values are stored in a separate file
        strings = []
        if 'xl/sharedStrings.xml' in names:
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')):
                text = ''.join(t.text or '' for t in si.iter() if tag_name(t.tag) == 't')
                strings.append(text)

        # The first sheet contains the table
        sheet = sorted(n for n in names if n.startswith('xl/worksheets/sheet'))[0]

        rows = []
        for row in ET.fromstring(z.read(sheet)).iter():
            if tag_name(row.tag) != 'row':
                continue
            values = {}
            for cell in row:
                v = next((x.text for x in cell if tag_name(x.tag) == 'v'), None)
                if v is None:
                    continue
                if cell.get('t') == 's':
                    values[column_index(cell.get('r'))] = strings[int(v)]
                else:
                    values[column_index(cell.get('r'))] = float(v)
            rows.append(values)

    # Build the table: first row = column names
    n_columns = max(max(r) for r in rows if r) + 1
    table = [[r.get(i) for i in range(n_columns)] for r in rows]
    df = pd.DataFrame(table[1:], columns=table[0])
    return df.apply(pd.to_numeric)


if __name__ == '__main__':
    df = load_xlsx()
    print(df.shape)
    print(df.iloc[:, 0].value_counts())
