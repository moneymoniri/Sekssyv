"""Leser dataset-uci.xlsx direkte.
Filen er lagret i 'Strict Open XML'-format, som pandas/openpyxl ikke kan lese.
Derfor leser vi xlsx-filen (en zip med XML) selv."""
import zipfile, re
import xml.etree.ElementTree as ET
import pandas as pd

def _col(ref):
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1

def load_xlsx(path='dataset-uci.xlsx'):
    try:                                   # fungerer for vanlige xlsx-filer
        return pd.read_excel(path)
    except Exception:
        pass
    strip = lambda t: t.split('}')[-1]
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        strings = []
        if 'xl/sharedStrings.xml' in names:
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')):
                strings.append(''.join(t.text or '' for t in si.iter() if strip(t.tag) == 't'))
        sheet = sorted(n for n in names if n.startswith('xl/worksheets/sheet'))[0]
        rows = []
        for row in ET.fromstring(z.read(sheet)).iter():
            if strip(row.tag) != 'row':
                continue
            r = {}
            for c in row:
                v = next((x.text for x in c if strip(x.tag) == 'v'), None)
                if v is None:
                    continue
                r[_col(c.get('r'))] = strings[int(v)] if c.get('t') == 's' else float(v)
            rows.append(r)
    ncol = max(max(r) for r in rows if r) + 1
    table = [[r.get(i) for i in range(ncol)] for r in rows]
    df = pd.DataFrame(table[1:], columns=table[0])
    return df.apply(pd.to_numeric)

if __name__ == '__main__':
    d = load_xlsx(); print(d.shape); print(d.iloc[:, 0].value_counts())
