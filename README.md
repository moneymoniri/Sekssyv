# Gallestein-prosjektet

Prosjektet bruker maskinlæring til å skille pasienter med og uten gallestein, og undersøker hvilke målinger modellen legger mest vekt på.

## Filer

### Inndata

| Fil | Beskrivelse |
|---|---|
| `dataset-uci.xlsx` | Datasettet (ikke inkludert, du har den allerede): 319 pasienter med målinger som alder, vekt og blodprøver. Legg den i samme mappe som programmene. |
| `load_data.py` | Hjelpefil som leser Excel-filen. Filen er lagret i et spesialformat som vanlige verktøy ikke klarer å åpne, så denne leser den på en annen måte. |
| `requirements.txt` | Liste over Python-pakker som må installeres. |

### Programmer

| Fil | Beskrivelse |
|---|---|
| `analysis.py` | Trener og sammenligner fire modeller, og skriver ut hvor godt hver av dem gjør det. Lagrer underveisresultater i `state.pkl`. |
| `explain.py` | Finner ut hvilke målinger som betyr mest for modellens svar, og lager tabellene og figurene under. |

Legg alle filene, inkludert din egen `dataset-uci.xlsx`, i samme mappe. Kjør `analysis.py` først og `explain.py` etterpå.

### Underveisfil

| Fil | Beskrivelse |
|---|---|
| `state.pkl` | Mellomlagring mellom de to programmene. Trenger ikke åpnes. |

### Resultater

| Fil | Beskrivelse |
|---|---|
| `importance.csv` | Hvor viktig hver måling er for modellen. Høyere tall betyr viktigere. |
| `univariate.csv` | Sammenligning av gruppene med og uten gallestein, én måling om gangen. |
| `lr_coef.csv` | Om en høyere verdi øker eller reduserer sannsynligheten (logistisk regresjon). |
| `fig_importance.png` | Stolpediagram over de viktigste målingene. |
| `fig_pdp.png` | Hvordan modellens svar endrer seg når én måling går fra lav til høy. |
| `fig_coef.png` | Hvilke målinger som trekker mot «gallestein» og hvilke som trekker mot «ikke gallestein». |

## Kjøring (Mac)

```
cd ~/Downloads/files
pip3 install -r requirements.txt
python3 analysis.py
python3 explain.py
```

## Kjøring (Windows)

1. Installer Python fra python.org. Huk av for **«Add python.exe to PATH»** på første skjermbilde i installasjonen.
2. Pakk ut zip-filen, eller legg alle filene i én mappe, for eksempel `C:\Users\DittNavn\Downloads\gallstone_project`.
3. Åpne mappen i Filutforsker, skriv `cmd` i adressefeltet øverst og trykk Enter. Et svart kommandovindu åpnes i riktig mappe.
4. Kjør kommandoene én og én:

```
pip install -r requirements.txt
python analysis.py
python explain.py
```

Hvis `python` ikke finnes, prøv `py` i stedet (`py -m pip install -r requirements.txt`, `py analysis.py`, `py explain.py`).

Figurene (`fig_*.png`) og tabellene (`*.csv`) dukker opp i samme mappe når `explain.py` er ferdig.

## Kort oppsummering

- Modellene har en treffsikkerhet på rundt 75–78 %.
- Viktigste målinger: CRP, deretter vitamin D.
- Alder, kolesterol og lignende har liten betydning.

## Forbehold

- Datasettet er lite (319 pasienter) fra ett sykehus, så resultatene er usikre.
- Resultatene viser sammenhenger, ikke årsaker.
- Det bør kontrolleres at kodingen «0 = gallestein» stemmer, siden retningen på enkelte effekter virker uvanlig.
