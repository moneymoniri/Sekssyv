# Gallestein-prosjektet

Prosjektet bruker maskinlæring til å skille pasienter med og uten gallestein, og undersøker hvilke målinger modellen legger mest vekt på.

## Filer

### Inndata

| Fil | Beskrivelse |
|---|---|
| `dataset-uci.xlsx` | Datasettet (Excel-filen din, brukes direkte): 319 pasienter, én rad per pasient, med målinger som alder, vekt og blodprøver. |
| `load_data.py` | Hjelpefil som leser Excel-filen. Filen er lagret i et spesialformat som vanlige verktøy ikke klarer å åpne, så denne leser den på en annen måte. |
| `requirements.txt` | Liste over Python-pakker som må installeres. |

### Programmer

| Fil | Beskrivelse |
|---|---|
| `analysis.py` | Trener og sammenligner fire modeller, og skriver ut hvor godt hver av dem gjør det. Lagrer underveisresultater i `state.pkl`. |
| `explain.py` | Finner ut hvilke målinger som betyr mest for modellens svar, og lager tabellene og figurene under. |

Legg alle filene, inkludert `dataset-uci.xlsx`, i samme mappe. Kjør `analysis.py` først og `explain.py` etterpå.

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

## Kort oppsummering

- Modellene har en treffsikkerhet på rundt 75–78 %.
- Viktigste målinger: CRP, deretter vitamin D.
- Alder, kolesterol og lignende har liten betydning.

## Forbehold

- Datasettet er lite (319 pasienter) fra ett sykehus, så resultatene er usikre.
- Resultatene viser sammenhenger, ikke årsaker.
- Det bør kontrolleres at kodingen «0 = gallestein» stemmer, siden retningen på enkelte effekter virker uvanlig.
