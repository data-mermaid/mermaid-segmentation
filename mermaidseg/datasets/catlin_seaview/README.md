### Catlin Seaview

- From the paper "Seaview Survey Photo-quadrat and Image Classification Dataset" (González-Rivero et al., 2019)
- The raw data can be downloaded at: http://data.qld.edu.au/public/Q1281/
```
wget http://data.qld.edu.au/public/Q1281/tabular-data.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/ATL.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/IND_CHA.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/IND_MDV.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_AUS.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_IDN_PHL.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_SLB.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_TLS.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_TWN.zip
wget http://data.qld.edu.au/public/Q1281/annotated-images/PAC_USA.zip
```

In total, the dataset contains 11387 images, of which 30 are damaged:

Area Code | Place | Number of Regions | Number of Annotations | Number of Label Classes Present in Split | Number of Damaged JPG files
--- | --- | --- | --- | --- | ---
ATL | Atlantic | 1407 | 92900 | 67 | 0
IND_CHA | Indian Ocean, Chagos Archipelago | 686 | 52450 | 42 | 0
IND_MDV | Indian Ocean, Maldives | 1612 | 144100 | 38 | 0
PAC_AUS | Pacific Ocean, Australia | 2657 | 186420 | 29 | 0
PAC_IDN_PHL | Pacific Ocean, Indonesia and Philippines | 1608 | 118600 | 67 | 30
PAC_SLB | Pacific Ocean, Solomon Islands | 732 | 59200 | 55 | 0
PAC_TLS | Pacific Ocean, Timor-Leste | 864 | 71600 | 59 | 0
PAC_TWN | Pacific Ocean, Taiwan | 638 | 50000 | 54 | 0
PAC_USA | Pacific Ocean, USA (Hawaii) | 1153 | 83100 | 39 | 0
| | TOTAL | 11357 | 858370 | 194 | 30
