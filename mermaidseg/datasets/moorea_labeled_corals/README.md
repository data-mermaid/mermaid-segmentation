From the data reposiotry:
"The Moorea Labeled Corals dataset is a subset of the MCR LTER packaged for computer vision research. It contains 2055 images from three habitats IDs: fringing reef outer 10m and outer 17m, from 2008, 2009 and 2010. It also contains random point annotation (row, col, label) for the nine most abundant labels, four non coral labels: (1) Crustose Coralline Algae (CCA), (2) Turf algae, (3) Macroalgae and (4) Sand, and ﬁve coral genera: (5) Acropora, (6) Pavona, (7) Montipora, (8) Pocillopora, and (9) Porites. These nine classes account for 96% of the annotations and total to almost 400,000 points. These nine classes are the ones analyzed in (Beijbom, 2012); less-abundant genera not treated in the automation are also present in the dataset. These data were published in Beijbom O., Edmunds P.J., Kline D.I., Mitchell G.B., Kriegman D., 'Automated Annotation of Coral Reef Survey Images', IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, Rhode Island, 2012. [BibTex] [pdf] These data are a subset of the raw data from which knb-lter-mcr.4 is derived. This material is based upon work supported by the U.S. National Science Foundation under Grant No. OCE 16-37396 (and earlier awards) as well as a generous gift from the Gordon and Betty Moore Foundation. Research was completed under permits issued by the French Polynesian Government (Délégation à la Recherche) and the Haut-commissariat de la République en Polynésie Francaise (DTRT) (Protocole d'Accueil 2005-2018). This work represents a contribution of the Moorea Coral Reef (MCR) LTER Site." The original data package is released under the Creative Commons license Attribution 4.0 International.

The data can be downloaded at: [https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006](https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006), but you need to authenticate. Download all the data, and then the file structure should look like:

```
/path/to/moorea_labeled_corals_downloaded/
    knb-lter-mcr.5006.3.report.xml
    knb-lter-mcr.5006.3.txt
    knb-lter-mcr.5006.3.xml
    manifest.txt
    2010/
        mcr_lter6_out17m_pole5-6_qu2_20100410.png.txt
        mcr_lter6_out17m_pole5-6_qu2_20100410.png
        ...
    2009/
        mcr_lter6_out17m_pole5-6_qu8_20090331.jpg.txt
        mcr_lter6_out17m_pole5-6_qu8_20090331.jpg
        ...
    2008/
        mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg
        mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg.txt
        ...
```

In total, the data repository contains 2055 images, of which 53 are damaged/missing:


| Year      | Images   | Annotations (9 Classes) | Annotations (20 Classes) | Label Classes Present | Damaged/Missing Images |
| --------- | -------- | ----------------------- | ------------------------ | --------------------- | ---------------------- |
| 2008      | 671      | 131260                  | 133045                   | 26                    | 0                      |
| 2009      | 695      | 132112                  | 133072                   | 21                    | 0                      |
| 2010      | 689      | 113256                  | 120324                   | 26                    | 53                     |
| **TOTAL** | **2055** | **376628**              | **386441**               | **29**                | **53**                 |
