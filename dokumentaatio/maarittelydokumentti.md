# Määrittelydokumentti

## Kurssitiedot

Opinto-ohjelma: Tietojenkäsittelytieteen kandidaatin tutkinto

Toteutan harjoitustyön Pythonilla. Voin tarvittaessa vertaisarvioida myös
JavaScriptillä tehtyjä töitä suomeksi tai englanniksi.

## Ohjelmakuvaus

Harjoitustyön aihe on kasvojentunnistusohjelma. Kasvojentunnistukseen
käytetään ominaiskasvoja (eigenfaces) ja pääkomponenttianalyysia (PCA).
Ohjelma koulutetaan joukolla edestäpäin otettuja mustavalkoisia samankokoisia 
kasvokuvia, jonka jälkeen ohjelma osaa tunnistaa kasvot, jos ne kuuluvat 
koulutusjoukkoon, tai todeta että kyseessä ovat uudet kasvot.

PCA-algoritmin aikavaativuus on luokkaa `$O(n*p^2+p^3)$`, missä n on 
havaintoaineiston otosten määrä ja p niiden piirteden lukumäärä.

## Lähteet

[MML book]( https://mml-book.com)
[Eigenfaces Wikipedia](https://en.wikipedia.org/wiki/Eigenface)
[PCA Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
 

