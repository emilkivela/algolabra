# Toteutusdokumentti

## Ohjelman rakenne
Harjoitustyöni aihe on kasvojentunnistusohjelma, joka osaa luokitella annetun kuvan ihmisen kasvoista jos hänestä on jo annettu kuvia
harjoitusdataan, tai tuntemattomaksi jos ohjelma ei ole ennen nähnyt kuvaa kyseisestä henkilöstä.
Ohjelma käyttää harjoitusdatanaan [AT&T The Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)-kuvapankkia, jossa on 10 kuvaa 40 eri henkilöstä, eli 400 kuvaa. Olen poistanut ensimmäisiltä
kymmeneltä henkilöltä yhden kuvan, joita käytän testisyötteinä. Jokainen kuva on kokoa 92 x 112 pikseliä. Jokaisessa kuvassa on siis 
hirvittävän paljon dimesioita, mutta kasvojentunnistusta varten merkittävää dataa on paljon vähemmän. Löytääksemme nämä keskeisimmät piirteet joita tarvitsemme tunnistamaan kasvot, hyödynnämme pääkomponenttianalyysia ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)). Pääkomponenttianalyysi on dimension redusointitekniikka, jonka avulla löydetään datasta ne komponentin, joilla voidaan esittää datan keskeisimmät piirteet ilman, että informaatiota menee hukkaan.


Jokainen 390 kuvaa muutetaan yksiulotteisiksi vektoreiksi, jotka lisätään matriisin T sarakkeiksi. Sen jälkeen matriisista T lasketaan
keskiarvovektori, joka vähennetään jokaisesta sarakkeesta. Data siis keskitetään. Kutsumme tätä keskitettyä matriisia matriisiksi A.
Haluamme laskea matriisin A kovarianssimatriisista ominaisarvot ja ominaisvektorit. Ominaisvektorien voidaan ajatella kuvaavan joukkona piirteitä, jotka yhdessä kuvaavat variaatiota jokaisen kasvokuvan välillä. Jokaisesta ominaisvektorista voi rakentaa kuvan,  ja näistä syntyneitä haamumaisia kasvoja kutsumme ominaiskasvoiksi [Eigenface](https://en.wikipedia.org/wiki/Eigenface). Jokainen kasvo voidaan kuvata näiden ominaisvektorien lineaarikombinaationa. Jokainen kasvo voidaan approksimoida käyttäen "parhaimpia" ominaiskasvoja, eli niitä joilla on suurin ominaisarvon. Nämä ominaiskasvot vastaavat suurimmasta määrästä variaatiota kuvajoukossamme.

Normaalisti laskisimme kovarianssimatriisin S kaavalla $S = A * A^T$, mutta koska matriisi A on kokoa 10 304 x 400, olisi silloin S kokoa 10 304 x 10 304 ja siten turhan raskas laskea.
Laskemmekin "pienen" kovarianssimatriisin $L = A^T * A$, joka on vain kokoa 400 x 400, mutta jolla on samat ominaisarvot kuin matriisilla S ja jonka ominaisvektorit ovat matriisin S ominaisvektorien kertomia. Laskemme matriisin L ominaisarvot ja -vektorit [power iteration-algoritmin](https://en.wikipedia.org/wiki/Power_iteration) avulla, joka laskee matriisin suurinta ominaisarvoa vastaavan ominaisvektorin. Ominaisarvon saamme laskettua ominaisvektorista [Rayleighin osamäärällä](https://en.wikipedia.org/wiki/Rayleigh_quotient). Tämän jälkeen voimme poistaa suurimman ominaisarvon vaikutuksen matriisista L Hotellingin deflaatiolla. Sen jälkeen voimme taas laskea suurimman ominaisarvon deflatoidusta matriisista ja niin edelleen, kunnes olemme saaneet k tärkeintä ominaiskasvoa.

Valitsemme k tärkeintä pääkomponenttia (= ominaiskasvoa) päättämällä jonkun mielivaltaisen rajan kokonaisvarianssille ε, jonka haluamme ominaiskasvomme kattavan. Tässä ohjelmassa se on toistaiseksi 95% kaikesta varianssista.
Kokonaisvarianssi $v = (λ_1 + λ_2 + ... + λ_n)$, missä n = kaikkien komponenttien määrä ja λ on komponentin ominaisarvo. K on pienin luku jolle pätee $((λ_1 + λ_2 + ... + λ_k) / v) > ε$

Talletamme tällä tavalla saadut ominaiskasvot matriisiin, ja power iteration-algoritmin ansiosta ne ovat valmiiksi suuruusjärjestyksessä ominaisarvon mukaan. Ominaiskasvojen avulla laskemme jokaiselle harjoitusdatasta löytyvälle henkilölle oman painovektorin laskemalla keskiarvon jokaisen henkilön kuvan painovektorin keskiarvon. Ohjelma tunnistaa ihmisen laskemalla syötekuvalle myös painovektorin ja laskemalla minkä henkilön painovektoriin on pienin Euklidinen etäisyys.

## Viitteet
- [Turk and Pentland, Eigenfaces for Recognition](https://direct.mit.edu/jocn/article/3/1/71/3025/Eigenfaces-for-Recognition)
- [Eigenface (Wikipedia)](https://en.wikipedia.org/wiki/Eigenface)
- [Geek for Geeks Eigenface](https://www.geeksforgeeks.org/machine-learning/ml-face-recognition-using-eigenfaces-pca-algorithm/)
- [Power iteration (Wikipedia)](https://en.wikipedia.org/wiki/Power_iteration)
- [Rayleigh quotient (Wikipedia)](ps://en.wikipedia.org/wiki/Rayleigh_quotient)
- [Computation of matrix eigenvalues and eigenvectors, Oxford lecture slides](https://www.robots.ox.ac.uk/~sjrob/Teaching/EngComp/ecl4.pdf)
- 