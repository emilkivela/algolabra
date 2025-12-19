# Testausdokumentti

## Kattavuusraportti

![image](./coverage.png)

## Yksikkötestit
Yksikkötesteissä on käytetty pytest-kirjastoa. Testit ovat kirjoitettu src/services/-kansion tiedostoille eigenfaces.py, formulas.py ja utils.py, koska ne sisältävät ohjelman toimintalogiikan joka ei koske käyttöliittymää.

### formulas.py
- power_iteration(): Testataan että löytää suurinta ominaisarvoa vastaavan vektorin. Syötteenä pieni testimatriisi josta oikeellisuus on helppo varmistaa
- rayleighs_quotient(): Testataan että löytää annetttua ominaisvektoria vastaavan ominaisarvon. Syötteenä pieni matriisi.

### utils.py
- load_dataset_faces(): Testataan, että muuntaa syötetyt kuvat matriisiksi, ja matriisi on oikean muotoinen. Syötteenä kaksi pientä kuvina toimivaa matriisia, jotka tallenettaan väliaikaisiksi tiedostoiksi

### eigenfaces.py
- get_eigs(): Testataan että löytää oikeat ominaisarvot. Syötteenä pieni matriisi josta oikeellisuus on helppo varmistaa.
- get_input_weight(): Testataan, että syötekuvan painovektori lasketaan oikein. Syötteenä pienet vektorit ja matriisit kuvina ja ominaiskasvoina, jotta oikeellisuus on helppo varmistaa
- calculate_eigenfaces(): Testataan, että palauttaa oikean keskiarvovektorin ja oikean muotoisen matriisin. Syötteenä pieni matriisi
- recognise_input_face(): Testataan, että löytää pienimmän etäisyyden päässä olevan testisyötteistä löytyvän henkilön. Syötteenä kolme pientä vektoria jotka vastaavat henkilöiden kuvia
- load_input_face(): Testataan, että muuntaa syötekuvan oikein. Syötteinä sekä tiedostopolku että FileObject
- get_training_weights(): Testataan, että harjoitusdatan painovektorit lasketaan oikein. Kuvat ja ominaiskasvot pieniä vektoreita, jotta oikeat painovektorit on helppo laskea