# Plan implementacji

_Kacper Trzciński_

## Fixed-Lenght

### Kompresja

TODO

### Dekompresja

TODO

## Run-Length

### Kompresja

Posłużymy się prostym przykładem do wizualizacji sposobu:

- input: **[5, 5, 8, 8, 8, 7, 7, 7, 7, 3, 4, 4, 4]**
- output: **[(5, 2), (8, 3), (7, 4), (3, 1), (4, 3)]**

Kolejne kroki:

1. Utworzenie tablicy _startMask_, która jest długości takiej samej jak dane wejściowe, i `startMask[i] = 1` wtedy i tylko wtedy, gdy `input[i]` jest początkiem nowej sekwencji do zakodowania. W naszym przypadku tablica `startMask` będzie miała postać:

   ```
   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
   ```

   Uruchamiamy thread dla każdego elementu tablicy `input`. Aby otrzymać `startMask`, wystarczy:

   - dla `i = 0` zawsze `startMask[0] = 1`
   - dla `i > 0` `startMask[i] = 1` wtw gdy `input[i] != input[i-1]`

   Jako, że i-ty thread zapisuje dane tylko do i-tego elementu tablicy `input` to nie ma problemu z brakiem synchronizacji.

2. Uruchomienie algorytmu `Scan` na tablicy `startMask`, tworząc `scannedStartMask`. W naszym przypadku otrzymamy:

   ```
   [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5]
   ```

   Implementacja algorytmu `Scan` zgodnie z wskazówkami ze slajdów z wykładu. Zauważmy, że dla każdej sekwencji wartość w tabliy `scannedStartMask` oznacza finałową pozycje (powiększona o 1). Np. dla sekwencji siódemek `scannedStartMask` zawiera wartość 3, i faktycznie w naszym outpucie 7 jest zapisana w `output[2]`.

3. Następnie chcemy obliczyć tablicę `startIndices`, która będzie zawierała indeksy kolejnych początków sekwencji do zakodowania. W naszym przypadku byłaby to tablica postaci:

   ```
   [0, 2, 5, 9, 10]
   ```

   Uruchamiamy threada dla każdego elemetnu `scannedStartMask` i obliczamy `startIndices` w następujący sposób (`i` oznacza numer threada):

   - dla `i = 0` zawsze zachodzi `startIndices[0] = 0`
   - dla `i > 0` jeśli `scannedStartMask[i] != scannedStartMask[i - 1]` to oznacza, że i jest początkiem nowej sekwencji i wówczas `startIndices[scannedStartMask[i] - 1] = i`

4. Na koniec używamy przygotowanych tablic do obliczenia finalnego wyniku. Tworzymy threada dla każdego elementu tablicy `startIndices` i tworzymy tablicę output, poprzez (niech `i` - numer threada oraz `n` - liczba threadow);
   - jesli `i < n - 1` to `output[i] = (input[startIndices[i]], startIndices[i + 1] - startIndices[i])`
   - jeśli `i == n - 1` to `output[n - 1] = (input[startIndices[n-1]], n - startIndices[n - 1])`

### Dekompresja

TODO
