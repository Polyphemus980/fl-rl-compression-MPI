# Plan implementacji

_Kacper Trzciński_

## Fixed-Lenght

### Kompresja

Posłużymy się prostym przykładem do wizualizacji sposobu:

- input: **[0, 2, 1, 5, 5, 7, 10, 1, 13]**
- outputBits: **[2, 3, 4]**
- outputValues (binarnie): **00_10_01---101_101_111---1010_0001_1101**

Zakładamy, że w tym przypadku frame ma wielkość 3 (w finalnym projekcie będzie to albo wartość możliwa do ustawienia podczas uruchomienia programu, albo inna większa wartość).
`outputBits` oznacza liczbę bitów potrzebną do zakodowania danych w framie, `outputValues` zawiera dane już zakodowane, gdzie kodowanie w naszym przypadku to po prostu ucinanie nieznaczących zer. W przykładowym outpucie `---` oznacza przejście do nowego frame'a (jest ono wprowadzone tylko w celu łatwiejszego odczytania przykładu). Dodatkowo, zapis binarny wyniku nie jest w takiej postaci, w jakiej będzie to faktycznie zapisane w pamięci - jest to raczej przedstawienie, w jaki sposób chcemy daną liczbę zakodować. Nie przedstawiam faktycznego obrazu pamięci jako, że uważam, ze zaciemni to jedynie obraz i ogólny koncept algorytmu.

1. Tworzymy tablicę `outputBits`, która będzie długości `ceil(input.length / framelength)`, czyli dokładnie tyle, ile będzie framów w finalnym wyniku. Tworzymy threada dla każdego elementu tablicy `input` i na każdym z nich wyliczamy minimalną liczbę potrzebnych bitów do reprezentacji danej wartości. Obliczony wynik przekazujemy do funkcji `atomicMax`, jako drugi argument podając `outputBits[j]`, gdzie `j = i / frameLength`.

2. Potrzebujemy obliczyć jeszcze tablicę `frameStartIndices`, która w naszym przypadku będzie postaci:

   ```
   [0, 2 * 3, 2 * 3 + 3 * 3] = [0, 6, 15]
   ```

   Aby ją utworzyć wystarczy:

   - utworzyć tablice `frameBitsLength` stworzonej poprzez `frameBitsLength[i] = outputBits[i] * frameLength`
   - wywołać na tablicy `frameBitsLength` algorytm `Prescan`.

3. Teraz chcemy utworzyć tablicę `outputValues`. Bedzię ona miała dlugość (w bajtach) `ceil(outputValuesMinBitLength / 8)`, gdzie `outputValuesMinBitLength = frameStartIndicies.last + lastFrameElementsCount * outputBits.last`, natomiast `lastFrameElementsCount = frameLength` gdy `input.Length % frameLength = 0`, wpp `lastFrameElementsCount = input.Length - (input.Length / frameLength) * frameLength`.

   Następnie tworzymy `n` threadów, gdzie `n` to liczba elementów tablicy `input`. Dla każdego threada wykonujemy następujące operacje:

   - obliczamy `frameId = i / frame.Length`
   - obliczamy `frameElementId = i % frame.Length`
   - obliczamy `requiredBits = outputBits[frameId]`
   - obliczamy `bitsOffset = frameStartIndices[frameId] + frameElementId * requiredBits`
   - obliczamy `outputId = bitsOffset / 8`
   - obliczamy `outputOffset = bitsOffset % 8` (jest to offset wewnątrz bajta)
   - obliczamy `encodedValue = input[i] << outputOffset` (maska nie jest potrzebna, bo i tak najstarsze bity są wyzerowane)
   - zapisujemy wynik `atomicOr(output[outputId], encodedValue)`
   - Dodatkowo musimy rozpatrzeć przypadek, gdy wartość będzie rozbita na dwa sąsiadujące inty (czyli kiedy `outputOffset + requiredBits > 8`) - wtedy obliczamy `overflowValue = encodedValue >> (8 - outputOffset)` (maska niepotrzebna z tego samego powodu co wyżej)
     i zapisujemy wynik w kolejnym elemencie poprzez `atomicOr(output[outputId + 1], overflowValue)`

   Oczywiście operacje atomiczne robimy najpierw na shared memory a dopiero później przepisujemy wyniki do pamięci globalnej, jednak dla czytelności opisu nie rozpisywałem tych szczegółów.

### Dekompresja

Posłużymy się prostym przykładem do wizualizacji sposobu:

- inputBits: **[2, 3, 4]**
- inputValues (binarnie): **00_10_01---101_101_111---1010_0001_1101**
- output: **[0, 2, 1, 5, 5, 7, 10, 1, 13]**

  Przy czym zachodzą te same uwagi, które były we wstępie do `Kompresji Fixed-Length` odnośnie zapisu przykładu.

Kolejne kroki:

1. Tworzymy tablicę `frameStartIndices` (tak samo jak w pkt.3 kompresji Fixed-length, z tą różnicą, że używamy `inputBits` zamiast `outputBits`).
2. Tworzymy tablicę `output`, która będzie miała długość `outputSize` zakodowaną w pliku podczas kompresji (więc mamy ją od razu za darmo).
3. Uruchamiamy `n` threadów, gdzie `n` to długość tablicy `output`. Dla threada o indeksie `i` wykonujemy następujące operacje:

   - obliczamy `frameId = i / frame.Length`
   - obliczamy `frameElementId = i % frame.Length`
   - obliczamy `usedBits = inputBits[frameId]`
   - obliczamy `bitsOffset = frameStartIndices[frameId] + frameElementId * usedBits`
   - obliczamy `inputId = bitsOffset / 8`
   - obliczamy `inputOffset = bitsOffset % 8`
   - obliczamy `mask = (1 << usedBits) - 1`
   - obliczamy `decodedValue = (inputValues[inputId] >> inputOffset) & mask`
   - dodatkowo rozpatrujemy przypadek, gdy wartość była zapisana na dwóch sąsiednich elementach w tablicy (czyli kiedy zachodzi `inputOffset + usedBits > 8`) - wtedy obliczamy:
     - `overflowBits = inputOffset + usedBits - 8`
     - `overflowMask = (1 << overflowBits) - 1`
     - `overflowValue = inputValues[inputId + 1] & overflowMask << (usedBits - overflowBits)`
     - `decodedValue |= overflowValue`
   - tak otrzymany wynik zapisujemy w finałowej tablicy `output[i] = decodedValue`

   Podobnie jak w przypadku kompresji, używamy shared memory do optymalizacji odczytywania danych z pamięci.

## Run-Length

### Kompresja

Posłużymy się prostym przykładem do wizualizacji sposobu:

- input: **[5, 5, 8, 8, 8, 7, 7, 7, 7, 3, 4, 4, 4]**
- outputCount: **[2, 3, 4, 1, 3]**
- outputValues: **[5, 8, 7, 3, 4]**

Kolejne kroki:

1. Utworzenie tablicy `startMask`, która jest długości takiej samej jak dane wejściowe, i `startMask[i] = 1` wtedy i tylko wtedy, gdy `input[i]` jest początkiem nowej sekwencji do zakodowania. W naszym przypadku tablica `startMask` będzie miała postać:

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

4. Nasze wynikowe tablice chcemy traktować jako `uint8_t`, co oznacza, że w przypadku gdy długość sekwencji przekroczy `255` musimy daną sekwencję rozbić na części (np. w wypadku `256` tych samych symboli pod rząd utworzymy dwie sekwencje - jedna z `255` elementami i druga z `1` elementem, obydwie odnoszące się do tej samej wartości). Aby to zrobić, najpierw musimy sprawdzić czy taka sytuacja zachodzi poprzez:

   - utworzenie tablicy `recalculateSequence` o dlugości o `1` mniejszej niż dlugość `startIndices`
   - utworzenie zmiennej globalnej `bool shouldRecalcuate` (używając shared memory dla poprawy wydajności)
   - odpalenie threada dla każdego elementu tablicy `recalculateSequence`
   - jeśli `startIndices[i + 1] - startIndices[i] > 255` to zmieniamy flagę oznaczającą potrzebę poprawy na true (tzn ustawiamy `shouldRecalculate` na `true` za pomocą operacji atomicznej i dodatkowo ustawiamy `recalculateSequence[i] = (startIndices[i + 1] - startIndices[i]) / 255`).

   Zauważmy, że jeśli `shouldRecalculate = false` to możemy od razu przejść do ostatniego punktu.
   W przeciwnym przypadku, musimy utworzyć dodatkowe sekwencje poprzez:

   - uruchomienie algorytmu `Prescan` na tablicy `recalculateSequence` i zapisanie wyniku w tablicy `recalculatePrescan`
   - odpalenie `n` threadów, gdzie `n = recalculatePrescan.last + recalculate.Seqence.last`
   - w czasie `O(log(n))` (za pomocą binary searcha) `i`-ty thread jest w stanie znaleźć takiej `j`, że zachodzi
     ```
     recalculatePrescan[j] <= i < recalculatePrescan[j + 1]
     ```
   - obliczamy `k = i - recalculatePrescan[j]`
   - edytujemy `startMask` poprzez
     ```
     startMask[startIndices[j] + k * 255] = 1
     ```

   Po wykonaniu tych kroków wracamy do punktu `2` tym razem wiedząc, że nie będzie już przypadku z sekwencją dłuższą niż `255`.
   Pomysł na to może wydawać się nieco "na około", jednak przypadek z sekwencją dłuższą niż `255` zakładam, że jest raczej rzadki, stąd główna idea była taka, żeby rozważyć to jak najmniejszym kosztem w przypadku, gdy takiej sekwencji nie ma.

5. Na koniec używamy przygotowanych tablic do obliczenia finalnego wyniku. Tworzymy threada dla każdego elementu tablicy `startIndices` i tworzymy tablicę output, poprzez (niech `i` - numer threada oraz `n` - liczba threadow);
   - jesli `i < n - 1` to `outputValues[i] = input[startIndices[i]]` oraz `outputCount[i] = startIndices[i + 1] - startIndices[i]`
   - jeśli `i == n - 1` to `outputValues[n - 1] = input[startIndices[n-1]]` oraz `outputCount[n-1] = input.Length - startIndices[n - 1]`

### Dekompresja

Posłużymy się prostym przykładem do wizualizacji sposobu:

- inputCount: **[3, 2, 1, 2]**
- inputValues: **[8, 9, 2, 4]**
- output: **[8, 8, 8, 9, 9, 2, 4, 4]**

Kolejne kroki:

1. Używamy algorytmy `Prescan` do utworzenia tablicy `startIndices`. W naszym przypadku wynik będzie postaci:

   ```
   [0, 3, 5, 6]
   ```

   Algorytm zaimplementowany zgodnie ze slajdami z wykładu. Zauważmy, że ta tablica zawiera indeksy początków kolejnych sekwencji.
   Dodatkowo zauważmy, że `startIndices.Last()` + `inputCount.Last()` daje nam łączną liczbę wszystkich elementów (czyli długość tablicy `output`).

2. Tworzymy `n` threadów, gdzie `n` to długość tablicy `output`. Dla `i`-tego threada jesteśmy w stanie w czasie `O(log(n))` (używając binary searcha) znaleźć `j` takie, że zachodzi

   ```
   startIndices[j] <= i < startIndices[j + 1]
   ```

   przy czym dla `j == startIndices.length - 1` przyjmujemy `startIndices[j + 1] = inf`. Zauważmy, że wówczas `output[i] = inputValues[j]`, co kończy algorytm.
