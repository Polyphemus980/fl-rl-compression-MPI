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

1. Tworzymy tablicę `requiredBits` o dlugości `n`, gdzie `n` to długość tablicy `input`, która będzie zawierała informację o tym, ile bitów jest potrzebnych do zapisania danej wartości. W naszym przypadku to będzie:

   ```
   [1, 2, 1, 3, 3, 3, 4, 1, 4]
   ```

   Możemy ją utworzyć poprzez utworzenie threada dla każdego elementu inputu i obliczenia dla niego `32 - __clz(input[i])`, przy czym musimy uważać na przypadek gdy `input[i] == 0`, wtedy po prostu zwracamy 1.

2. Następnie chcemy obliczyć tablicę `outputBits`, która będzie długości `ceil(input.length / frame.length)`, czyli dokładnie tyle, ile będzie framów w finalnym wyniku. Wówczas `outputBits[i] = thrust::max_element(requiredBits[i * frame.Length], requiredBits[(i + 1) * frame.Length])`. Dodatkowo chcemy utworzyć zmienna globalną `outputValuesMinLength`, którą każdy thread (przy użyciu shared memory per block dla optymalizacji) będzie zwiększał o `outputBits[i] * frame.Length`.

3. Potrzebujemy obliczyć jeszcze tablicę `frameStartIndices`, która w naszym przypadku będzie postaci:

   ```
   [0, 2 * 3, 2 * 3 + 3 * 3] = [0, 6, 15]
   ```

   Aby ją utworzyć wystarczy:

   - utworzyć tablice `frameBitsLength` stworzonej poprzez `frameBitsLength[i] = outputBits[i] * frame.Length`
   - wywołać na tablicy `frameBitsLength` algorytm `Prescan`.
     (algorytm `Prescan` zaimplementowany zgodnie ze wskazówkami z wykładu)

4. Teraz chcemy utworzyć tablicę `outputValues`. Bedzię ona miała dlugość (w bajtach) `ceil(outputValuesMinlength / 8)`. Następnie tworzymy `n` threadów, gdzie `n` to liczba elementów tablicy `input`. Dla każdego threada wykonujemy następujące operacje:

   - obliczamy `frameId = i / frame.Length`
   - obliczamy `frameElementId = i % frame.Length`
   - obliczamy `requiredBits = outputBits[frameId]`
   - obliczamy `bitsOffset = frameStartIndices[frameId] + frameElementId * requiredBits`
   - obliczamy `outputId = bitsOffset / 32` (dzielimy przez 32, bo obliczone wartości są w bitach, a my mamy tablicę intów).
   - obliczamy `outputOffset = bitsOffset % 32` (jest to offset wewnątrz inta)
   - obliczamy `mask = (1 << requiredBits) - 1`
   - obliczamy `encodedValue = (input[i] & mask) << outputOffset`
   - zapisujemy wynik `atomicOr(output[outputId], encodedValue)`
   - Dodatkowo musimy rozpatrzeć przypadek, gdy wartość będzie rozbita na dwa sąsiadujące inty (czyli kiedy `outputOffset + requiredBits > 32`) - wtedy obliczamy `overflowValue = (encodedValue & mask) >> (32 - outputOffset)`
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
2. Tworzymy tablicę `output`, która będzie miała długość `frameLength * inputBits.Length` (bo każdy frame ma `frameLength` elementów, a framów jest tyle samo, co elementów w tablicy `inputBits`).
3. Uruchamiamy `n` threadów, gdzie `n` to długość tablicy `output`. Dla threada o indeksie `i` wykonujemy następujące operacje:

   - obliczamy `frameId = i / frame.Length`
   - obliczamy `frameElementId = i % frame.Length`
   - obliczamy `usedBits = inputBits[frameId]`
   - obliczamy `bitsOffset = frameStartIndices[frameId] + frameElementId * usedBits`
   - obliczamy `inputId = bitsOffset / 32`
   - obliczamy `inputOffset = bitsOffset % 32`
   - obliczamy `mask = (1 << usedBits) - 1`
   - obliczamy `decodedValue = (inputValues[inputId] >> inputOffset) & mask`
   - dodatkowo rozpatrujemy przypadek, gdy wartość była zapisana na dwóch sąsiednich elementach w tablicy (czyli kiedy zachodzi `inputOffset + usedBits > 32`) - wtedy obliczamy:
     - `overflowBits = inputOffset + usedBits - 32`
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
   - jesli `i < n - 1` to `outputValues[i] = input[startIndices[i]]` oraz `outputCount[i] = startIndices[i + 1] - startIndices[i]`
   - jeśli `i == n - 1` to `outputValues[n - 1] = input[startIndices[n-1]]` oraz `outputCount[n-1] = n - startIndices[n - 1]`

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
