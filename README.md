# Przetwarzanie Języka Naturalnego - zadanie 2

Model klasyfikujący napisany z wykorzystaniem biblioteki PyTorch Lightning. 
Modelem jest MLP przyjmujące na wejściu średnią embeddingów słów w tekście.


## Trenowanie 

Trenowanie modelu uruchamiamy skryptem `train_model`:
```
python -m src.scripts.train_model \
  --data_dir ${MULTIEMO_HOTEL_DATA_DIR} \
  --word_embedding_model_dir ${WORD_EMB_MODEL_PATH} \
  --word_embedding_type ${WORD_EMB_TYPE} \
  --input_size ${INPUT_SIZE} \
  --hidden_size ${HID_SIZE} \
  --output_size ${OUT_SIZE} \
  --dropout ${DROPOUT} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --weight_decay ${WD} \
  --eval ${EVAL}
```
gdzie wymagane jest by podać:
 - `data_dir` - lokalizacja, gdzie mieszczą się pliki `hotels.sentence.train.pl.txt`, 
   `hotels.sentence.dev.pl.txt`, `hotels.sentence.test.pl.txt`,
 - `word_embedding_model_dir` - ścieżka do pliku modelu word embedding
 - `word_embedding_type` - typ modelu word embeddingu, możliwe wartości to `fasttext` lub `word2vec,
Pozostałe argumenty nie są wymagane, gdyż mogą przyjąć wartości domyślne:
 - `input_size` - wymiar wejściowy modelu, równy wymiarowi word embeddingu, domyślnie: 100,
 - `hidden_size` - rozmiar warstwy ukrytej, domyślnie: 32,
 - `output_size` - rozmiar wyjścia równy liczbie klasy, domyślnie: 4,
 - `dropout` - prawdopodobieństwo dropoutu, domyślnie: 0.01,
 - `batch_size` - rozmiar batcha podczas trenowania, domyślnie: 32,
 - `learning_rate` - współczynnik uczenia, domyślnie: 0.005,
 - `weight_decay` - współczynnik _weight decay_, domyśłnie: 0.0001,
 - `eval` - wartość `True` bądź `False` wskazująca, czy po trenowaniu wykona się
    ewaluacja modelu na zbiorze testowym, domyślnie `False`.

Przykładowe wywołanie
```
python -m src.scripts.train_model \
  --data_dir data/multiemo2 \
  --word_embedding_model_dir data/fastext/model_dim100.bin \
  --word_embedding_type fasttext \
  --eval True
```

W wyniku trenowania modelu w folderze `models` zapisze się checkpoint modelu, hiperparametry w pliku
`hp.json` oraz logi tensorboarda z trenowania.

## Ewaluacja

Zapisany model można ewaluować przy pomocy skryptu:
```
python -m src.scripts.test_model \
  --model_dir ${MODEL_DIR}
  --data_dir ${MULTIEMO_HOTEL_DATA_DIR} \
```
gdzie wymagane jest by podać:
 - `data_dir` - lokalizacja, gdzie mieszczą się pliki `hotels.sentence.train.pl.txt`, 
   `hotels.sentence.dev.pl.txt`, `hotels.sentence.test.pl.txt`; do ewaluacji zostanie wykorzystany zbiór
    `hotels.sentence.test.pl.txt`,
 - `model_dir` - lokalizacja, gdzie znajdują się pliki `model.chkpt` and `hp.json` dla wytrenowanego modelu.

Przykładowe wywołanie
```
python -m src.scripts.test_model \
  --model_dir models/MLP-fasttext-run-1 \
  --data_dir data/multiemo2
```
W wyniku ewaluacji w lokalizacji modelu zapisze się plik `test_results.json` z miarami jakości
oraz `confusion_matrix.pdf` z obrazkiem macierzy pomyłek.

Przy trenowaniu ustawienie falgi `eval` na `True` podowuje automatyczną ewaluację na zbiorze testowym, 
więc nie trzeba wywoływać skryptu ewaluacyjnego jeszcze raz.

