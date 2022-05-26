# AvitoTest
## Задача 1
### Реализация
В качестве фичаэкстрактора была выбрана предобученная модель DeepPavlov/distilrubert-base-cased-conversational c HuggingFace (https://huggingface.co/DeepPavlov/distilrubert-base-cased-conversational).
Поверх него два фулли конектед слоя и один выходной нейрон. Модель была дообучена на половине train. Обучающая выборка составила примерно 30к уникальных батчей по 32 предложения. Предложением явлалсь склейка description и title.
Так как ограничение модели по длине предложения в 512 токенов, То нужноо производить обрезку предложения, т.к. быыла выдвинуто предположение, что контактная информация чаще всего содержится в конце объявления и возможно в title, то нужно было сохранить эту информацию.
Поэтому обрезка происходила так: [101] * (len(tokens) > 512) + tokens[-511:], то есть последние min(len(tokens), 512) токенов из description + title.
Интересно, что при использовании шедулеров, в какой-то момент лосс резко увеличивался, поэтому обучение проиходило без шедулеров.
Модель выдает вероятность пренадлежности к 1 классу.
### Метрика
Средний ROC-AUC по категориям на валидации составил 0.977
## Задача 2
#### Как происходит классификация: 
Прогоняется все предложение, с последнего 6-ого слоя берется скрытое представление токена начала предложения и подается в FC.
#### Идея на основе классификации:
Будем подавать не только скртыое представление начала предложения, но и все токены, тогда по предположению мы возьмем токены, которые дают наибольшую вероятность 1 класса.
#### Проблемы идеи:
Если подавать скрытые представление токенов из позитивного сэмпла, то они все выдают большую вероятность. Подход не работает
#### Идея перебором:
Можно подавать участки s[left:right], где left - левая граница рассматриваемого участка, right - правая. Тогда мы хотим найти - argmax(bert_clf(s[left:right])).
#### Проблемы идеи:
Даже для одного текста сложность работы только перебора - O(len(string)^2), что очень много. Перебирать токены - тоже очень долго и в ограничения не влезет.
#### Идея векторов Шепли:
Вектора шепли основаны на идеи, что каждый участник коллективной игры вносит свой вклад в результат. В нашем случае игра - классификатор, игроки - токены. Тогда существуют методы найти вклад каждого токена.
#### Проблемы идеи:
Не смог реализовать, а имеющиеся реализации не получилось имплементировать.
#### Наивная реализация (Используется)
Брать выходы классификатора, понятно, что там, где классификатор выдает вероятность меньше TRASHHOLD (вычислен на основе roc-auc), то там ,вероятно, нет контактной информации, поэтому там будем выдавать None.
Для тех сэмплов, где мы предсказали 1 класс, запустим регулрку по поиску номеров и почт. Регулярка что-то нашла - победа, иначе выводим 0, len(sample)

## Сложности
1. Обучить модельку и получить веса. Не хватало памяти либо Colab вылетал в самый неподходящий момент. В начале сильно текла память.
2. Запустить докер, немного пришлось переписать под macos
3. Придумать решение для 2-ой задачи
4. работа с Git... Случайно добавил веса модели, а они весят больше 100 мб.

## Впечатление
Задача интересная, над второй задачей думал долго и не смог реализовать очень хорошо с текущими ограничениями. Понравилось, что это почти симуляция прода.





