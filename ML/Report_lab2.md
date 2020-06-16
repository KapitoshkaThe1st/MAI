# Лабораторная работа №2

## ФИО, Группа
Куликов Алексей Владимирович, М8О-308Б-17

## Подготовка данных

Для корректной работы алгоритмов категориальные признаки необходимо привести к числовым. Это было сделано с использованием one-hot encoding'а. Числовые же признаки необходимо отнормировать и центрировать.

## Алгоритмы

Логистиеская регрессия реализована самым стандартным способом.

KNN так же вполне стандартен, и реализован с использованием обычной Евклидовой нормы.

При реализации решающего используется критерий разбиения Джини, построение дерева заканчивается при достижении максимальной глубины. Предусмотрена возможность выбора числа используемых признаков для использования в алгоритме случайного леса. При обучении используется бутстрапированная выборка.

Алгоритм случайного леса реализован на основе алгоритма решающего дерева. При этом используется беггинг и метод случайных подпространств. Беггинг заложен в использовании деревом бутстрапированной выборки, и выбор числа используемых признаков для метода случайных подпространств предусмотрен так же в алгоритме решающего дерева. В качестве предсказываемого класс берется самый встречающийся.

## Метрики
Для оценки качества классификации использованы метрики: accuracy, precision, recall. Метрики подсчитываются на кросс-валидации по всей выборке. 

##  Результаты

Наихудшим для обоих задач оказался алгоритм логистической регрессии.

KNN моей реализации почему-то превзошел по качеству библиотечный на первом датасете, и сопоставим на втором.

Качество работы решающего самодельного и библиотечного решающего дерева идентичны на обоих датасетах.

Библиотечная реализация случайного леса незначительно превосходит самодельную по качеству на обоих датасетах.

Все самодельные алгоритмы значительно уступают по скорости соответствующим алгоритмам из библиотеки sklearn. Это обусловленно однопоточной реализацией, и, возможно, не самыми оптимальными методами, использованными внутри них.

На обоих датасетах при выбранных параметрах алгоритмов видно, что модели не переобучились т.к. разница в точности классификации на обучающей и тестовой выборках ничтожно мала.

# Возникшие проблемы
Основные трудности, как и полагается, состояли в понимании алгоритма и последующей его реализации.
Так же довольно неприятным оказался тот факт, что после удаления нескольких записей из датасета номера оставшихся остаются прежними. Тогда при разбиении на категориальные и числовые признаки, нормировании числовых и кодировании категориальных и последующей "склейке" обратно датасет наполняется null значениями. Решением оказался сброс индекса датасета после удаления записей.

# Выводы

В результате работы над реализацией алгоритмов машинного обучения стало понятно, что делать это стоит исключительно в образовательных целях. Не стоит "изобретать велосипед", когда существуют общедоступные реализации алгоритмов, значительно превосходящие самодельные.

В целом, качеством классификации я остался доволен, и считаю, что для данных задач данные алгоритмы, хоть и в разной степени, но подходят все. 