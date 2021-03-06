# Отчет по лабораторной работе №1
## Работа со списками и реляционным представлением данных
## по курсу "Логическое программирование"

### студент: Куликов А.В.

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |
| Левинская М.А.|              |       5       |

## Введение

В отличии от императивных языков, для которых самым характерным способом хранения данных являются массивы, в Прологе основным способом являются списки.

Списки в Прологе - это рекурсивный тип данных, который представляет из себя по сути бинарное дерево, левый потомок которого является одиночным элементом - головой списка, правый - тоже дерево, левый потомок которого - это следующий элемент списка, а правый уже дерево, представляющее список без первых двух элементов, и так далее. Голова всегда является элементом, хвост является списком.

Именно поэтому все операции над списками реализуют при помощи рекурсии.

Массив из императивных языков же, в свою очередь, просто набор ячеек, расположеных в памяти подряд и последовательно.

Достоинства:

- не требует предварительного задания размеров;
- в теории имеет неограниченный размер;
- в реализациях Пролога без строгой типизации может хранить разнотипные объекты.

Недостатки:

- список в Прологе не имеет произвольного доступа к элементам.
  
В традиционных языках программирования наиболее похожими структурами являются односвязные списки т.к. тоже упорядочены, так же ссылаются только на следующий, но не на предыдущий элемент, имеют последоватедьный доступ, тоже потенциально бесконечны и т.д.

 Предикат обработки списка (Вариант 10)

## Задание 1.1: Реализация аналогов стандартных предикатов:

```prolog
my_length([], 0).   % Получение длина списка
my_length([_|T], N):-
    my_length(T, TN), N is TN+1.
```

```prolog
my_member(E, [E|_]).    % Проверка, входит ли элемент в список
my_member(E,[_|T]):-
    my_member(E, T).
```

```prolog
my_append([], L, L).    % Добавление списка L в конец первому
my_append([H|T1], L, [H|T2]):-
    my_append(T1, L, T2).
```

```prolog
my_remove([E|L], E, L). % Удаление элемента E из списка
my_remove([H|T], E, [H|L]):-
    my_remove(T, E, L).
```

```prolog
my_select(L1, E, L2):-  % Выбор элемента E из списка L1 с удалением
    my_remove(L1, E, L2).  
```

```prolog
my_permute([], []).     % получение перестановок списка
my_permute([H1|T1], L):-
    my_permute(T1, L1), my_select(L, H1, L1).
```

```prolog
my_prefix(L1, L2):-     % проверка является ли список L1 префиксом L2
    my_append(L1, _, L2).
```

```prolog
my_sublist(L1, L2):-    % проверка, входит ли список L1 в список L2
    my_prefix(L1, L2).
my_sublist(L, [_|T]):-
    my_sublist(L, T).
```

Как и было сказано выше, в основе всех предикатов для обработки списков лежит рекурсия.

## Задание 1.2: Предикат обработки списка (Вариант 10)

`my_insert(E, N, L, RL)` - предикат, вставляющий элемент E на N-ую позицию (начиная с 0) в список L и помещающий результат в RL.

Примеры использования:

```prolog
?- ?- my_insert(2, 4, [1, 3, 5, 7, 9, 11], X).
X = [1, 3, 5, 7, 2, 9, 11].
?- my_insert(3, 1, X, [1, 3, 2]).
X = [1, 2].
?- my_insert(E, 1, [1, 2], [1, 3, 2]).
E = 3.

?- my_insert_std(0, 2, [1,2,3,4,5], X).
X = [1, 2, 0, 3, 4, 5].

```

Реализация без использования стандартных предикатов:

```prolog
my_insert(E, 0, L, [E|L]):-!.
my_insert(E, N, [H|T], [H|L]):-
    TN is N-1, my_insert(E, TN, T, L).
```

Пока N больше нуля рекурсивно спускаемся, отсекая головы исходного списка и списка-результата. Далее, при N = 0, срабатывает условие окончания рекурсии и списку-результату без первых N элементов сопоставляется остаток исходного списка с добавленным в начало вставляемым элементом. Далее, в процессе обратного хода рекурсии, спискам "возвращаются" отсеченные элементы, причем на каждом шаге головой списка результата становится текущая голова исходного списка. Получаем список со вставленным на N-ю позицию элементом.

Реализация с использованием стандартных предикатов:

```prolog
my_insert_std(E, N, L, R):-
    prefix(PR, L),
    length(PR, N),
    append(PR, PO, L),
    append(PR, [E|PO], R).
```

Ищем префикс PR исходного списка, имеющий длину N, затем находим остаток списка PO (т.е. то, что нужно добавить к найденному префиксу, чтобы получить исходный список). Далее конкатенируем в таком порядке: PR|E|PO.

## Задание 1.3: Предикат обработки числового списка (Вариант 15)

`my_first_negative(L, N)` - предикат для нахождения позиции N (начиная с 0) первого отрицательного элемента в списке L.

Примеры использования:

```prolog
?- my_first_negative([1, 2, 3, -4, 5, -6, -7], N).
N = 3.
?- my_first_negative_std([1, 2, -3, 4, -5, 6], X).
X = 2.
```

Реализация без использования стандартных предикатов:

```prolog
my_first_negative([H|_], 0):-
    H<0, !.
my_first_negative([_|T], P):-
    my_first_negative(T, PP), P is PP+1.
```

Рекурсивно отсекаем голову списку, пока голова списка не окажется отртцательной, далее переменной P сопоставляется значение 0. В процессе обратного хода рекурсии на каждом к переменной P сопоставляется ее значение на предыдущем шаге увеличенное на 1. Получаем позицию первого отрицательного элемента в списке. Отсечение нужно для того, чтобы предикат возвращал позицию первого и только первого отрицательного элемента.

Реализация с использованием стандартных предикатов:

```prolog
my_first_negative_std(L, N):-
    member(X, L), X < 0, nth0(N, L, X), !.
```

Перебираем элементы списка, пока не найдем отрицательный, а затем с помощью стандартного предиката `nth0` находим его позицию.

Пример совместного использования предикатов, разработанных в п. 3, 4.

`my_change(E, N, L, LR)` - предикат для замены элемента, стоящего на N-ом месте (начиная с 0) в списке L, на элемент E.

Примеры использования:

```prolog
?- my_change(7, 0, [1, 2, 3], X).
X = [7, 2, 3].
?- my_change(X, 0, [1, 2, 3], [5, 2, 3]).
X = 5.
```

Реализация:

```prolog
my_change(E, N, L, LR):-
    my_insert(X, N, Z, L),
    my_remove(L, X, Z),
    my_insert(E, N, Z, LR), !.
```

Находим в списке L такой элемент X, что при его вставке на N-ю позицию в некий список Z получим сам список L. Т. о. Z - это список без элемента X, который стоял в исходном списке на N-й позиции. После вставляем на N-ю позицию в список Z новый элемент E. Получен список с замененным N-м элементом.

## Задание 2: Реляционное представление данных (Вариант 2-2)

В силу отсутствия опыта в работе с базами данных, видятся лишь следующие качества реляционного представления данных.

Преимущества:

- Простота и доступность для понимания пользователем. Единственной используемой информационной конструкцией является таблица;
- Существует мощный математический аппарат, который позволяет лаконично описывать необходимые операции над данными.

Недостатки:

- Иногда, для более оптимального хранения данных приходится создавать несколько отдельных таблиц затем связывая их. Это может приводить к трудностям понимания структуры данных;
- Низкая скорость доступа к данным при поиске по не ключевым полям.

Данные были представлены в виде отношения grade(GN, SN, SB, MK), где GN - номер группы, SN - фамилия студента, SB - предмет, MK - оценка.

Данное представление данных имеет следующие достоинства и недостатки.

Преимущества:

- Удобная обработка. Данное представление по сути является одной единственной таблицей, хранящей все интересующие нас данные. В следствии чего выборка происходит за один проход по данным.

Недостатки:

- Не самое компактное представление данных. Происходит дублирование значений отдельных атрибутов. К примеру, название каждой из дисциплин в данном представлении встречается 28 раз, тогда как в остальных представлениях по одному разу. При этом данное поле является самым объемным.

Задание:

- Напечатать средний балл для каждого предмета
- Для каждой группы, найти количество не сдавших студентов
- Найти количество не сдавших студентов для каждого из предметов

Стандартный предикат `findall(E, cond(E, ...), L)` формирует список L таких элементов E, для которых выполняется условие cond. Поэтому были созданы предикаты mark, bad_student, group, subject для выборки из базы знаний, с помощью которых получаем списки интересующих нас данных.

К примеру `bad_student(Grp, Std)` выполняется только для студентов группы Grp. Список же формируется из фамилий студентов Std.

 Далее, производя необходимые операции над списками при помощи нескольких подсобных предикатов sum_of, average, count_of получаем и печатаем таблицу-результат. Подробнее предназначение каждого предиката описано в коде (См. код task2.pl).

## Выводы

Списки в Прологе довольно мощный инструмент в умелых руках. Из-за вышеперечисленых достоинств списков программист может меньше думать о слежении за памятью, типизации и пр. и сосредоточиться на решении задачи.

При работе со списками из-за их структуры оказывается незаменимой рекурсия. Все операции над списками работают на основе отсечения головы списка и рекурсивных вызовов для того, что от него осталось. Так же выясняется, что если однажды удасться "обуздать" рекурсию, то многие вещи могут быть реализованы на основе уже решенных задач. Примеры этого были приведены выше.

Так же на руку играет двунаправленность работы предикатов в прологе. Предикат может как получить результат на основе исходных данных, так и, зная результат, при достаточности определенности переменных восстановить исходное данное. Это так же было использовано. Существуют, однако, некоторые ограничения из-за использования is т.к. левая часть должна быть определена до вычислений.

В ходе работы над данной лабораторной были разработаны собственные аналоги стандартных предикатов для работы со списками, и, как следствие,пришло понимание того, как они работают. На их примере и были усвоены основные методы работы со списками.

Так же были получены начальные представления о реляционной модели данных, ее достоинствах и недостатках. Познакомился с формированием запросов к данным на примере базы данных Пролога. Весьма вероятно, что в специализированных СУБД это делается совсем другими путями, но с чего-то же надо начинать.

После обучения программированию на традиционных языках программирования вроде СИ и C++ с непривычки было достаточно сложно переключиться на программирование в декларативном стиле. Из-за отсутствия опыта программирования на Прологе реализация даже простейших стандартных предикатов оказалась довольно нетривиальной задачей. Было над чем подумать.

Приобретенный опыт, несомненно, полезен. Как минимум он пригодится в дальнейшем изучении предмета.

Надеюсь, когда-нибудь доведется применить полученные знания на практике.
