% --- Аналоги к стандартным предикатам для работы со списками --- 

my_length([], 0).   % Получение длина списка 
my_length([_|T], N):-
    my_length(T, TN), N is TN+1.

my_member(E, [E|_]).    % Проверка, входит ли элемент в список 
my_member(E,[_|T]):-
    my_member(E, T).

my_append([], L, L).    % Добавление списка L в конец первому 
my_append([H|T1], L, [H|T2]):-
    my_append(T1, L, T2).

my_remove([E|L], E, L). % Удаление элемента E из списка
my_remove([H|T], E, [H|L]):-
    my_remove(T, E, L).

my_select(L1, E, L2):-  % Выбор элемента E из списка L1 с удалением 
    my_remove(L1, E, L2).	

my_permute([], []).     % получение перестановок списка 
my_permute([H1|T1], L):-
    my_permute(T1, L1), my_select(L, H1, L1).

my_prefix(L1, L2):-     % проверка является ли список L1 префиксом L2 
    my_append(L1, _, L2).

my_sublist(L1, L2):-    % проверка, входит ли список L1 в список L2 
	my_prefix(L1, L2).
my_sublist(L, [_|T]):-
    my_sublist(L, T).

% --- Для задачи 1.1 --- 

my_insert(E, 0, L, [E|L]):-!.   % вставка элемента в список на N-ю позицию (начиная с 0) 
my_insert(E, N, [H|T], [H|L]):- % без использования стандартных предикатов 
    TN is N-1, my_insert(E, TN, T, L).

my_insert_std(E, N, L, R):- % с использованием стандартных 
    prefix(PR, L),
    length(PR, N),
    append(PR, PO, L),
    append(PR, [E|PO], R).

% --- Для задачи 1.2 --- 

my_first_negative([H|_], 0):-   % поиск номера вхождения первого отрицательного элемента в списке (начиная с 0) 
    H<0, !.
my_first_negative([_|T], P):-
    my_first_negative(T, PP), P is PP+1.

my_first_negative_std(L, N):-   % с использованием стандартных
    member(X, L), X < 0, nth0(N, L, X), !.

% --- Содержательный пример использования вместе двух предикатов из пункта 3 и 4 задания 1 --- 
my_change(E, N, L, LR):-    % Замена элемента, стоящего на N-ом месте (начиная с 0) в списке L, на элемент E 
    my_insert(X, N, Z, L), my_remove(L, X, Z), my_insert(E, N, Z, LR), !.
