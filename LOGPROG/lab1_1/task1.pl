% Своя реализация стандартных предикатов
length_list([], 0).   % Получение длины списка 
length_list([_|T], N):-
    length_list(T, TN), N is TN+1.

member_list(E, [E|_]).    % Проверка, входит ли элемент в список 
member_list(E,[_|T]):-
    member_list(E, T).

append_list([], L, L).    % Добавление списка L в конец списка  
append_list([H|T1], L, [H|T2]):-
    append_list(T1, L, T2).

remove_list([E|L], E, L). % Удаление элемента из списка
remove_list([H|T], E, [H|L]):-
    remove_list(T, E, L).

permute_list([], []).     % Перестановка списка 
permute_list([H1|T1], L):-
    permute_list(T1, L1), my_select(L, H1, L1).

prefix_list(L1, L2):-     % Проверка является ли список L1 префиксом L2 
    append_list(L1, _, L2).

sublist_list(L1, L2):-    % Проверка, входит ли список L1 в список L2 
	prefix_list(L1, L2).
sublist_list(L, [_|T]):-
    sublist_list(L, T).

% Предикаты обработки списков задание 1.1 (вариант 3)
% Удаление трех последних элементов

% без использования стандартных предикатов
remove_last3([_, _, _], []):-!.		% удаление последних 3-х в списке
remove_last3([H|T1], [H|T2]):-
    remove_last3(T1, T2).

% с использованием стандартных
sremove_last3(List, Result):-	% удаление последних 3-х в списке
    prefix(Result, List),
    length(List, Len),
    Res_len is Len - 3,
    length(Result, Res_len).

% Предикаты обработки списков задание 1.2 (вариант 8)
% Вычисление среднего арифметического элементов
sum_list([], 0).      % сумма чисел в списке 
sum_list([H|T], S):-
    sum_list(T, SS), S is SS + H.
% без использования стандартных предикатов
average_list(List, Avg):-    % среднее арифметическое для списка 
    sum_list(List, Sum),
    length_list(List, Cnt),
    Avg is Sum/Cnt.

% с использованием стандартных
saverage_list(List, Avg):-	% среднее арифметическое для списка
    sum_list(List, Sum), length(List, Cnt), Avg is Sum/Cnt.


% пример совместного использования двух предикатов
% перестановка послежних 3-х элементов списка в начало
swap3_list(List, Res):-
    remove_last3(List, Tmp1),
    append_list(Tmp1, Tmp2, List),
    append_list(Tmp2, Tmp1, Res).