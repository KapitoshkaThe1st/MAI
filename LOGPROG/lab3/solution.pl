% state([CLM, CLK],[CRM, CRK], WB) будет обозначать некое состояние,
% при котором на левом берегу находится CLM миссионеров и CLK каннибалов,
% на правом берегу CRM миссионеров и CRK каннибалов. Лодка находится на берегу WB
% (l -- левый берег, r -- правый). 

% Числа от 0 до 3
in_pool(X):-
    member(X, [0,1,2,3]).

% Проверка на возможность такого состояния. 
% Для того чтобы не было -5 каннибалов на берегу и т.п.
possible(state([X, Y], [V, W], _)):-
    in_pool(X),     % от 0 до 3
    in_pool(Y),
    in_pool(V),
    in_pool(W),
    SM is X + V, SM = 3,    % в сумме 3 миссионера
    SK is Y + W, SK = 3, !. % в сумме 3 каннибала

% Проверка на корректность состояния. 
% Для того чтобы каннибалы не съели миссионеров.
correct(state([X, Y], [V, W], _)):-     % миссионеров должно быть не меньше, чем каннибалов
    X >= Y, V >= W, !.
correct(state([0, _], [_, _], _)):-!.   % либо, если на одном из берегов миссионеров нет,
                                        % то им ничего не угрожает
correct(state([_, _], [0, _], _)).

% переправа с левого на правый берег
move(state([X, Y], [V, W], l), state([XX, YY], [VV, WW], r)):-
   	in_pool(CM),    % не больше 3-х миссионеров в лодку
    in_pool(CK),    % не больше 3-х каннибалов в лодку
    (CM >= CK; CM = 0),     % в лодке либо не должно быть миссионеров совсем,
                            % либо должно быть не меньше, чем каннибалов
    S is CM + CK,   % и в сумме
    S =< 3,         % в лодке не больше 3-х пассажиров,
    S > 0,          % но и пустая лодка, не может уплыть
                    % (должен быть минимум один в качестве гребца)
    XX is X - CM,   % CM миссионеров уплыло
    YY is Y - CK,   % CK каннибалов уплыло
    VV is V + CM,   % CM миссионеров приплыло
    WW is W + CK,   % CK каннибалов приплыло
   	possible(state([XX, YY], [VV, WW], r)),     % проверка на возможность состояния
    correct(state([XX, YY], [VV, WW], r)).      % проверка на безопасность состояния

% Переправа с правого на левый берег
% (просисходит ровно то же самое, только с/на другой берег)
move(state([X, Y], [V, W], r), state([XX, YY], [VV, WW], l)):-
   	in_pool(CM),
    in_pool(CK),
    (CM >= CK; CM = 0),
    S is CM + CK, 
    S =< 3, 
    S > 0,
    XX is X + CM, 
    YY is Y + CK,
    VV is V - CM, 
    WW is W - CK, 
   	possible(state([XX, YY], [VV, WW], l)),
    correct(state([XX, YY], [VV, WW], l)).

% Распечатка списка в столбик по одному элементу
print_list(L):-
    member(E, L),
    write(E), nl,
    fail.
print_list(_).

% Поиск в ширину
b_path([[Cur|T]|_], Cur, [Cur|T]).  % если конец текущего пути -- целевое состояние, то путь найден
b_path([[Cur|T]|TT], Goal, Result):-    % для первого пути в очереди
    setof(                              % получаем список продолженных  путей
        [Next, Cur|T],
    	(move(Cur, Next), not(member(Next, [Cur|T]))),
    	New
    ),
    append(TT, New, RR), !,     % добавляем его в конец очереди, удалив уже обработанный путь
    b_path(RR, Goal, Result);   % продолжаем поиск
    b_path(TT, Goal, Result).   % если текущий путь -- тупик, и у него нет продолжений,
                                % то ищем продолжения для следующих в очереди

breadth_first_search(Start, Finish, Path, Time):-   
	get_time(Start_time),   % замер времени выполнения для анализа
    b_path([[Start]], Finish, Reversed_path),   % находим путь (список состояний получается инвертированным)
    reverse(Reversed_path, Path),               % переворачиваем чтобы получить в прямом порядке
    get_time(Finish_time), 
    Time is Finish_time - Start_time.

% Поиск в глубину
path([Cur|T], Cur, [Cur|T]).    % если конец текущего пути -- целевое состояние, то путь найден
path([Cur|T], Goal, Result):-   
    move(Cur, Next),                    % продляем путь смежной с последней вершиной в графе состояний
    not(member(Next, [Cur|T])),         % такой, которую еще не посетили
    path([Next, Cur|T], Goal, Result).  % продолжаем поиск

depth_first_search(Start, Finish, Path, Time):-
	get_time(Start_time),   % замер времени выполнения для анализа
    path([Start], Finish, Reversed_path),   % находим путь (список состояний получается инвертированным)
    reverse(Reversed_path, Path),           % переворачиваем чтобы получить в прямом порядке
    get_time(Finish_time), 
    Time is Finish_time - Start_time.

% Поиск пути с ограниченной длинной
limited_path([Cur|T], Cur, 1, [Cur|T]).     % если конец текущего пути -- целевое состояние, то путь найден
                                            % 1 для предотвращения повторной генерации путей меньшей длины
limited_path([Cur|T], Goal, N, Result):-
    N > 0,
    move(Cur, Next),                % продляем путь смежной с последней вершиной в графе состояний
    not(member(Next, [Cur|T])),     % такой, которую еще не посетили
    N1 is N - 1,                    % уменьшаем ограничение
    limited_path([Next, Cur|T], Goal, N1, Result).  % продолжаем искать
% Т.о. будет найден путь длины не больше N, если он существует 

% Генератор натуральных чисел начиная с 1
num_generator(1).
num_generator(X):-
    num_generator(XX),
    X is XX + 1. 

depth_limit(40).    % ограничение для глубины поиска с итеративным заглублением

iterative_deeping_search(Start, Finish, Path, Time):-
	get_time(Start_time), 
    num_generator(N),   % получаем очередное значение максимальной глубины
                        % каждый раз на 1 болшее предыдущего
    depth_limit(Lim),   % получаем предел глубин
    (N > Lim, !, fail;  % если достигли предела, то прекращаем искать
    limited_path([Start], Finish, N, Reversed_path),    % иначе ищем очередной путь
    reverse(Reversed_path, Path)),                      % переворачиваем чтобы получить в прямом порядке
    get_time(Finish_time), 
    Time is Finish_time - Start_time.

% Перевод из пары состояний, в действие, приведшее к переходу
make_action(state([X,Y], _, _), state([XX,YY], _, Dest), A):-
    CM is abs(X - XX),  % получаем число перевезенных миссионеров
    CK is abs(Y - YY),  % и число перевезенных каннибалов
    A = carry([CM, CK], Dest).    % составляем структуру 
% carry([CM, CK], Dest) означает, что в результате перехода
% перевезено CM миссионеров и CK каннибалов на берег Dest.
% Например, carry([2,1],l) -- перевезено 2 миссионера и 1 каннибал на левый берег.


% Перевод списка состояний в список соответствующих действий
action_sequence([State1, State2], [Act]):-
    make_action(State1, State2, Act).
action_sequence([State1, State2|T], [Act|TT]):-
    action_sequence([State2|T], TT),
    make_action(State1, State2, Act).

% Вывод информации для анализа работы поисков
analysis:-
    write('\tДлина пути\tВремя выполнения'), nl,
	depth_first_search(state([3,3],[0,0],l),state([0,0],[3,3],r), DFS_path, DFS_time),
	breadth_first_search(state([3,3],[0,0],l),state([0,0],[3,3],r), BFS_path, BFS_time),
	iterative_deeping_search(state([3,3],[0,0],l),state([0,0],[3,3],r), ID_path, ID_time),
    length(DFS_path, DFS_len),
    length(BFS_path, BFS_len),
    length(ID_path, ID_len),
    write('DFS\t'), write(DFS_len), write('\t\t'), write(DFS_time), nl,
    write('BFS\t'), write(BFS_len), write('\t\t'), write(BFS_time), nl,
    write('ID\t'), write(ID_len), write('\t\t'), write(ID_time), nl, !.
% Начальное состояние: 3 миссионера, 3 каннибала и лодка на левом берегу
% Конечное состояние: 3 миссионера, 3 каннибала и лодка на правом берегу

% Вывод решения задачи
solution:-
    % ищем кратчайший путь при помощи поиска в ширину
    breadth_first_search(state([3,3],[0,0],l),state([0,0],[3,3],r), BFS_path, _), 
    write('Лучший способ переправиться:'), nl,
    % переводим цепочку состояний в цепочку действий для соответствующих переходов
    action_sequence(BFS_path, Acts), print_list(Acts),
    % находим количество действий
    length(Acts, Acts_count),
    write('Включает '), write(Acts_count), write(' действий.'), !.