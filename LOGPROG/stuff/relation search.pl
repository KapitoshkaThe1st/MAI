mother(ann, vlad).
mother(marry, ann).
mother(marry, robert).

father(vlad, no).
father(robert, no).
father(john, vlad).
father(mark, ann).
father(mark, robert).

male(X):-
    father(X,_).
female(X):-
    mother(X,_).

parent(X, Y):-
    father(X, Y);mother(X, Y).

brother(X, Y):-
    father(Z, X), father(Z, Y),
    male(X).

sister(X, Y):-
    father(Z, X), father(Z, Y),
    female(X).

son(X, Y):-
    parent(Y, X), male(X).

daughter(X, Y):-
    parent(Y, X), female(X).

limited_path([Cur|T], Cur, 1, [Cur|T], []).     % если конец текущего пути -- целевое состояние, то путь найден
                                            % 1 для предотвращения повторной генерации путей меньшей длины
limited_path([Cur|T], Goal, N, Result, [Rel|TT]):-
    N > 0,
    move(Cur, Next, Rel),                % продляем путь смежной с последней вершиной в графе состояний
    not(member(Next, [Cur|T])),     % такой, которую еще не посетили
    N1 is N - 1,                    % уменьшаем ограничение
    limited_path([Next, Cur|T], Goal, N1, Result, TT).  % продолжаем искать
% Т.о. будет найден путь длины не больше N, если он существует 

num_generator(1).
num_generator(X):-
    num_generator(XX),
    X is XX + 1. 

depth_limit(40).

iterative_deeping_search(Start, Finish, Path, Rel):-
    num_generator(N),   % получаем очередное значение максимальной глубины
                        % каждый раз на 1 болшее предыдущего
    depth_limit(Lim),   % получаем предел глубин
    (N > Lim, !, fail;  % если достигли предела, то прекращаем искать
    limited_path([Start], Finish, N, Path, Rel)).    % иначе ищем очередной путь

move(X, Y, father):-father(X, Y).
move(X, Y, mother):-mother(X, Y).
move(X, Y, brother):-brother(X, Y).
move(X, Y, sister):-sister(X, Y).
move(X, Y, son):-son(X, Y).
move(X, Y, daughter):-daughter(X, Y).


relative(X, Y, R):-
    iterative_deeping_search(X, Y, _, R). 
