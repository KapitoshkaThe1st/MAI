limited_path([Cur|T], Cur, 1, [Cur|T], []). 
limited_path([Cur|T], Goal, N, Result, [Rel|TT]):-
    N > 0,
    move(Cur, Next, Rel),
    not(member(Next, [Cur|T])),
    N1 is N - 1,
    limited_path([Next, Cur|T], Goal, N1, Result, TT).

num_generator(1).
num_generator(X):-
    num_generator(XX),
    X is XX + 1. 

depth_limit(40).

iterative_deeping_search(Start, Finish, Path, Rel):-
    num_generator(N),
    depth_limit(Lim),
    (N > Lim, !, fail;
    limited_path([Start], Finish, N, Path, Rel)).

move(X, Y, father):-father(X, Y).
move(X, Y, mother):-mother(X, Y).
move(X, Y, brother):-brother(X, Y).
move(X, Y, sister):-sister(X, Y).
move(X, Y, son):-son(X, Y).
move(X, Y, daughter):-daughter(X, Y).

relative(X, Y, R):-
    iterative_deeping_search(X, Y, _, R), !.