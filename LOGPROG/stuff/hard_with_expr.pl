 
get_ops(0, []):-!.
get_ops(N, [H|T]):-
    NN is N - 1,
    member(H, ['+','-','*','/']),
    get_ops(NN, T).

insert_ops(L, R):-
    length(L, N),
    NN is N - 1,
    get_ops(NN, O),
    append(O, L, R).

to_expr([X], X):-!.
to_expr([H|T], E):-
    append(T1, T2, T),
    to_expr(T1, E1),
    to_expr(T2, E2),
    member(H, ['+','-','*','/']),
    E=..[H, E1, E2].

can_be_zero(L, E):-
    insert_ops(L, T),
    permutation(T, TT),
    to_expr(TT, E),
    0 is E.
   
