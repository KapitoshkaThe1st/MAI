% предикат для получения списка входных фактов
inputs(L):-
    bagof(X, input(X), L).

% считывание входных данных
read_input:-
    inputs(I),
    print_inputs(I),
    read_string(user_input, "\n", "\r", _, X),
    split_string(X, " ", " ", Y),
    fill_base(I, Y).

% заполнение базы знаний
fill_base(_, []):-!.
fill_base(I, ["не", H|T]):-
    number_string(N, H),
    nth1(N, I, F),
    asserta((fact(F, f):-!)),
    fill_base(I, T), !.
fill_base(I, [H|T]):-
    number_string(N, H),
    nth1(N, I, F),
    asserta((fact(F, t):-!)),
    fill_base(I, T).

% распечатка перечня входных фактов
print_inputs_aux([], _).
print_inputs_aux([H|T], N):-
    format("~d. ~s?~n", [N, H]),
    N1 is N+1,
    print_inputs_aux(T, N1).

print_inputs(L):-
    print_inputs_aux(L, 1).