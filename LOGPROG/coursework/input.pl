strings_to_atoms([], []):-!.
strings_to_atoms([H|T], [R|TT]):-
    split_string(H, "", "\s\t\n", [RR]),
    atom_string(R, RR),
    strings_to_atoms(T, TT).

sss([],[]).
sss([H], [R]):-
    atom_string(X, H),
    split_string(X, " ", "", L),
    strings_to_atoms(L, R).
sss([H, HH|T], [R, HH|TT]):-
    atom_string(X, H),
    split_string(X, " ", "", L),
    strings_to_atoms(L, R),
    sss(T, TT).

linearize([], []).
linearize([H|T], RR):-
    is_list(H), linearize(H, R),
    linearize(T, TT), !, append(R, TT, RR).
linearize([H|T], [H|TT]):-
    linearize(T, TT).

pr(X):-
    read_string(user_input, "\n", "\r", _, String),
    split_string(String, "'", "", R),
    strings_to_atoms(R, RR),
    sss(RR, RRR),
    linearize(RRR, X), !.