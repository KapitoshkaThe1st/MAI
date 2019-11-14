natural_number(0).
natural_number(X):- natural_number(Y), X is Y+1.

dump_db:-
  natural_number(X),
  write(X), nl,
  fail.
dump_db.

insert_facts:-
	member(X, ['lol', 'kek', '4eburek']),
	P=..[fact, X],
	assert(P),
  fail.
insert_facts.

do_smth(X, T, M):-
    functor(_, X, 1),
    P=..[X,Y],
    call(P),
    O=..[T,P],
    call(O), M,
    fail.
do_smth(_, _, _).

range(Max, Max, [Max]):-!.
range(Min, Max, [Min|T]):-
    Min1 is Min + 1,
    range(Min1, Max, T).

print_facts:-
  fact(X),
  write(X), nl,
  fail.
print_facts.

% class(X, zer):-
%     X >= 0,
%     X =< 0, !.
% class(X, pos):-
%     X >= 0, !.
% class(_, neg).

class(X, pos):-
    X > 0, !.
class(0, zer):-!.
class(_, neg).

% split_by_sign([H], [H], []):-
% 	  H >= 0, !.
% split_by_sign([H], [], [H]):-!.
% split_by_sign([H|T], [H|T1], T2):-
%     split_by_sign(T, T1, T2),
%     H >= 0, !.
% split_by_sign([H|T], T1, [H|T2]):-
%     split_by_sign(T, T1, T2).

% проверка на унифицируемость
unifiable_l([H], E, [H]):-
    my_not(my_not(E = H)), !.
unifiable_l([_], _, []).
unifiable_l([H|T], E, [H|TT]):-
    unifiable_l(T, E, TT),
    my_not(my_not(E = H)), !.
unifiable_l([_|T], E, TT):-
    unifiable_l(T, E, TT).
    
my_not(C):-
    C, !, fail;
    true.

split_by_sign([H], [H], []):-
	H >= 0.
split_by_sign([H], [], [H]):-
    H < 0.
split_by_sign([H|T], [H|T1], T2):-
    split_by_sign(T, T1, T2),
    H >= 0.
split_by_sign([H|T], T1, [H|T2]):-
    split_by_sign(T, T1, T2),
    H < 0.