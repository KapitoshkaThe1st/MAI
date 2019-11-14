% facts database
:-['data.pl'].
% rules
:-['rules.pl'].
% relative chain search 
:-['relative_search.pl'].
% input
:-['input.pl'].

:- dynamic user/1.
:- dynamic last/1.

last('Алексей Куликов').
user('Алексей Куликов').

qst(user, [Actor], im) --> [i], [am], [Actor], ['.'].
qst(Keyword, [Actor], wi) --> [who, is], [Actor], [Keyword], [?].
qst(Keyword, [Actor], wa) --> [who, are], [Actor], [Keyword], [?].
%qst(Keyword, [Actor], Y) --> [who], aux_v(X), {X = s, Y = wi; X = m, Y = wa} , [Actor], [Keyword], [?].
qst(Keyword, [Actor], hm) --> [how, many], [Keyword], ([does]; [do]), [Actor], [have], [?]. 
qst(Keyword, [Actor], ws) --> [whose], [Keyword], aux_v(s), [Actor], [?].
%qst(Keyword, [Actor1, Actor2], is) --> [is], [Actor1], [Actor2], [Keyword], [?]. 
qst(Keyword, [Actor1, Actor2], is) --> aux_v(s), [Actor1], [Actor2], [Keyword], [?].
%qst(Keyword, [Actor1, Actor2], is) --> aux_v(s), [Actor1], [Actor2], [Keyword], [?].
qst(_, [Actor1, Actor2], re) --> [who], aux_v(s), [Actor1], [to], [Actor2], [?].

% aux(i) --> [am].
aux_v(s) --> ([is];[am]).
aux_v(m) --> [are].

find_answer(X, Y, wi, A):-
    append([X, A], Y, R),
    P=..R,
    call(P).

find_answer(X, Y, wa, L):-
    append([X, A], Y, R),
    P=..R,
    setof(A, P, L).

find_answer(X, Y, hm, N):-
    append([X, A], Y, R),
    P=..R,
    setof(A, P, L), !,
    length(L, N).
find_answer(_, _, hm, 0).

find_answer(X, [Y], ws, A):-
    append([X, Y], [A], R),
    P=..R,
    call(P).

find_answer(X, Y, is, yes):-
    append([X], Y, R),
    P=..R,
    call(P),!.
find_answer(_, _, is, no).

find_answer(_, Y, im, ok):-
    [H] = Y,
    renew_user(H).

find_answer(_, Z, re, A):-
    [X, Y] = Z,
    relative(X, Y, A).

request(Qst, Ans):-
    phrase(qst(Pred, Act, Cls), Qst),
    change(Act, R),
    find_answer(Pred, R, Cls, Ans),
    [H|_] = Act,
    renew_last(H).

change([], []).
change([X|T], [L|TT]):-
    change(T, TT),
    member(X, [she, he, his, her]), last(L),!.
change([X|T], [U|TT]):-
   	change(T, TT),
    member(X, [i, my, me]), user(U), !.
change([X|T], [X|TT]):-
    change(T, TT).

renew_user(X):-
    retract(user(_)),
    asserta(user(X)).

renew_last(X):-
    not(member(X, [i, me, my, she, he, his, her])), !,
    retract(last(_)),
    asserta(last(X)).
renew_last(_).

main:-
    write('Hello!'), nl,
    repeat,
    write('What do you want to know?'), nl,
    pr(X),
    (
        X = [goodbye],! , write('Bye! See you soon!');
        request(X, A),
        write(A), nl, fail
    ).
% :-main, halt.