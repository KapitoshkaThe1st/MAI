    % Place your solution here

% 9 вариант
% Один из пяти братьев разбил окно. Андрей сказал: Это или Витя, или Толя.
% Витя сказал: Это сделал не я и не Юра. Дима сказал: Нет, один из них сказал
% правду, а другой неправду. Юра сказал: Нет, Дима ты не прав. Их отец, которому,
% конечно можно доверять, уверен, что не менее трех братьев сказали 
% правду. Кто разбил окно? 

says(andrey, b, [vitya, tolya]).
says(vitya, e, [vitya, yura]).
%says(dima, p, andrey).
%says(dima, d, vitya).
%says(dima, p, vitya).

says(dima, b, [vitya, andrey, dima]).
%says(dima, b, [andrey, dima]).

%%says(dima, d, andrey).
says(yura, d, dima).


not_member(X, [Y]):-X \= Y.
not_member(X, [Y|T]):-
    X \= Y, not_member(X, T).

boys(L):-
    L = [andrey, vitya, dima, yura, tolya].

complement(S1, S2, S):-
    setof(X, (member(X, S1), not_member(X, S2)), S).

blame(X, L):-
    says(X, b, L).
blame(X, L):-
    says(X, e, L1), boys(B), complement(B, L1, L).
blame(X, L):-
    says(X, p, Z), blame(Z, L).
blame(X, L):-
    says(X, d, Z), blame(Z, L1), boys(B), complement(B, L1, L).

state(X, Y, t):-
    blame(X, L), member(Y, L).
state(X, Y, f):-
    blame(X, L), not_member(Y, L).

insert(E, 0, L, [E|L]):-!.   % вставка элемента в список на N-ю позицию (начиная с 0) 
insert(E, N, [H|T], [H|L]):- % без использования сторонних предикатов 
    TN is N-1, insert(E, TN, T, L).

solve(X):-
    member(N, [0, 1, 2, 3]), insert(f, N, [t, t, t], L), [A, B, C, D] = L,
    state(andrey, X, A),
    state(vitya, X, B),
    state(dima, X, C),
    state(yura, X, D).

solve(X):-
    state(andrey, X, t),
    state(vitya, X, t),
    state(dima, X, t),
    state(yura, X, t).