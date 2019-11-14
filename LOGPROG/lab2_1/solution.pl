% В одном городе живут 7 любителей птиц. И фамилии у них птичьи.
% Каждый из них тезка птицы, которой владеет один из его товарищей.
% У троих из них живут птицы, которые темнее, чем пернатые "тезки" их хозяев.
% "Тезка" птицы, которая живет у Воронова, женат.
% Голубев и Канарейкин единственные холостяки из всей компании.
% Хозяин грача женат на сестре жены Чайкина.
% Невеста хозяина ворона очень не любит птицу, с которой возится ее жених.
% "Тезка" птицы, которая живет у Грачева, хозяин канарейки.
% Птица, которая является "тезкой" владельца попугая, принадлежит "тезке" той птицы,
% которой владеет Воронов.
% У голубя и попугая оперение светлое.
% Кому принадлежит скворец?

%список птиц
birds(L):-
    L = [voron, kanareyka, golub, grach, chayka, skvorec, popugay].

%   Тезки человек-птица
tezka(voronov, voron).
tezka(kanareykin, kanareyka).
tezka(golubev, golub).
tezka(grachev, grach).
tezka(chaykin, chayka).
tezka(skvorcov, skvorec).
tezka(popugaev, popugay).

solve(L):-
%    список друзей-птичников
%    o(X, Y) означает, что у X живет птица Y
    L = [o(voronov, A), o(kanareykin, B), o(golubev, C), o(grachev, D),
        o(chaykin, E), o(skvorcov, F), o(popugaev, G)],
    birds(Birds),
%    Птицы с темным пером
    DarkBirds = [voron, grach, skvorec],
    permutation(Birds, [A, B, C, D, E, F, G]),
%    Каждый из них тезка птицы, которой владеет один из его товарищей.
    A\=voron, B\=kanareyka, C\=golub, D\=grach, E\=chayka, F\=skvorec, G\=popugay,
%    У троих из них живут птицы, которые темнее, чем пернатые "тезки" их хозяев.
	member(o(X, XX), L), tezka(X, TX), member(XX, DarkBirds),
    not(member(TX, DarkBirds)),
	member(o(Y, YY), L), tezka(Y, TY), member(YY, DarkBirds),
    not(member(TY, DarkBirds)),
	member(o(Z, ZZ), L), tezka(Z, TZ), member(ZZ, DarkBirds),
    not(member(TZ, DarkBirds)),
    XX\=YY, YY\=ZZ, XX\=ZZ,
%    Голубев и Канарейкин единственные холостяки из всей компании.
    Bachelors = [kanareykin, golubev],
%    "Тезка" птицы, которая живет у Воронова, женат
    member(o(voronov, QQ), L), tezka(Q, QQ), not(member(Q, Bachelors)),
%    Хозяин грача женат на сестре жены Чайкина.
    member(o(V, grach), L), V\=chaykin,
%    Невеста хозяина ворона очень не любит птицу, с которой возится ее жених.
    member(o(W, voron), L), member(W, Bachelors),
%    "Тезка" птицы, которая живет у Грачева, хозяин канарейки.
    member(o(grachev, UU), L), tezka(U, UU), member(o(U, kanareyka), L),
%    Птица, которая является "тезкой" владельца попугая, принадлежит "тезке" 
%    той птицы, которой владеет Воронов.
    member(o(R, popugay), L), tezka(R, RR), member(o(voronov, TT), L),
    tezka(T, TT), member(o(T, RR), L).

who(X):-
	solve(L), member(o(X, skvorec), L).