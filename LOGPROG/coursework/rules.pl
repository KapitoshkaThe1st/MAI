%--RULES--

male(X):-
	ifather(X,_),!.
female(X):-
	imother(X,_),!.

child(X, Y):-
	parent(Y, X).

son(X, Y):-
	child(X, Y),
	male(X).

daughter(X, Y):-
	child(X, Y),
	female(X).

father(X, Y):-
	ifather(X, Y),
	Y \== nobody.

mother(X, Y):-
	imother(X, Y),
	Y \== nobody.

parent(X, Y):-
	mother(X, Y);
	father(X, Y).

grandmother(X, Y):-
	mother(X, Z), 
	(mother(Z, Y); father(Z, Y)).

grandfather(X, Y):-
	father(X, Z),
	(mother(Z, Y); father(Z, Y)).

grandson(X, Y):-
	grandfather(Y, X);
	grandmother(Y, X),
	male(X).

granddaughter(X, Y):-
	grandfather(Y, X);
	grandmother(Y, X),
	female(X).

sibling(X, Y):-
	(father(Z, X),father(Z, Y)),
	X\==Y.

brother(X, Y):-
	sibling(X, Y),
	male(X).

sister(X, Y):-
	sibling(X, Y),
	female(X).

marrieds(X, Y):-
	father(X, Z),
	mother(Y, Z).

husband(X, Y):-
	marrieds(X, Y),
	male(X).

wife(X, Y):-
	marrieds(X, Y),
	female(X).

brotherinlaw(X, Y):-  % деверь
	husband(Z, Y),
	brother(X, Z).