% --- ������� � ����������� ���������� ��� ������ �� �������� --- 

my_length([], 0).   % ��������� ����� ������ 
my_length([_|T], N):-
    my_length(T, TN), N is TN+1.

my_member(E, [E|_]).    % ��������, ������ �� ������� � ������ 
my_member(E,[_|T]):-
    my_member(E, T).

my_append([], L, L).    % ���������� ������ L � ����� ������� 
my_append([H|T1], L, [H|T2]):-
    my_append(T1, L, T2).

my_remove([E|L], E, L). % �������� �������� E �� ������
my_remove([H|T], E, [H|L]):-
    my_remove(T, E, L).

my_select(L1, E, L2):-  % ����� �������� E �� ������ L1 � ��������� 
    my_remove(L1, E, L2).	

my_permute([], []).     % ��������� ������������ ������ 
my_permute([H1|T1], L):-
    my_permute(T1, L1), my_select(L, H1, L1).

my_prefix(L1, L2):-     % �������� �������� �� ������ L1 ��������� L2 
    my_append(L1, _, L2).

my_sublist(L1, L2):-    % ��������, ������ �� ������ L1 � ������ L2 
	my_prefix(L1, L2).
my_sublist(L, [_|T]):-
    my_sublist(L, T).

% --- ��� ������ 1.1 --- 

my_insert(E, 0, L, [E|L]):-!.   % ������� �������� � ������ �� N-� ������� (������� � 0) 
my_insert(E, N, [H|T], [H|L]):- % ��� ������������� ����������� ���������� 
    TN is N-1, my_insert(E, TN, T, L).

my_insert_std(E, N, L, R):- % � �������������� ����������� 
    prefix(PR, L),
    length(PR, N),
    append(PR, PO, L),
    append(PR, [E|PO], R).

% --- ��� ������ 1.2 --- 

my_first_negative([H|_], 0):-   % ����� ������ ��������� ������� �������������� �������� � ������ (������� � 0) 
    H<0, !.
my_first_negative([_|T], P):-
    my_first_negative(T, PP), P is PP+1.

my_first_negative_std(L, N):-   % � �������������� �����������
    member(X, L), X < 0, nth0(N, L, X), !.

% --- �������������� ������ ������������� ������ ���� ���������� �� ������ 3 � 4 ������� 1 --- 
my_change(E, N, L, LR):-    % ������ ��������, �������� �� N-�� ����� (������� � 0) � ������ L, �� ������� E 
    my_insert(X, N, Z, L), my_remove(L, X, Z), my_insert(E, N, Z, LR), !.
