% ���� ���������� ����������� ����������
length_list([], 0).   % ��������� ����� ������ 
length_list([_|T], N):-
    length_list(T, TN), N is TN+1.

member_list(E, [E|_]).    % ��������, ������ �� ������� � ������ 
member_list(E,[_|T]):-
    member_list(E, T).

append_list([], L, L).    % ���������� ������ L � ����� ������  
append_list([H|T1], L, [H|T2]):-
    append_list(T1, L, T2).

remove_list([E|L], E, L). % �������� �������� �� ������
remove_list([H|T], E, [H|L]):-
    remove_list(T, E, L).

permute_list([], []).     % ������������ ������ 
permute_list([H1|T1], L):-
    permute_list(T1, L1), my_select(L, H1, L1).

prefix_list(L1, L2):-     % �������� �������� �� ������ L1 ��������� L2 
    append_list(L1, _, L2).

sublist_list(L1, L2):-    % ��������, ������ �� ������ L1 � ������ L2 
	prefix_list(L1, L2).
sublist_list(L, [_|T]):-
    sublist_list(L, T).

% ��������� ��������� ������� ������� 1.1 (������� 3)
% �������� ���� ��������� ���������

% ��� ������������� ����������� ����������
remove_last3([_, _, _], []):-!.		% �������� ��������� 3-� � ������
remove_last3([H|T1], [H|T2]):-
    remove_last3(T1, T2).

% � �������������� �����������
sremove_last3(List, Result):-	% �������� ��������� 3-� � ������
    prefix(Result, List),
    length(List, Len),
    Res_len is Len - 3,
    length(Result, Res_len).

% ��������� ��������� ������� ������� 1.2 (������� 8)
% ���������� �������� ��������������� ���������
sum_list([], 0).      % ����� ����� � ������ 
sum_list([H|T], S):-
    sum_list(T, SS), S is SS + H.
% ��� ������������� ����������� ����������
average_list(List, Avg):-    % ������� �������������� ��� ������ 
    sum_list(List, Sum),
    length_list(List, Cnt),
    Avg is Sum/Cnt.

% � �������������� �����������
saverage_list(List, Avg):-	% ������� �������������� ��� ������
    sum_list(List, Sum), length(List, Cnt), Avg is Sum/Cnt.


% ������ ����������� ������������� ���� ����������
% ������������ ��������� 3-� ��������� ������ � ������
swap3_list(List, Res):-
    remove_last3(List, Tmp1),
    append_list(Tmp1, Tmp2, List),
    append_list(Tmp2, Tmp1, Res).