:- ['two.pl'].

% --- Подсобные предикаты --- 
sum_of([], 0).      % сумма чисел в списке 
sum_of([H|T], S):-
    sum_of(T, SS), S is SS+H.

average(List, Avg):-    % среднее арифметическое для списка 
    sum_of(List, Sum), length(List, Cnt), Avg is Sum/Cnt.

count_of(_, [], 0):-!.  % подсчет вхождений элемента в списке 
count_of(E, [H|T], C):-
     H==E, count_of(E, T, CC), C is CC+1.
count_of(E, [_|T], C):-
     count_of(E, T, C), !.

% --- Подсчет значений для конкретной группы/дисциплины --- 
count_of_bad_students_by_group(Grp, Cnt) :-
    bad_students(Grp, L), length(L, Cnt).

count_of_bad_students_by_subject(Sbj, Cnt):-
    marks_by_subject(Sbj, L), count_of(2, L, Cnt).

average_grade_by_subject(Sbj, Gr):-
    marks_by_subject(Sbj, L), average(L, Gr).

% --- Предикаты для выборки из базы знаний --- 
mark(Sbj, Mk):-    % оценок 
    grade(_, _, Sbj, Mk).

bad_student(Grp, Std):-     % не сдавших экзамен 
    grade(Grp, Std , _, 2).

group(Grp):-          % групп 
    grade(Grp, _, _, _).

subject(Sbj):-        % предметов 
    grade(_, _, Sbj, _).


% --- Получение списка... --- 
bad_students(Grp, List) :-      % не сдавших студентов в группе 
    setof(Std, bad_student(Grp, Std), List).

marks_by_subject(Sbj, List):-   % оценок по предмету 
    findall(Mk, mark(Sbj, Mk), List).

groups(List):-      % групп 
    setof(Grp, group(Grp), List).

subjects(List):-    % предметов 
    setof(Sbj, subject(Sbj), List).

% --- Предикаты для прохода и печати по всем значениям --- 

print_average([]).      % печать среднего балла по предмету для списка предметов 
print_average([H|T]):-
    average_grade_by_subject(H, Avg), 
    write('\t'), write(H), write(' -- '), write(Avg), nl, print_average(T).

print_count_by_g([]).       % печать количества не сдавших по группе для списка групп 
print_count_by_g([H|T]):-
    count_of_bad_students_by_group(H, Cg),
    write('\t'), write(H), write(' -- '), write(Cg), nl, print_count_by_g(T).

print_count_by_s([]).       % печать количества не сдавших по дисциплине для списка дисциплин 
print_count_by_s([H|T]):-
    count_of_bad_students_by_subject(H, Cs),
    write('\t'), write(H), write(' -- '), write(Cs), nl, print_count_by_s(T).

result:-
    write('1) Средний балл по экзаменам:'), nl,
    subjects(Subjects), print_average(Subjects),
    write('2) Количество не сдавших студентов по группам:'), nl,
    groups(Groups), print_count_by_g(Groups),
    write('3) Количество не сдавших студентов по предметам:'), nl,
    print_count_by_s(Subjects).
    
:-result, halt.