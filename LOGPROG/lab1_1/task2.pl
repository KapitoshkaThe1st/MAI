:- ['three.pl'].

% Предикаты обработки списков задание 2 (вариант 3)
% Для каждого студента, найти средний балл, и сдал ли он экзамены или нет
% Для каждого предмета, найти количество не сдавших студентов
% Для каждой группы, найти студента (студентов) с максимальным средним баллом

length_list([], 0).  
length_list([_|T], Len):-
    length_list(T, Tmp), Len is Tmp + 1.

sum_list([], 0).      % сумма чисел в списке 
sum_list([H|T], S):-
    sum_list(T, SS), S is SS + H.
    
average_list(List, Avg):-    % среднее арифметическое для списка 
    sum_list(List, Sum),
    length_list(List, Cnt),
    Avg is Sum/Cnt.

max_list([X], X):-!.    % поиск максимального элемента в списке
max_list([H|T], H):-
    max_list(T, MM), H > MM.
max_list([H|T], MM):-
    max_list(T, MM), H =< MM.

stud_grade(St, Grd):-    % оценка студента
    student(_, St, GrdList), member(grade(_, Grd), GrdList).  

stud_grades(St, Grds):-  % оценки студента
    findall(X , stud_grade(St, X), Grds).

stud_avg_grade(St, Avg):-    % средний балл студента
    stud_grades(St, Grds), average_list(Grds, Avg).

w_stud_achievs(St):-    % распечатать достижения студента
    stud_grades(St, Grds), average_list(Grds, Avg),
    write(St), write('`s средний балл: '), write(Avg), 
    (member(2, Grds), write(' Не сдал :('),!); write(' Сдал! :)').

bad_stud(Sbj_sn, St):-   % проверка, является ли студент двоечником
    student(_, St, Grds),
    member(grade(Sbj_sn, 2), Grds).

subj_sh_name(Sbj_sn, Sbj):-    % получение полного названия предмета по сокращенному
    subject(Sbj_sn, Sbj).

bad_studs(Sbj, Sts):-    % не сдавшие предмет студенты
    subj_sh_name(Sbj_sn, Sbj),
    setof(X, bad_stud(Sbj_sn, X), Sts).

w_not_pass_cnt(Sbj):-   % распечатка количества несдавших предмет
    bad_studs(Sbj, Sts), length_list(Sts, Cnt),
    write(Sbj), write(' не сдали '), write(Cnt), write(' студента.').

gr_studs(Grp, Sts):-  % список студентов в группе
    findall(X, student(Grp, X, _), Sts).

gr_avgs(Grp, Avgs):- % средние баллы в группе
    findall(Avg, (gr_studs(Grp, Sts), member(St, Sts), stud_avg_grade(St, Avg)), Avgs).

best_stud(Grp, Bst):-    % лучший(-ие) студент группы
    gr_studs(Grp, Sts), gr_avgs(Grp, Avgs), max_list(Avgs, Max),
    member(Bst, Sts), stud_avg_grade(Bst, Avg), Avg = Max.

w_best_stud(Grp):-   % распечатать лучших студентов в группе
    best_stud(Grp, Bst),
    write('В группе '), write(Grp), write(' лучший студент -- '), write(Bst).

grp(Grp):-    % выбор группы
    student(Grp, _, _).

grps(Grps):-  % получение списка групп
    setof(X, grp(X), Grps).

print_avg_grades:-  % распечатка средних баллов всех студентов
    student(_, X, _),
    w_stud_achievs(X), nl,
    fail.
print_avg_grades.

print_bad_studs_count:-  % распечатка по предметам количество несдавших
    subject(_, X), 
    w_not_pass_cnt(X), nl,
    fail.
print_bad_studs_count.

print_best_studs:-   % распечатка лучших студентов для всех групп
    grps(Grps), member(Grp, Grps), 
    w_best_stud(Grp), nl,
    fail.
print_best_studs.

result:-    % объединение целей
    print_avg_grades,
    print_bad_studs_count,
    print_best_studs.

:-result, halt. % основная цель