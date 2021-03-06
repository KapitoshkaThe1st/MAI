% Предложение состоит из подлежащего и подпредложения (назовем это так)
sentence(X) --> obj(Y), sub(Y, X).

% Подпредложение состоит из группы глагола
sub(Y, X) --> verb_g(Y, X).
% либо из группы глагола, информация из которой нас интересует, и еще одного подпредложения
sub(Y, X) --> verb_g(Y, X), sub(_, _).
% либо из группы глагола, и еще одного подпредложения, информация из которого нас интересует
sub(Y, X) --> verb_g(_, _), sub(Y, X).

% группа глагола состоит из союза(опционально) и словосочетания с каким-то субъектом и сказуемого "любит"
verb_g(X, likes(X, Y)) --> (pr_g; []), likes(Y).
% либо из союза и сказуемого "не любит"
verb_g(X, not_likes(X, Y)) --> (pr_g; []), not_likes(Y).

% частица с запятой
pr_g --> [","], (["но"]; ["а"]).

% Обьект -- подлежащее
obj(X) --> [X].
% Субьект -- дополнение
subj(X) --> [X].

% группа субьектов состоит либо из субьекта, который нас интересует и субьекта, который не интересует
subj_g(X) --> subj(X), ["и"], subj(_).
% либо из субьекта, который нас не интересует и субьекта, который интересует
subj_g(X) --> subj(_), ["и"], subj(X).
% либо из одиночного субьекта
subj_g(X) --> subj(X).

% словосочетания с каким-то субъектом сказуемым "любит"
likes(X) --> ["любит"], subj_g(X).
% словосочетания с каким-то субъектом сказуемым "не любит"
not_likes(X) --> ne, ["любит"], subj_g(X).

% частица не
ne --> ["не"].

% Предикат для вычленения всех атомарных глубинных структур вида likes(X, Y) и not_likes(X, Y), где X -- объект, субьект, из предложения.
decompose(X, Y):-
    setof(Z, phrase(sentence(Z), X), Y).

% Пример:
% decompose(["Саша", "любит", "кубики", "и", "шарики", ",", "но",
% "не", "любит", "ролики", "и", "квадратики",",", "а", "любит", "штуковины"], X).
% decompose(["Саша", "любит", "игрушки", ",", "но", "не", "любит", "кубики", "и", "мячи"], X).
% 
% decompose(["Ира", "не", "любит", "стихи", "и", "прозы", ",", "а", "любит", "пьессы" ], X). 