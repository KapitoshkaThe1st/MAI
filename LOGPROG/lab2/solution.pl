 % Андрей утверждает, что виновен Витя или Толя
statement("Andrey", Perp):-
    Perp = "Vitya"; Perp = "Tolya".

% Витя утверждает, что ни Витя, ни Юра не виновны
statement("Vitya", Perp):-        
    Perp \= "Vitya", Perp \= "Yura".

% Дима утверждает, правду сказал только один из других мальчиков. Т.о. либо Андрей сказал правду,
% а Витя соврал, либо наоборот Андрей соврал, а Витя сказал правду.
statement("Dima", Perp):-         
    (statement("Andrey", Perp), not(statement("Vitya", Perp)));
    (statement("Vitya", Perp), not(statement("Andrey", Perp))).
    
% Юра сказал, что Дима соврал.
statement("Yura", Perp):-
    not(statement("Dima", Perp)).

% Получить список говоривших
speakers(S):-
    S = ["Andrey", "Vitya", "Dima", "Yura"].
% Получить список всех мальчиков
boys(B):-
    B = ["Andrey", "Vitya", "Dima", "Yura", "Tolya"].

% Проверить выполняется ли предположения всех людей в списке
truly([], _).
truly([H|T], Perp):-
   	statement(H, Perp), truly(T, Perp).

% Отец (которому точно можно доверять) сказал, что правду сказали не менее 3-х мальчиков,
% т.е. либо трое из говоривших, либо все четверо.
solve(Perp):-
    boys(B), member(Perp, B),
    speakers(S), member(Liar, S),
    delete(S, Liar, List),
    truly(List, Perp);
    speakers(S), truly(S, Perp).

:-solve(X), write(X), write(' broke the window.'), nl, halt.
% ОТВЕТ: окно разбил Толя.