:-consult("rules.pl").
:-consult("search.pl").
:-consult("input.pl").

% вызов экспертной системы
main:-
    read_input,
    direct_search,
    summarize.

:-main.
