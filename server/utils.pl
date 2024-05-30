strings_to_predicates([],[]).
strings_to_predicates([String|Rest], [Predicate|RestPredicates]) :-
    atom_string(Predicate,String),
    strings_to_predicates(Rest, RestPredicates).

map(_, [], []).
map(Goal, [X|Xs], [Y|Ys]) :-
    call(Goal, X, Y),
    map(Goal, Xs, Ys).

separateAction(Action, _{modifier: Modifier, action: BaseAction}) :-
    compound_name_arguments(Action, _, [Modifier, BaseAction]).

getBaseAction(Action, BaseAction) :-
    compound_name_arguments(Action, _, [_, BaseAction]).

getModifier(Action, Modifier) :-
    compound_name_arguments(Action, _, [Modifier, _]).