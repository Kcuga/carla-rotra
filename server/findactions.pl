:- include('UKrotr.pl').
% :- include('UKrotrSmall.pl').
:- include('utils.pl').

getRecommendedActions([RName|Tail], Context, B, I, A, R) :-
	rule( RName, Context, Br, Ir, Ar),
	((subset(Br, B), subset(Ir, I)) -> append(Ar, A, Ao); Ao = A),
	getRecommendedActions(Tail, Context, B, I, Ao, R).
	
getRecommendedActions(RName, Context, B, I, A, R) :-
	rule(RName, Context, Br, Ir, Ar),
	((subset(Br, B), subset(Ir, I)) -> append(Ar, A, Ao); Ao = A),
	getRecommendedActions([], Context, B, I, Ao, R).
	
getRecommendedActions([], _, _, _, A, Ao) :- Ao = A.	
	
getRecommendedActions(Context, B, I, R) :-
	findall(X, rule(X,Context,_,_,_), L),
	getRecommendedActions(L, Context, B, I, [], A),
	map(separateAction, A, R).

getAllActions(Context, BaseActionsSet) :-
	findall(Action, rule(_, Context, _, _, Action), R),
	flatten(R, ActionsList),
	list_to_set(ActionsList, ActionsSet),
	map(getBaseAction, ActionsSet, BaseActions),
	list_to_set(BaseActions, BaseActionsSet).

getAllBeliefs(Context, Beliefs) :-
	findall(Belief, rule(_, Context, Belief, _, _), BeliefsList),
	flatten(BeliefsList, BeliefsSet),
	list_to_set(BeliefsSet, Beliefs).

getAllIntentions(Context, Intentions) :-
	findall(Intention, rule(_, Context, _, Intention, _), IntentionsList),
	flatten(IntentionsList, IntentionsSet),
	list_to_set(IntentionsSet, Intentions).