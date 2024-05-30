:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_error)).
:- use_module(library(http/http_json)).
:- use_module(library(http/json_convert)).
:- use_module(library(http/json)).
:- use_module(library(http/http_parameters)).
:- use_module(library(http/http_client)).
:- use_module(library(http/http_header)).

:- include('findactions.pl').

% Define the API routes

:- http_handler(root(.), get_actions, [method(patch)]).
:- http_handler(root(actions), get_all_actions, [method(get)]).
:- http_handler(root(beliefs), get_beliefs, [method(get)]).
:- http_handler(root(intentions), get_intentions, [method(get)]).

% Define the handler for the /actions route
get_all_actions(Request) :-
    http_parameters(Request, [context(Context, [atom, default(standard)])]),
    getAllActions(Context, Actions),
    reply_json_dict(Actions).


% Define the handler for the /beliefs route
get_beliefs(Request) :-
    http_parameters(Request, [context(Context, [atom, default(standard)])]),
    getAllBeliefs(Context, Beliefs),
    reply_json_dict(Beliefs).


% Define the handler for the /intentions route
get_intentions(Request) :-
    http_parameters(Request, [context(Context, [atom, default(standard)])]),
    getAllIntentions(Context, Intentions),
    reply_json_dict(Intentions).


% Define the handler for the / route
get_actions(Request) :-
    http_read_json_dict(Request, Json),

    atom_string(Context, Json.get(context)),
    strings_to_predicates(Json.get(intention), Intention),
    strings_to_predicates(Json.get(belief), Belief),

    getRecommendedActions(Context, Belief, Intention, Action),
    
    reply_json_dict(Action).
    

% Start the server
start_server(Port) :-
    http_server(http_dispatch, [port(Port)]).
    writeln('Server started on port 6060').
