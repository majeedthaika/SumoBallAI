define = {
    "swf": "sumoball.swf",
    "model": "model.h5",
    "dataset": "dataset.p",
    "action_space": 10,
    "action_names": ["pink_wins", "purple_wins", "load_screen", "blue_wins", "red_wins", 
                     "start_screen", "yellow_wins", "green_wins", "in_game_screen", "selection_screen"],
    "state_space": (84, 84, 1),
    "ingame_model": "",
    "ingame_action_space": 8,
    "ingame_action_names": ["UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT"]
}
