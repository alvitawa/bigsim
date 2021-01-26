# Simple pygame program
# Import and initialize the pygame library
from lib.config import CLUSTERING_METHOD
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
environ['SDL_AUDIODRIVER'] = 'dsp'

import pygame
from pygame_widgets import Button
from .sliders import LabeledSlider
import numpy as np
import time

# 53626fff
OCEAN_COLOR = (30, 32, 40) # (49, 36, 131) # (255, 255, 255) 
BOID_COLOR = (219, 126, 67) # (0, 0, 0)
PROGRESS_COLOR = (OCEAN_COLOR[0], OCEAN_COLOR[1], OCEAN_COLOR[2]+5)

draw_progress = True

N_COLORS = 100
COLORS = np.random.choice(range(256), size=3*N_COLORS).reshape(N_COLORS, 3)

# the changable parameters during the simulation
SLIDABLE_PARAMETERS = [
#   Name                    Max Value
    ("speed",               0.4),
    ("agility",             1),
    ("separation_weight",   200),
    ("separation_range",    3),
    ("cohesion_weight",     1),
    ("cohesion_range",      3),
    ("alignment_weight",    1),
    ("alignment_range",     3),
    ("obstacle_weight",     200),
    ("obstacle_range",      3),
    ("shark_weight",        200),
    ("shark_range",         3),
    ("shark_separation_range", 3),
    ("shark_separation_weight", 200),
    ("shark_speed",         0.4),
    ("shark_agility",       1),
]


def change_clustering():
    pass

# Set up pygame
def init(simulation, resolution=[1080, 720], enable_menu=True, enable_metrics=True, fps=20, sync=True):
    global _simulation
    _simulation = simulation

    global _resolution
    _resolution=resolution

    # Menu flag
    global _menu
    _menu = False

    global _stop
    _stop = False

    global _fps
    _fps = fps

    global _sync
    _sync = sync

    global _last_render
    _last_render = 0


    pygame.init()

    pygame.display.set_caption("Bad Boids 4 Life Simulator")

    global _screen
    _screen = pygame.display.set_mode(resolution)

    global _clustering_method
    _clustering_method = "LARS_CLUSTERING"

    button_data = [
#       Button rectangle                               function                     TEXT
        [[0, _screen.get_height()-60, 60, 60],          toggle_menu,                 "menu"],
        [[_screen.get_width()-130, _screen.get_height() - 60, 60, 60],   save,        "SAVE"],
        [[_screen.get_width()-60, _screen.get_height() - 60, 60, 60],    load,        "LOAD"],
        [[_screen.get_width()-200, _screen.get_height() - 60, 60, 60],    change_clustering,        _clustering_method]
    ]

    # buttons
    global _buttons
    _buttons = []
    for rect, func, text in button_data:
        button = Button(_screen, rect[0], rect[1], rect[2], rect[3],
                        text=text,
                        onClick=func)
        _buttons.append(button)

    # time
    global _clock
    _clock = pygame.time.Clock()

    global _menu_enabled
    _menu_enabled = enable_menu

    global _metrics_enabled
    _metrics_enabled = enable_metrics

    global _sliders
    _sliders = []

    if enable_menu:
        for n, (par, max_value) in enumerate(SLIDABLE_PARAMETERS):
            slider = LabeledSlider(
                _screen, 10, resolution[1] - 60 - n * 40, par, initial=_simulation.pars[par], min=0, max=max_value
            )
            _sliders.append(slider)

    return _screen, _clock

def tick():
    global _simulation, _screen, _clock, _stop, _last_render

    if not _sync:
        now = time.time()
        if _last_render + 1/_fps > now:
            return
        _last_render = now

    _stop = check_input()

    clear_screen(_screen)
    
    # draw population
    debug_draw(_screen, _simulation.pars.max_steps)
    draw_population(_screen)

    # draw UI
    if _menu_enabled:
        draw_sliders()
        draw_buttons()

    if _metrics_enabled:
        # draw FPS counter
        draw_number(_screen, int(_clock.get_fps()) if _sync else round(_fps, 2), (0,0), np.abs(np.array(OCEAN_COLOR)-255))

        # draw Population counter
        draw_number(_screen, _simulation.population.shape[0], (0.9*_resolution[0], 0), np.abs(np.array(OCEAN_COLOR)-200))

    update_screen()
    
    if _sync:
        _clock.tick(_fps)

    return _stop

def quit():
    pygame.quit()

def save():
    _simulation.save_pars()

def load():
    global _simulation
    pars = _simulation.load_pars()
    for (par, _), slider in zip(SLIDABLE_PARAMETERS, _sliders):
        slider.set_value(pars[par])

def check_input():
    global _simulation

    events = pygame.event.get()

    if _menu_enabled:
        # Button interaction
        for button in _buttons:
            button.listen(events)

        # Update _sliders
        for (par, _), slider in zip(SLIDABLE_PARAMETERS, _sliders):
            slider.update(events)
            _simulation.pars[par] = slider.get_value()

    # Keyboard presses
    for event in events:
        # Exit nicely
        if event.type == pygame.QUIT:
            return True

        # example
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pass

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3: # right click = select fish
                pos = np.array(pygame.mouse.get_pos())
                scaled = pos / np.array(pygame.display.get_window_size()) * _simulation.pars.shape

                fish_rel = _simulation.population[:, 0, :] - scaled
                distances = np.sqrt(np.power(fish_rel, 2).sum(axis=-1))

                mindin = np.min(distances)

                if mindin < 0.5:
                    _simulation.selected_index = np.argmin(distances)
                else:
                    _simulation.selected_index = None

            if event.button == 2: # middle click place obstacle
                pos = np.array(pygame.mouse.get_pos())
                scaled = pos / np.array(pygame.display.get_window_size()) * _simulation.pars.shape

                new_shape = np.array(_simulation.obstacles.shape)
                new_shape[0] += 1
                _simulation.obstacles = np.append(_simulation.obstacles, scaled).reshape(new_shape)

    return False


def clear_screen(_screen):
    # Fill ocean background
    _screen.fill(OCEAN_COLOR)

def draw_sliders():
    if _menu:
        for slider in _sliders:
            slider.draw()

# _buttons
def toggle_menu():
    global _menu
    _menu = not _menu

def draw_buttons():
    global _buttons
    for button in _buttons:
        # Always draw menu button
        if button.string == "menu":
            button.draw()
        # Draw other buttons with menu
        elif (_menu):
            button.draw()
            
def draw_number(_screen, number, location, color):
    '''Displays a fps number on the _screen'''
    font = pygame.font.SysFont('arial', 50)
    text = font.render(str(number), True, color)
    _screen.blit(text, location)
    # pygame.display.update()

def debug_draw(_screen, max_steps=1000000):
    global _simulation

    window = pygame.display.get_window_size()

    # draw progress
    prog = _simulation.stats.iterations / _simulation.pars.max_steps
    pygame.draw.rect(_screen, PROGRESS_COLOR, (0, 0, window[0], prog * window[1]))

    if _simulation.selected_index == None:
        return
    selected_fish = _simulation.population[_simulation.selected_index]
    scaling = np.array(window) / _simulation.pars.shape

    location = selected_fish[0] * scaling

    WALL_COLOR = (0, 0, 255)
    OBSTACLE_COLOR = (255, 0, 0)
    SEPERATION_COLOR = (255, 0, 255)
    COHESION_COLOR = (0, 255, 0)
    ALIGNMENT_COLOR = (0, 128, 128)
    SHARK_COLOR = (200, 200, 200)

    # draw ranges
    pygame.draw.circle(_screen, WALL_COLOR, tuple(location), int(_simulation.pars.wall_range * scaling[0]), width=1)
    pygame.draw.circle(_screen, OBSTACLE_COLOR, tuple(location), int(_simulation.pars.obstacle_range * scaling[0]), width=1)
    pygame.draw.circle(_screen, SHARK_COLOR, tuple(location), int(_simulation.pars.shark_range * scaling[0]), width=1)

    pygame.draw.circle(_screen, SEPERATION_COLOR, tuple(location), int(_simulation.pars.separation_range * scaling[0]), width=1)
    pygame.draw.circle(_screen, COHESION_COLOR, tuple(location), int(_simulation.pars.cohesion_range * scaling[0]), width=1)
    pygame.draw.circle(_screen, ALIGNMENT_COLOR, tuple(location), int(_simulation.pars.alignment_range * scaling[0]), width=1)



    # get vectors
    assigned_box = selected_fish[0] // _simulation.grid_size
    grid_coordinates = _simulation.population[:, 0, :] // _simulation.grid_size
    outer_idx = (np.sum(np.abs(grid_coordinates - assigned_box), axis=1) <= _simulation.box_sight_radius)
    vectors = _simulation.fish_move_vectors(np.array([selected_fish]), _simulation.population[outer_idx], _simulation.obstacles, _simulation.sharks, _simulation.pars)

    total_norm = np.sum(np.linalg.norm(np.array(vectors)))

    HUNDRED_PERCENT_SIZE = 1
    vectors /= total_norm * HUNDRED_PERCENT_SIZE

    pygame.draw.line(_screen, WALL_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[4]) * scaling))
    pygame.draw.line(_screen, OBSTACLE_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[3]) * scaling))
    pygame.draw.line(_screen, SHARK_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[5]) * scaling))
    
    pygame.draw.line(_screen, SEPERATION_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[1]) * scaling))
    pygame.draw.line(_screen, COHESION_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[0]) * scaling))
    pygame.draw.line(_screen, ALIGNMENT_COLOR, tuple(selected_fish[0] * scaling), tuple((selected_fish[0] + vectors[2]) * scaling))



om_de_zoveel = 100
om_de_zoveel2 = 2
draw_count = 120

colors = None
def draw_population(_screen):
    global _simulation, colors, draw_count, om_de_zoveel

    scaling = np.array(pygame.display.get_window_size()) / _simulation.pars.shape

    # Coloring with GMM

    for boid, label in zip(_simulation.population, _simulation.labels):

        if boid[1][1] >= 0:
            rotation = np.arccos(boid[1][0])
        else:
            rotation = -np.arccos(boid[1][0])

        color = COLORS[label % N_COLORS]
        draw_fish(_screen, boid[0] * scaling, rotation, color, 7, 4)

    for obstacle in _simulation.obstacles:
        location = tuple((obstacle[0] * scaling))

        pygame.draw.circle(_screen, (245, 40, 40), location, 6)

    for i, shark in enumerate(_simulation.sharks):
        if shark[1][1] >= 0:
            rotation = np.arccos(shark[1][0])
        else:
            rotation = -np.arccos(shark[1][0])

        if _simulation.shark_state[i] > 0:
            draw_shark(_screen, shark[0] * scaling, rotation, (189, 20, 10), 40, 40, eatin = 'mouth_wide_open') 
        elif _simulation.shark_state[i] < 0:
            draw_shark(_screen, shark[0] * scaling, rotation, (192,192,192), 40, 40, eatin = 'mouth_closed')
        else:
            draw_shark(_screen, shark[0] * scaling, rotation, (192,192,192), 40, 40, eatin = 'mouth_open')
            
            
    return True
    


def update_screen():
    # Update _screen
    pygame.display.flip()



def draw_fish(surface, position, rotation, color=BOID_COLOR, length=30, width=15):
    # the fish array
    head_up_down = np.array(
        [[0.5 * length, 0],
        [0.25 * length, 0.5 * width],
        [-0.25 * length, 0.1 * width],
        [-0.5 * length, 0.5 * width],
        [-0.5 * length, -0.5 * width],
        [-0.25 * length, -0.1 * width],
        [0.25 * length, -0.5 * width]
        ]
    )

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    rotated = R.dot(head_up_down.T).T

    rotated += position

    positions = [(int(np.round(a)), int(np.round(b))) for a, b in rotated]

    pygame.draw.polygon(surface, color, positions, width=0)

def draw_shark(surface, position, rotation, color, length, width, eatin = 'not_eatin'):
    if eatin == 'mouth_open':
        # shark array normal
        head_up_down = np.array(
            [[1.2 * length - length / 1.2, -0.1 * width],
            [0.7 * length - length / 1.2, 0 * width],
            [1.2 * length - length / 1.2, 0.1 * width],
            [0.6 * length - length / 1.2, 0.35 * width],
            [0.45 * length - length / 1.2, 0.6 * width],
            [0.4 * length - length / 1.2, 0.35 * width],
            [-0.25 * length - length / 1.2, 0.1 * width],
            [-0.4 * length - length / 1.2, 0.5 * width],
            [-0.5 * length - length / 1.2, -0.5 * width],
            [-0.25 * length - length / 1.2, -0.1 * width],
            [0.5 * length - length / 1.2, -0.2 * width]
            ]
        )
    elif eatin == 'mouth_wide_open':
        # shark when eating
        head_up_down = np.array(
            [[1.2 * length - length / 1.2, -0.3 * width],
            [0.7 * length - length / 1.2, 0 * width],
            [1.2 * length - length / 1.2, 0.3 * width],
            [0.6 * length - length / 1.2, 0.35 * width],
            [0.45 * length - length / 1.2, 0.6 * width],
            [0.4 * length - length / 1.2, 0.35 * width],
            [-0.25 * length - length / 1.2, 0.1 * width],
            [-0.4 * length - length / 1.2, 0.5 * width],
            [-0.5 * length - length / 1.2, -0.5 * width],
            [-0.25 * length - length / 1.2, -0.1 * width],
            [0.5 * length - length / 1.2, -0.2 * width]
            ]
        )
    else:
        head_up_down = np.array(
            # shark when just ate
            [
            [.5 * length   - length / 1.2, -0.2 * width],
            [1.2 * length     - length / 1.2, 0. * width],
            [0.5 * length   - length / 1.2, 0.35 * width],
            [0.375 * length - length / 1.2, 0.6 * width],
            [0.33 * length  - length / 1.2, 0.35 * width],
            [-0.21 * length - length / 1.2, 0.1 * width],
            [-0.33 * length - length / 1.2, 0.5 * width],
            [-0.42 * length - length / 1.2, -0.5 * width],
            [-0.21 * length - length / 1.2, -0.1 * width],
            [0.42 * length  - length / 1.2, -0.2 * width]
            ]
        )


    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    rotated = R.dot(head_up_down.T).T

    rotated += position

    positions = [(int(np.round(a)), int(np.round(b))) for a, b in rotated]

    pygame.draw.polygon(surface, color, positions, width=0)


# Done! Time to quit.
pygame.quit()
