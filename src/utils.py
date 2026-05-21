import os
import pickle
from shutil import copy
import subprocess
import sys
import time

from pygame import rect
import numpy as np
from gym.envs.registration import register
from sklearn.neighbors import NearestNeighbors

# import matplotlib.pyplot as plt
import signal
from functools import wraps


"""
Game utilities
"""

BG_NAME = "floor"
AVATAR_NAME = "avatar"
BG_COLOR = "LIGHTGRAY"
COLORS = [
    "BLUE",
    "GRAY",
    "WHITE",
    "BROWN",
    "ORANGE",
    "YELLOW",
    "PINK",
    "GOLD",
    "LIGHTORANGE",
    "LIGHTBLUE",
    "LIGHTGREEN",
    "DARKBLUE",
    "PURPLE",
]
# Canonical palette matches tsividis_vgdl/vgdl/colors.py and
# heroku_vgdl/static/ontology/constants.js -- the RGB values the fMRI
# participants actually saw on screen. Colors not present in tsividis
# (DARKGREEN, LIGHTPURPLE, LIGHTPINK) retain their legacy values.
COLOR_DICT = {
    "DARKBLUE": (1, 87, 155),
    "LIGHTRED": (255, 82, 82),
    "PURPLE": (92, 107, 192),
    "RED": (211, 47, 47),
    "PINK": (255, 138, 128),
    "GOLD": (255, 196, 0),
    "LIGHTORANGE": (255, 112, 67),
    "WHITE": (250, 250, 250),
    "DARKGRAY": (68, 90, 100),
    "GREEN": (129, 199, 132),
    "BLACK": (55, 71, 79),
    "BROWN": (109, 76, 65),
    "LIGHTGRAY": (207, 216, 220),
    "LIGHTGREEN": (185, 246, 202),
    "YELLOW": (255, 245, 157),
    "GRAY": (69, 90, 100),
    "LIGHTBLUE": (144, 202, 249),
    "DARKGREEN": (0, 200, 0),
    "ORANGE": (230, 81, 0),
    "BLUE": (25, 118, 210),
    "LIGHTPURPLE": (200, 150, 220),
    "LIGHTPINK": (255, 230, 230),
}

interactions_that_prevent_others = ["stepBack", "stepBackIfHasLess", "bounceForward"]

dir_dict = dict(
    UP=np.array([0, -1]),
    DOWN=np.array([0, 1]),
    LEFT=np.array([-1, 0]),
    RIGHT=np.array([1, 0]),
)
inv_dir_dict = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}


# Whether the game advances between key presses (step_by_step=False) or not
def is_step_by_step(game_id):
    if game_id in ["preconditions", "pushBoulders", "relational", "watergame"]:
        return True
    elif game_id in [
        "avoidGeorge",
        "beesAndBirds",
        "plaqueAttack",
        "portals",
        "boulderDash",
        "aliens",
        "missile_command",
        "closing_gates",
        "antagonist",
        "jaws",
        "test",
    ]:
        return False
    else:
        raise NotImplementedError


# Register gym enviroment
def register_game(game, generated=False, level=0, fast=False):
    repo_path = get_repo_path()
    games_path = repo_path + "/games/"
    if generated:
        games_path += "generated/"
    name = f"{game}-v0"
    game_path = games_path + f"{game}_v0" + "/"
    register(
        id=name,
        entry_point="src.vgdl.interfaces.gym:VGDLEnv",
        kwargs={
            "game_file": os.path.join(game_path, game + ".txt"),
            "level_file": os.path.join(game_path, game + f"_lvl{level}.txt"),
            "obs_type": "objects",
            "block_size": 1 if fast else 50,
        },
    )


# Reads VGDL text file into a string
def load_vgdl(vgdl_path):
    with open(vgdl_path, "r") as f:
        vgdl = f.read()
    return vgdl


"""
System utilities
"""


def get_repo_path(one_above=False):
    if one_above:
        path = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/"
    else:
        path = "/".join(os.path.abspath(__file__).split("/")[:-2]) + "/"
    return path


def find_experiment_path(repo_path, game_id, exp_name="", trial_id=0, overwrite=False):
    inference_data_path = repo_path + f"/{exp_name}/{game_id}/"
    exp_path = inference_data_path + str(trial_id) + "/"
    if not overwrite:
        if not is_run_locally():
            assert not os.path.exists(exp_path)
        else:
            while os.path.exists(exp_path):
                trial_id += 100
                exp_path = inference_data_path + str(trial_id) + "/"
    os.makedirs(exp_path, exist_ok=True)
    return exp_path, trial_id


def get_machine_id():
    with open(os.environ["HOME"] + "/machine_id", "r") as f:
        machine_id = f.read()[:-1]
    return machine_id


def is_running_in_terminal():
    # Method 1: Check if it's interactive
    return hasattr(sys, "ps1")  # True for interactive/REPL


def is_run_locally():
    if "PYCHARM_HOSTED" in os.environ:
        return True
    elif "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/" in sys.argv[0]:
        return True
    else:
        return False


def prepare_run_continuation(exp_path, params):
    # detect current gen
    files = os.listdir(params["exp_path"] + "theories_and_descriptions/")
    start_gen = 0
    for f in files:
        if "description_gen" in f:
            gen = int(f.split("description_gen_")[1].split(".txt")[0])
            if gen >= start_gen and f"theory_gen_{gen}.txt" in files:
                start_gen = gen + 1

    # now clean the rest
    files = os.listdir(exp_path + "dumps/")
    for f in files:
        if f == "interaction_data":
            continue
        gen = int(f.split("generation_")[1].split("_life")[0])
        if gen >= start_gen:
            os.remove(exp_path + "dumps/" + f)
    files = os.listdir(exp_path + "dumps/interaction_data/")
    for f in files:
        gen = int(f.split("generation_")[1].split("_life")[0])
        if gen >= start_gen:
            os.remove(exp_path + "dumps/interaction_data/" + f)

    # rename output file
    counts = len([f for f in os.listdir(exp_path) if "output" in f])
    copy(exp_path + "output.txt", exp_path + f"output_{counts}.txt")
    if start_gen > 0:
        with open(
            params["exp_path"]
            + f"theories_and_descriptions/description_gen_{start_gen - 1}.txt",
            "r",
        ) as f:
            description = f.read()
    else:
        description = None

    return start_gen, description


def mpi_fork(n, extra_mpi_args=[], oversubscribe=False):
    """
    Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        # "-bind-to core" is crucial for good performance
        if oversubscribe:
            args = (
                ["mpirun", "--oversubscribe", "-n", str(n)]
                + extra_mpi_args
                + [sys.executable]
            )
        else:
            args = ["mpirun", "-np", str(n)] + extra_mpi_args + [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI

    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()

    sys.excepthook = new_hook


def pickle_save(data, path):
    if os.path.exists(path):
        copy(path, path + ".copy")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    if os.path.exists(path + ".copy"):
        os.remove(path + ".copy")


"""
Main
"""


class Logger(object):
    """ """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Flush the write buffer

    def flush(self):
        for f in self.files:
            f.flush()


class TimeDict:
    """ """

    def __init__(self, rank=0):
        self.rank = rank  # only track time in rank 0
        self.times = dict()
        self.times_init = dict()
        self.ticked = set()
        self.start = dict()
        self.timestep = 0

    def tic(self, key):
        if self.rank == 0:
            if key not in self.start.keys():
                self.start[key] = self.timestep
            if key not in self.times.keys():
                self.times[key] = [0]
            assert key not in self.ticked, f"ticked twice key {key}"
            self.times_init[key] = time.time()
            self.ticked.add(key)

    def toc(self, key):
        if self.rank == 0:
            assert key in self.times_init.keys()
            assert key in self.ticked, f"key {key} was not ticked"
            self.times[key][-1] += time.time() - self.times_init[key]
            self.ticked.remove(key)

    def step(self):
        if self.rank == 0:
            for key in self.times.keys():
                self.times[key].append(0)
            self.timestep += 1

    def __str__(self):
        s = "TimeTracker:\n"
        for key, times in self.times.items():
            s += f"  > {key}: {times[-1]:.3f}s\n"
        return s

    def plot(self, data_path=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.rank == 0:
            fig = plt.figure(figsize=(25, 15))
            for i, (key, times) in enumerate(self.times.items()):
                start = self.start[key]
                steps = np.arange(start, start + len(times))
                if i < 10:
                    linestyle = "-"
                elif i < 20:
                    linestyle = "--"
                else:
                    linestyle = "-."
                plt.plot(steps, times, label=key, linestyle=linestyle, linewidth=2)
            plt.legend()
            if data_path is None:
                plt.show()
            else:
                plt.savefig(data_path + "/time_tracker.png")
            plt.close(fig=fig)


"""
Data structures
"""


def array2str(array):
    return [f"{value:.3f}" for value in array]


class hashdict(dict):
    def __hash__(self):
        keys = [k[0] if isinstance(k, tuple) else k for k in self.keys()]
        sorted_keys = np.sort(keys)
        sorted_values = [hash(self.get(k)) for k in sorted_keys]
        return hash(tuple(zip(sorted_keys, sorted_values)))

    def __eq__(self, other):
        return hash(self) == hash(other)


def is_subset_of(dict1, dict2):
    # return True if dict1 is a subset of dict2
    return all(dict2.get(key, None) == val for key, val in dict1.items())


# data handling
def get_chunk(lst, size, i):
    n = len(lst)
    chunk_size = n // size
    remainder = n % size

    if i < remainder:
        start = i * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = remainder * (chunk_size + 1) + (i - remainder) * chunk_size
        end = start + chunk_size

    return lst[start:end]


def custom_round(number, decimal=3):
    bigger = int(number * 10**decimal)
    if str(bigger)[-1] != "0":
        lower = bigger
        higher = bigger
        while True:
            lower = lower - 1
            higher = higher + 1
            if str(lower)[-1] == "0":
                return lower / (10**decimal)
            elif str(higher)[-1] == "0":
                return higher / (10**decimal)
    else:
        return bigger / (10**decimal)


"""
Datastore
"""


class CustomCollisions:
    """ """

    def __init__(self, positions, ids):
        self.ids = ids
        self.idx2ids = dict(zip(range(len(ids)), ids))
        self.res = 1000
        self.positions = np.array(positions)
        # radius = [0, 0.99, 1.99, 2.99]
        # self.rects = dict()
        # for rad in radius:
        #     if rad == 0:
        #         size = 1e-3
        #     else:
        #         size = rad
        #     size *= self.res
        #     self.rects[rad] = [pygame.Rect(pos[0], pos[1], size, size) for pos in self.positions * self.res]
        # collisions = []
        # for i in range(len(rects)):
        #     colliding_indices = rects[i].collidelistall(rects[i + 1:])
        #     collisions.extend([(i, i + 1 + j) for j in colliding_indices])
        # self.collisions[rad] = dict(zip(self.ids, [[] for _ in range(len(self.ids))]))
        # for (i, j) in collisions:
        #     self.collisions[rad][self.idx2ids[i]].append(self.idx2ids[j])
        #     self.collisions[rad][self.idx2ids[j]].append(self.idx2ids[i])

    def find_collisions(self, pos, remove_obj_ids=None, radius=2):

        dist = np.abs(self.positions - pos)
        eps = 1e-4
        if radius == 0:
            # obj have same position
            formula = dist.sum(axis=1) < 1e-3
        elif radius == 1:
            # obj collide
            formula = np.logical_and(dist[:, 0] < 1 - eps, dist[:, 1] < 1 - eps)
        elif radius == 2:
            # object collide on 1 dimension and are less than one move to collide on the other
            intersect_on_dim_0 = np.logical_and(
                dist[:, 0] < 1 - eps, dist[:, 1] < 2 - eps
            )
            intersect_on_dim_1 = np.logical_and(
                dist[:, 1] < 1 - eps, dist[:, 0] < 2 - eps
            )
            formula = np.logical_or(intersect_on_dim_0, intersect_on_dim_1)
        elif radius == 3:
            # object can collide if both move
            dim_0_double = np.logical_and(dist[:, 1] < 1 - eps, dist[:, 0] < 3 - eps)
            dim_1_double = np.logical_and(dist[:, 0] < 1 - eps, dist[:, 1] < 3 - eps)
            dim_0_dim_1 = np.logical_and(dist[:, 1] < 2 - eps, dist[:, 0] < 2 - eps)
            formula = np.logical_or(dim_0_double, dim_1_double, dim_0_dim_1)
        else:
            raise NotImplementedError
        colliding_indices = list(np.argwhere(formula).flatten())
        colliding_ids = [self.idx2ids[i] for i in colliding_indices]
        if remove_obj_ids in colliding_ids:
            colliding_ids.remove(remove_obj_ids)

        # if radius == 0:
        #     size = 1e-3
        # else:
        #     size = radius
        # size *= self.res
        # rect = pygame.Rect(pos[0] * self.res, pos[1] * self.res, size, size)
        # colliding_indices = rect.collidelistall(self.rects[radius])
        # colliding_ids = [self.idx2ids[i] for i in colliding_indices]
        # if remove_obj_ids in colliding_ids:
        #     colliding_ids.remove(remove_obj_ids)
        return colliding_ids


def convert_traj_into_object_mvt(
    states, block_size, objects=None, nn=None, steps_processed=0, several=False
):
    # if prev_states is provided, use this as previous obs instead of the previous element in states
    # is_simulation = prev_states is not None
    if not several:
        states = [[o] for o in states]
    n_ids = len(states[0])
    if objects is None:
        assert nn is None
        objects = [dict() for _ in range(n_ids)]
        nearest_neighbors = [[] for _ in range(n_ids)]
    else:
        assert nn is not None
        objects = [objects]
        nearest_neighbors = [nn]
        assert n_ids == 1
    for id in range(n_ids):
        for i_step, step in enumerate(states):
            if i_step < steps_processed:
                continue
            obj_tracked_at_this_step = set()
            positions = []
            obj_ids = []
            for i_col, column in enumerate(step[id]):
                for i_line, cell in enumerate(column):
                    for element in cell:
                        if element is not None:
                            positions.append(element["pos"])
                            obj_ids.append(element["obj_id"])
                            if (
                                element["obj_id"] not in objects[id].keys()
                            ):  # this is a new object
                                objects[id][element["obj_id"]] = dict(
                                    name=element["name"],
                                    obj_id=element["obj_id"],
                                    prev_pos=[None] * (i_step + 1),
                                    pos=[None] * i_step + [element["pos"]],
                                    mov=[None] * (i_step + 1),
                                    resources=[None] * i_step + [element["resources"]],
                                    resources_max=[element["resources_max"]] * i_step
                                    + [element["resources_max"]],
                                    rect=[None] * i_step
                                    + [
                                        rect.Rect(
                                            element["pos"][0] * block_size,
                                            element["pos"][1] * block_size,
                                            block_size,
                                            block_size,
                                        )
                                    ],
                                )
                            else:
                                new_pos = element["pos"]
                                old_pos = objects[id][element["obj_id"]]["pos"][-1]
                                if old_pos is None:
                                    mvt = None
                                else:
                                    mvt = (
                                        custom_round(new_pos[0] - old_pos[0]),
                                        custom_round(new_pos[1] - old_pos[1]),
                                    )
                                objects[id][element["obj_id"]]["mov"].append(mvt)
                                objects[id][element["obj_id"]]["pos"].append(new_pos)
                                objects[id][element["obj_id"]]["resources"].append(
                                    element["resources"]
                                )
                                objects[id][element["obj_id"]]["resources_max"].append(
                                    element["resources_max"]
                                )
                                objects[id][element["obj_id"]]["rect"].append(
                                    rect.Rect(
                                        new_pos[0] * block_size,
                                        new_pos[1] * block_size,
                                        block_size,
                                        block_size,
                                    )
                                )
                                objects[id][element["obj_id"]]["prev_pos"].append(
                                    old_pos
                                )
                            if element["obj_id"] not in obj_tracked_at_this_step:
                                obj_tracked_at_this_step.add(element["obj_id"])
                            else:
                                assert False
                        else:
                            assert False

            nearest_neighbors[id].append(CustomCollisions(positions, obj_ids))
            # track objects that died
            for obj_id in objects[id].keys():
                if obj_id not in obj_tracked_at_this_step:
                    old_pos = objects[id][obj_id]["pos"][-1]
                    objects[id][obj_id]["prev_pos"].append(old_pos)
                    objects[id][obj_id]["pos"].append(None)
                    objects[id][obj_id]["rect"].append(None)
                    objects[id][obj_id]["resources"].append(None)
                    objects[id][obj_id]["resources_max"].append(None)
                    objects[id][obj_id]["mov"].append(None)
    if not several:
        return objects[0], nearest_neighbors[0]
    else:
        return objects, nearest_neighbors


#
def convert_traj_into_events(
    states,
    prev_states,
    wins,
    loses,
    rewards,
    events=None,
    steps_processed=0,
    several=False,
    verbose=False,
):

    map_of_prev_state_elements = []
    for i_step, step in enumerate(prev_states):
        map_of_prev_state_elements.append(dict())
        if i_step >= steps_processed and step is not None:
            for i_col, column in enumerate(step):
                for i_line, cell in enumerate(column):
                    for element in cell:
                        map_of_prev_state_elements[-1][element["obj_id"]] = element

    if not several:
        states = [[o] for o in states]
    n_ids = len(states[0])
    all_events = []
    for id in range(n_ids):
        obj_names = dict()
        if events is None:
            all_events.append([])  # list of events that occurred
            all_objects_encountered = set()
        else:
            all_events.append(events)
            assert n_ids == 1
            all_objects_encountered = set()
            for step_event in events:
                for ev in step_event:
                    all_objects_encountered.add(ev[2])
                    obj_name, obj_id = ev[1:3]
                    if obj_id not in obj_names.keys():
                        obj_names[obj_id] = obj_name

        for i_step, step in enumerate(states):
            if i_step < steps_processed:
                continue
            if verbose:
                print(f"Events for transition {i_step}")
            new_obj_positions = dict()
            all_events[id].append([])  # list events at this timestep
            all_objects_encountered_this_step = set()
            if i_step > 0:
                if wins[i_step - 1]:
                    assert not loses[i_step - 1]
                    all_events[id][i_step].append(("win", None, None, None))
                elif loses[i_step - 1]:
                    all_events[id][i_step].append(("lose", None, None, None))
                all_events[id][i_step].append(("rew", rewards[i_step - 1], None, None))

            # go through the map
            for i_col, column in enumerate(step[id]):
                for i_line, cell in enumerate(column):
                    for element in cell:
                        if element["name"] == "wall":
                            continue
                        # for current element and current step
                        pos = element["pos"]
                        resources = element["resources"]
                        new_obj_positions[element["obj_id"]] = pos
                        all_objects_encountered.add(
                            element["obj_id"]
                        )  # track all objects ever encountered
                        all_objects_encountered_this_step.add(element["obj_id"])
                        if element["obj_id"] not in obj_names.keys():
                            obj_names[element["obj_id"]] = element["name"]
                        if i_step > 0:
                            prev_element = map_of_prev_state_elements[i_step].get(
                                element["obj_id"], None
                            )
                            if (
                                prev_element is None
                            ):  # it wasn't here, the obj was just born
                                prev_pos = None
                                prev_resources = None
                            else:  # we found it
                                prev_pos = prev_element["pos"]
                                prev_resources = prev_element["resources"]

                            if prev_pos is None:  # the object was just born
                                if verbose:
                                    print(f"  {element['name']} appeared in {pos}")
                                all_events[id][i_step].append(
                                    ("birth", element["name"], None, pos)
                                )
                            else:  # the objects moved or didn't move
                                new_pos = np.array(pos)
                                old_pos = np.array(prev_pos)
                                mvt = (
                                    custom_round(new_pos[0] - old_pos[0]),
                                    custom_round(new_pos[1] - old_pos[1]),
                                )
                                if mvt != (0, 0):
                                    if verbose:
                                        print(
                                            f"  {element['name']} #{element['obj_id']} moved from {prev_pos} to {pos}"
                                        )
                                    all_events[id][i_step].append(
                                        (
                                            "mvt",
                                            element["name"],
                                            element["obj_id"],
                                            prev_pos,
                                            pos,
                                            mvt,
                                        )
                                    )
                                else:
                                    all_events[id][i_step].append(
                                        (
                                            "no_mvt",
                                            element["name"],
                                            element["obj_id"],
                                            prev_pos,
                                            pos,
                                            mvt,
                                        )
                                    )
                                # was there a change in resources?
                                resources_names = list(
                                    set(
                                        list(prev_resources.keys())
                                        + list(resources.keys())
                                    )
                                )
                                for r_name in resources_names:
                                    prev_val = prev_resources.get(r_name, 0)
                                    new_val = resources.get(r_name, 0)
                                    if prev_val != new_val:
                                        all_events[id][i_step].append(
                                            (
                                                "resource_change",
                                                element["name"],
                                                element["obj_id"],
                                                r_name,
                                                prev_val,
                                                new_val,
                                                new_val - prev_val,
                                            )
                                        )

            # track object deaths
            for obj_id in all_objects_encountered - all_objects_encountered_this_step:
                if i_step > 0:
                    prev_element = map_of_prev_state_elements[i_step].get(obj_id, None)
                    # prev_element = find_in_map(prev_states[i_step], obj_id)  # find that object in the previous map
                    if prev_element is None:  # it was already dead
                        pass
                    else:
                        assert prev_element["pos"] is not None
                        prev_pos = prev_element["pos"]
                        if verbose:
                            print(
                                f"  {obj_names[obj_id]} #{obj_id} just died (prev_pos={prev_pos})"
                            )
                        all_events[id][i_step].append(
                            ("death", obj_names[obj_id], obj_id, prev_pos)
                        )
    if not several:
        return all_events[0]
    else:
        return all_events


# Count the number of objects ?
def get_count_objects_per_step(traj_episode, names):

    count_objs = []
    state_episode = traj_episode["state"]
    for obs in state_episode:
        count_objs.append(dict())
        for col in obs:
            for cell in col:
                for obj in cell:
                    if obj["name"] not in count_objs[-1].keys():
                        count_objs[-1][obj["name"]] = 0
                    count_objs[-1][obj["name"]] += 1
                    names.add(obj["name"])
    for obj in names:
        for c in count_objs:
            if obj not in c.keys():
                c[obj] = 0

    return count_objs, names


# Not used
def convert_simulated_traj_into_events(
    states, prev_states, true_steps, several=False, verbose=False
):
    map_of_prev_state_elements = []
    for i_step, step in enumerate(prev_states):
        map_of_prev_state_elements.append(dict())
        # if i_step > 0:
        for i_col, column in enumerate(step):
            for i_line, cell in enumerate(column):
                for element in cell:
                    map_of_prev_state_elements[-1][element["obj_id"]] = element

    if not several:
        states = [[o] for o in states]
    n_ids = len(states[0])
    all_events = []
    for id in range(n_ids):
        all_events.append([])  # list of events that occurred
        obj_names = dict()
        for i_step, step in enumerate(states):
            if verbose:
                print(f"Events for transition {i_step}")
            new_obj_positions = dict()
            all_events[id].append([])  # list events at this timestep
            all_objects_encountered_this_step = set()
            all_objects_previous_step = set()
            if step[id] is None:
                stop = 1
            if step[id] is not None:
                # go through the previous map
                for i_col, column in enumerate(prev_states[i_step]):
                    for i_line, cell in enumerate(column):
                        for element in cell:
                            # for current element and current step
                            if element["obj_id"] not in obj_names.keys():
                                obj_names[element["obj_id"]] = element["name"]
                            all_objects_previous_step.add(
                                element["obj_id"]
                            )  # track all objects ever encountered
                # go through the map
                for i_col, column in enumerate(step[id]):
                    for i_line, cell in enumerate(column):
                        for element in cell:
                            # for current element and current step
                            pos = element["pos"]
                            resources = element["resources"]
                            new_obj_positions[element["obj_id"]] = pos
                            all_objects_encountered_this_step.add(element["obj_id"])
                            prev_element = map_of_prev_state_elements[i_step].get(
                                element["obj_id"], None
                            )
                            # prev_element = find_in_map(prev_states[i_step], element['obj_id'])  # find that object in the previous map
                            if (
                                prev_element is None
                            ):  # it wasn't here, the obj was just born
                                prev_pos = None
                                prev_resources = None
                            else:  # we found it
                                prev_pos = prev_element["pos"]
                                prev_resources = prev_element["resources"]

                            if prev_pos is None:  # the object was just born
                                if verbose:
                                    print(f"  {element['name']} appeared in {pos}")
                                all_events[id][i_step].append(
                                    ("birth", element["name"], None, pos)
                                )
                            else:  # the objects moved or didn't move
                                new_pos = np.array(pos)
                                old_pos = np.array(prev_pos)
                                mvt = (
                                    custom_round(new_pos[0] - old_pos[0]),
                                    custom_round(new_pos[1] - old_pos[1]),
                                )
                                if mvt != (0, 0):
                                    if verbose:
                                        print(
                                            f"  {element['name']} #{element['obj_id']} moved from {prev_pos} to {pos}"
                                        )
                                    all_events[id][i_step].append(
                                        (
                                            "mvt",
                                            element["name"],
                                            element["obj_id"],
                                            prev_pos,
                                            pos,
                                            mvt,
                                        )
                                    )
                                else:
                                    all_events[id][i_step].append(
                                        (
                                            "no_mvt",
                                            element["name"],
                                            element["obj_id"],
                                            prev_pos,
                                            pos,
                                            mvt,
                                        )
                                    )
                                resources_names = list(
                                    set(
                                        list(prev_resources.keys())
                                        + list(resources.keys())
                                    )
                                )
                                for r_name in resources_names:
                                    prev_val = prev_resources.get(r_name, 0)
                                    new_val = resources.get(r_name, 0)
                                    if prev_val != new_val:
                                        all_events[id][i_step].append(
                                            (
                                                "resource_change",
                                                element["name"],
                                                element["obj_id"],
                                                r_name,
                                                prev_val,
                                                new_val,
                                                new_val - prev_val,
                                            )
                                        )
                # track object deaths
                for obj_id in (
                    all_objects_previous_step - all_objects_encountered_this_step
                ):
                    if true_steps[i_step] > 0:
                        prev_element = map_of_prev_state_elements[i_step].get(
                            obj_id, None
                        )
                        # prev_element = find_in_map(prev_states[i_step], obj_id)  # find that object in the previous map
                        if prev_element is None:  # it was already dead
                            pass
                        else:
                            assert prev_element["pos"] is not None
                            prev_pos = prev_element["pos"]
                            if verbose:
                                print(
                                    f"  {obj_names[obj_id]} #{obj_id} just died (prev_pos={prev_pos})"
                                )
                            all_events[id][i_step].append(
                                ("death", obj_names[obj_id], obj_id, prev_pos)
                            )
    if not several:
        return all_events[0]
    else:
        return all_events


"""
State inference
"""

neighbor_memory = dict()
obj_rect_mem = dict()


def detect_collisions_in_map(
    obj_pos,
    state,
    i_obj_step,
    format,
    obj_episode,
    obj_id=None,
    external_collisions=True,
):
    if obj_pos in obj_rect_mem.keys():
        obj_rect = obj_rect_mem[obj_pos]
    else:
        obj_rect = get_rect(obj_pos, format)
        obj_rect_mem[obj_pos] = obj_rect
    # search neighborhood
    neighbor_key = (
        (int(obj_pos[0]), int(obj_pos[1])),
        2,
        True,
        (format["w"], format["h"]),
    )
    # caching this
    if neighbor_key not in neighbor_memory.keys():
        neighbors = get_neighbor_pos(
            pos=(int(obj_pos[0]), int(obj_pos[1])),
            dist=2,
            no_wrap=True,
            size=(format["w"], format["h"]),
        )
        neighbor_memory[neighbor_key] = neighbors
    neighbors = neighbor_memory[neighbor_key]
    obj_ids = []
    # search possible collisions in the -2/2 neighborhood
    for neigh in neighbors:
        obj_ids += [
            o["obj_id"] for o in state[neigh[0]][neigh[1]] if o["obj_id"] != obj_id
        ]
    # now search for collisions
    other_rects = [obj_episode[o_id]["rect"][i_obj_step] for o_id in obj_ids]
    # collision_obj_ids = list(np.array(obj_ids)[obj_rect.collidelistall(other_rects)])
    collision_obj_ids = [obj_ids[i] for i in obj_rect.collidelistall(other_rects)]
    if external_collisions:
        if (
            obj_pos[0] < 0
            or obj_pos[0] > format["w"] - 1
            or obj_pos[1] < 0
            or obj_pos[1] > format["h"] - 1
        ):
            collision_obj_ids.append("EOS")
    return collision_obj_ids


def find_id_closest_in_map(map, obj_name, pos):
    obj_poss = []
    obj_ids = []
    for i_col, col in enumerate(map):
        for i_row, cell in enumerate(col):
            for o in cell:
                if o["name"] == obj_name:
                    obj_poss.append(o["pos"])
                    obj_ids.append(o["obj_id"])
    if len(obj_poss) == 0:
        return None
    obj_poss = np.array(obj_poss)
    distances = np.linalg.norm(np.array(pos).reshape(1, -1) - obj_poss, ord=1, axis=1)
    min_dist = np.min(distances)
    return obj_ids[np.argwhere(distances == min_dist).flatten()[0]]


"""
Agent, Planning
"""


def find_in_map(state, obj_id):
    for i_col, column in enumerate(state):
        for i_line, cell in enumerate(column):
            for element in cell:
                if element["obj_id"] == obj_id:
                    return element
    return None


def find_all_names_in_map(state, obj_name):
    out = []
    for i_col, column in enumerate(state):
        for i_line, cell in enumerate(column):
            for element in cell:
                if element["name"] == obj_name:
                    out.append(element)
    return out


def get_obj_ids_from_map(state, obj_name, return_pos=False):
    obj_ids = []
    obj_poss = []
    for i_col, column in enumerate(state):
        for i_line, cell in enumerate(column):
            for element in cell:
                if element["name"] == obj_name:
                    obj_ids.append(element["obj_id"])
                    obj_poss.append((i_col, i_line))
    if return_pos:
        return obj_ids, obj_poss
    else:
        return obj_ids


"""
Object and interaction types
"""


# Not used
def find_closest_in_map(map, obj_name, pos, ord=1):
    obj_poss = []
    for i_col, col in enumerate(map):
        for i_row, cell in enumerate(col):
            for o in cell:
                if o["name"] == obj_name:
                    obj_poss.append(o["pos"])
    if len(obj_poss) == 0:
        return None, np.nan
    obj_poss = np.array(obj_poss)
    distances = np.linalg.norm(np.array(pos).reshape(1, -1) - obj_poss, ord=ord, axis=1)
    min_dist = np.min(distances)
    return obj_poss[np.argwhere(distances == min_dist).flatten()], min_dist


def find_closest_target_in_map(prev_map, current_map, obj_name, pos, ord=2):
    prev_obj_poss = []
    prev_obj_ids = []
    for i_col, col in enumerate(prev_map):
        for i_row, cell in enumerate(col):
            for o in cell:
                if o["name"] == obj_name:
                    prev_obj_poss.append(o["pos"])
                    prev_obj_ids.append(o["obj_id"])
    obj_poss = []
    obj_ids = []
    for i_col, col in enumerate(current_map):
        for i_row, cell in enumerate(col):
            for o in cell:
                if o["name"] == obj_name:
                    obj_poss.append(o["pos"])
                    obj_ids.append(o["obj_id"])
    if len(obj_poss) + len(prev_obj_poss) == 0:
        return None
    all_obj_poss = []
    for obj_id in set(prev_obj_ids + obj_ids):
        if obj_id in obj_ids:  # object still alive
            all_obj_poss.append(obj_poss[obj_ids.index(obj_id)])
        else:
            all_obj_poss.append(prev_obj_poss[prev_obj_ids.index(obj_id)])
    all_obj_poss = np.array(all_obj_poss)
    distances = np.linalg.norm(
        np.array(pos).reshape(1, -1) - all_obj_poss, ord=ord, axis=1
    )
    min_dist = np.min(distances)
    return all_obj_poss[np.argwhere(distances == min_dist).flatten()]


neighbor_cache = dict()


def get_neighbors(
    nn_episode, i_ep, i_abs_step, i_step, prev_or_current, obj_id, pos, radius
):
    neighbor_key = (i_ep, i_abs_step, prev_or_current, obj_id, tuple(pos), radius)
    if neighbor_key not in neighbor_cache.keys():
        step = i_step if prev_or_current == "current" else i_step - 1
        nns = nn_episode[step].find_collisions(
            pos, remove_obj_ids=obj_id, radius=radius
        )
        # nns = nn_episode[step].find_neighbors(pos, remove_obj_ids=obj_id, radius=radius)
        neighbor_cache[neighbor_key] = nns
    else:
        nns = neighbor_cache[neighbor_key]
    return nns


def clear_neighbor_cache():
    neighbor_cache.clear()


"""
Likelihood
"""


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def normalize_log_joints(log_joints):
    max_log_joint = np.max(log_joints)
    log_joints_shifted = log_joints - max_log_joint
    sum_exp_log_joints = np.sum(np.exp(log_joints_shifted))
    log_normalizer = max_log_joint + np.log(sum_exp_log_joints)
    log_posteriors = log_joints - log_normalizer
    return log_posteriors


def normalize_log_probs(log_probs):
    max_log_prob = np.max(log_probs)
    shifted_log_probs = log_probs - max_log_prob
    exp_log_probs = np.exp(shifted_log_probs)
    sum_exp_log_probs = np.sum(np.exp(shifted_log_probs))
    probs = exp_log_probs / sum_exp_log_probs
    return probs


def compute_probability_and_confidence(loglikes, idk_loglike):
    # return loglikes, idk_loglike
    # the judgement of the llm might be seen confident if the loglikelihood of the "I don't know" answer is low compared to the highest likelihood
    # AND if the highest likelihood is relatively high compared to others.

    shift, temp = 1, 1
    shift2, temp2 = 1, 1
    beta_softmax = 0.5

    if len(loglikes) == 1:
        confidence = 1
    else:
        sorted_loglikes = np.flip(np.sort(loglikes))
        best_loglike, second_best_loglike = sorted_loglikes[:2]
        assert second_best_loglike != -np.inf

        diff_loglike = best_loglike - idk_loglike
        confidence = 1 / (1 + np.exp(-(diff_loglike - shift) / temp))
        diff_loglike = best_loglike - second_best_loglike
        confidence2 = 1 / (1 + np.exp(-(diff_loglike - shift2) / temp2))
        confidence = min(confidence, confidence2)

    # normalize answers
    loglikes = np.array(loglikes)
    tmp_loglikes = beta_softmax * loglikes
    sampling_probabilities = np.exp(tmp_loglikes) / np.nansum(np.exp(tmp_loglikes))
    sampling_probabilities /= sum(sampling_probabilities)

    # pickle_path = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Research/Scratch/infer-vgdl/data/inference_data/llm_prior_test_70b_loglike/plaqueAttack/0/prior_data.pkl"
    # with open(pickle_path, 'rb') as f:
    #     data = pickle.load(f)
    #
    # tests = dict()
    #
    # for key, key_data in data.items():
    #     if key in ['win', 'lose']:
    #         for name, value in key_data.items():
    #             key_to_store = key + '_' + name
    #             tests[key_to_store] = (value[1], value[-1], ['yes', 'no'])
    #     elif isinstance(key, str) or isinstance(key, tuple):
    #         likes = []
    #         options = []
    #         confidence = None
    #         for name, value in key_data.items():
    #             confidence = value[-1]
    #             likes.append(value[1])
    #             options.append(name)
    #         tests[key] = (likes, confidence, options)
    #     else: raise NotImplementedError
    #
    # shift, temp = 1, 1
    # shift2, temp2 = 1, 1
    # beta_softmax = 0.5
    #
    # for key, val in tests.items():
    #     loglikes, idk_loglike, options = val
    #     if len(loglikes) == 1 or idk_loglike == 1:
    #         continue
    #     sorted_loglikes = np.flip(np.sort(loglikes))
    #     best_loglike, second_best_loglike = sorted_loglikes[:2]
    #     assert second_best_loglike != - np.inf
    #
    #
    #
    #     diff_loglike = best_loglike - idk_loglike
    #     confidence = 1 / (1 + np.exp(- (diff_loglike - shift) / temp))
    #     diff_loglike = best_loglike - second_best_loglike
    #     confidence2 = 1 / (1 + np.exp(- (diff_loglike - shift2) / temp2))
    #
    #     # normalize answers
    #     loglikes = np.array(loglikes)
    #     tmp_loglikes = beta_softmax * loglikes
    #     sampling_probabilities = np.exp(tmp_loglikes) / np.nansum(np.exp(tmp_loglikes))
    #     sampling_probabilities /= sum(sampling_probabilities)
    #     # if min(confidence, confidence2) > 0.5:
    #     print('\n')
    #     print(key)
    #     print('loglikes:', ', '.join([f"{name}: {loglike:.2f}" for name, loglike in zip(options, loglikes)]))
    #     print('idk loglike', idk_loglike)
    #     print('confidences', confidence, confidence2, min(confidence, confidence2))
    #     print('sampling probas', ', '.join([f"{name}: {prob:.2f}" for name, prob in zip(options, sampling_probabilities)]))
    #         pass
    #     else:
    #         print('\n')
    #         print(key)
    #         print('confidences', confidence, confidence2, min(confidence, confidence2))

    return sampling_probabilities, confidence


def format_prompt(obj_names, type, params, colors, vgdl_lines=None):
    if type == "win":
        if obj_names is None:
            prompt = "you win if you survive long enough"
        else:
            names_str = ", ".join([colors[name] for name in obj_names])
            prompt = f"you win when you kill or reach all {names_str} objects"
    elif type == "lose":
        if obj_names is None:
            prompt = "you lose if you do not win before the timeout"
        else:
            names_str = " or ".join([colors[name] + " objects" for name in obj_names])
            prompt = f"you lose if all {names_str} die/disappear"
    elif type == "Immovable":
        prompt = f"{colors[obj_names[0]]} objects cannot move"
    elif type == "Portal":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            teleported = None
            for key, vgdl_line in vgdl_lines.dict.items():
                if (
                    vgdl_line.type == "teleportToExit"
                    and vgdl_line.name[1] == obj_names[0]
                ):
                    teleported = vgdl_line.name[0]
            exits = vgdl_lines.dict[obj_names[0]].params["stype"]
            if teleported is None:
                params = None
            else:
                params = [teleported, exits]
        if params is None:
            prompt = f"{colors[obj_names[0]]} objects are portals"
        else:
            prompt = f"{colors[obj_names[0]]} objects are portals teleporting {colors[params[0]]} objects to a random {colors[params[1]]} object"
    elif type == "Flicker":
        prompt = f"{colors[obj_names[0]]} objects cannot move and disappear after a certain time"
    elif type == "ResourcePack":
        prompt = f"{colors[obj_names[0]]} objects are resources than can be collected"
    elif type == "SpawnPoint":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            spawned = vgdl_lines.dict[obj_names[0]].params["stype"]
            params = spawned
        if params is None:
            prompt = (
                f"{colors[obj_names[0]]} objects regularly spawn/generate other objects"
            )
        else:
            prompt = f"{colors[obj_names[0]]} objects regularly spawn/generate {colors[params]} objects"
    elif type == "Passive":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            pusher = None
            for key, vgdl_line in vgdl_lines.dict.items():
                if (
                    vgdl_line.type == "bounceForward"
                    and vgdl_line.name[0] == obj_names[0]
                ):
                    pusher = vgdl_line.name[1]
            params = pusher
        if params is None:
            prompt = f"{colors[obj_names[0]]} objects can be pushed"
        else:
            prompt = f"{colors[obj_names[0]]} objects can be pushed by {colors[params]} objects"
    elif type == "Missile":
        prompt = f"{colors[obj_names[0]]} objects move along one axis"
    elif type == "Bomber":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            spawned = vgdl_lines.dict[obj_names[0]].params["stype"]
            params = spawned
        if params is None:
            prompt = f"{colors[obj_names[0]]} objects move along one axis and regularly spawn/generate objects"
        else:
            prompt = f"{colors[obj_names[0]]} objects move along one axis and regularly spawn/generate {colors[params]} objects"
    elif type == "RandomNPC":
        prompt = f"{colors[obj_names[0]]} objects move randomly"
    elif type == "Chaser":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            chased = vgdl_lines.dict[obj_names[0]].params["stype"]
            params = chased
        if params is None:
            prompt = f"{colors[obj_names[0]]} objects chase or flee another object"
        else:
            prompt = f"{colors[obj_names[0]]} objects chase or flee the nearest {colors[params]} object"
    elif type == "MovingAvatar":
        prompt = f"the {colors[obj_names[0]]} avatar can move around but does not shoot"
    elif type == "FlakAvatar":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            spawned = vgdl_lines.dict[obj_names[0]].params["stype"]
            params = spawned
        if params is None:
            prompt = f"the {colors[obj_names[0]]} avatar can move horizontally and shoot (press space bar)"
        else:
            prompt = f"the {colors[obj_names[0]]} avatar can move horizontally and shoot {colors[params]} objects (press space bar)"
    elif type == "ShootAvatar":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            spawned = vgdl_lines.dict[obj_names[0]].params["stype"]
            params = spawned
        if params is None:
            prompt = f"the {colors[obj_names[0]]} avatar can move around and shoot (press space bar)"
        else:
            prompt = f"the {colors[obj_names[0]]} avatar can move around and shoot {colors[params]} objects (press space bar)"
    elif type == "noInteraction":
        prompt = f"nothing happens to {colors[obj_names[0]]} objects when they collide with {colors[obj_names[1]]} objects"
    elif type == "killSprite":
        prompt = f"{colors[obj_names[0]]} objects die when they collide with {colors[obj_names[1]]} objects"
    elif type == "transformTo":
        if isinstance(params, dict):
            assert vgdl_lines is not None
            spawned = None
            for key, vgdl_line in vgdl_lines.dict.items():
                if vgdl_line.type == "transformTo" and vgdl_line.name == obj_names:
                    spawned = vgdl_line.params["stype"]
            params = spawned
        if params is None:
            prompt = f"{colors[obj_names[0]]} objects get transformed when they collide with {colors[obj_names[1]]} objects"
        else:
            prompt = f"{colors[obj_names[0]]} objects get transformed into {colors[params]} objects when they collide with {colors[obj_names[1]]} objects"
    elif type == "removeResource":
        if vgdl_lines:
            resource = vgdl_lines.dict.get(obj_names).params["resource"]
            prompt = f"{colors[obj_names[0]]} objects die when they collide with {colors[obj_names[1]]} objects but also take a {colors[resource]} resource from them"
            # prompt = f"{colors[obj_names[0]]} objects kill {colors[obj_names[1]]} but lose a {colors[resource]} resource in the process, when they both collide"
        else:
            prompt = f"{colors[obj_names[0]]} objects die when they collide with {colors[obj_names[1]]} objects but also take a resource from them"
            # prompt = f"{colors[obj_names[0]]} objects kill {colors[obj_names[1]]} but lose a resource in the process, when they both collide"
    elif type == "killIfHasLess":
        prompt = f"{colors[obj_names[0]]} objects die if they don't have enough resources when they collide with {colors[obj_names[1]]} objects"
    elif type == "stepBack":
        prompt = f"{colors[obj_names[0]]} objects are blocked by {colors[obj_names[1]]} objects (and vice versa)"
    elif type == "bounceForward":
        prompt = (
            f"{colors[obj_names[1]]} objects can push {colors[obj_names[0]]} objects"
        )
    elif type == "teleportToExit":
        portal = obj_names[1]
        exit_name = vgdl_lines.dict[portal].params["stype"]
        prompt = f"{colors[obj_names[0]]} objects can teleport from {colors[obj_names[1]]} portal objects to {colors[exit_name]} objects"
    elif type == "reverseDirection":
        prompt = f"{colors[obj_names[0]]} objects reverse direction when they collide with {colors[obj_names[1]]} objects"
    elif type == "addResource":
        prompt = f"{colors[obj_names[1]]} objects can collect {colors[obj_names[0]]} resources"
    else:
        assert False, f"{type} is unknown"
        prompt = None
    return prompt


def just_changed_direction_left_right(obj_episode, obj_id, i_step):
    last_mov, last_i_mov, future_mov, future_i_move = None, None, None, None
    for i_mov, mov in enumerate(reversed(obj_episode[obj_id]["mov"][:i_step])):
        if mov is None:
            break
        if np.abs(mov).sum() > 0:
            last_mov = mov
            last_i_mov = i_mov
            if last_mov[1] != 0:
                return False
            break
    if last_mov is None:
        return False

    if len(obj_episode[obj_id]["mov"]) > i_step:
        for i_mov, mov in enumerate(obj_episode[obj_id]["mov"][i_step + 1 :]):
            if mov is None:
                break
            if np.abs(mov).sum() > 0:
                future_mov = mov
                future_i_move = i_mov
                if future_mov[1] != 0:
                    return 0
                break
    if future_mov is None:
        return False
    if (
        future_i_move != last_i_mov
    ):  # make sure it was just before and just after with some cooldown
        return False
    return np.sign(future_mov[0]) + np.sign(last_mov[0]) == 0


"""
Not used?
"""


class CustomNN:
    """
    Nearest neighbors
    """

    def __init__(self, positions, ids):
        self.ids = np.array(ids)
        self.positions = np.array(positions)
        self.model = NearestNeighbors(radius=2, metric="manhattan")
        self.model.fit(self.positions)

    def find_neighbors(self, obj_pos, remove_obj_ids=None, radius=2):
        distances, indices = self.model.radius_neighbors([obj_pos], radius=radius)
        if remove_obj_ids is None:
            obj_ids = set()
        elif isinstance(remove_obj_ids, str):
            obj_ids = {remove_obj_ids}
        elif isinstance(remove_obj_ids, list):
            obj_ids = set(remove_obj_ids)
        else:
            raise NotImplementedError
        return sorted(set(self.ids[indices[0]]) - set(obj_ids))


def get_neighbor_pos(pos, dist, not_itself=False, no_wrap=False, size=None):
    pos_x, pos_y = np.meshgrid(
        np.arange(pos[0] - dist, pos[0] + dist + 1),
        np.arange(pos[1] - dist, pos[1] + dist + 1),
    )
    neigh_pos = np.array((pos_x, pos_y)).T.reshape(-1, 2)
    true_dist = np.linalg.norm(np.array(pos).reshape(1, 2) - neigh_pos, ord=1, axis=1)
    cond = true_dist <= dist
    if not_itself:
        cond = np.logical_and(cond, true_dist > 0)
    if no_wrap:
        cond = np.logical_and(cond, 0 <= neigh_pos[:, 0])
        cond = np.logical_and(cond, neigh_pos[:, 0] < size[0])
        cond = np.logical_and(cond, 0 <= neigh_pos[:, 1])
        cond = np.logical_and(cond, neigh_pos[:, 1] < size[1])
    return list(neigh_pos[np.where(cond)])


def get_neighbor_ids(obj_pos, state, obj_id, fmt):
    # search neighborhood
    neighbor_key = (
        (int(obj_pos[0]), int(obj_pos[1])),
        2,
        True,
        (fmt["w"], fmt["h"]),
    )
    # caching this
    if neighbor_key not in neighbor_memory.keys():
        neighbors = get_neighbor_pos(
            pos=(int(obj_pos[0]), int(obj_pos[1])),
            dist=2,
            no_wrap=True,
            size=(fmt["w"], fmt["h"]),
        )
        neighbor_memory[neighbor_key] = neighbors
    neighbors = neighbor_memory[neighbor_key]
    obj_ids = []
    # search possible collisions in the -2/2 neighborhood
    for neigh in neighbors:
        obj_ids += [
            o["obj_id"] for o in state[neigh[0]][neigh[1]] if o["obj_id"] != obj_id
        ]
    return obj_ids


def get_rect(pos, format):
    return rect.Rect(
        pos[0] * format["bs"], pos[1] * format["bs"], format["bs"], format["bs"]
    )


def detect_collisions(obj_poss, other_obj_pos, format):
    if isinstance(obj_poss, tuple):
        obj_poss = [obj_poss]
    collisions = []
    other_obj_rects = [get_rect(pos, format) for pos in other_obj_pos]
    for obj_pos in obj_poss:
        obj_rect = get_rect(obj_pos, format)
        collisions.append(obj_rect.collidelistall(other_obj_rects))

    return collisions


def mvt_align(mvt1, mvt2):
    assert np.sum(np.abs(mvt1) > 0) < 2
    assert np.sum(np.abs(mvt2) > 0) < 2
    if (
        (mvt1[0] > 0 and mvt2[0] > 0)
        or (mvt1[0] < 0 and mvt2[0] < 0)
        or (mvt1[1] > 0 and mvt2[1] > 0)
        or (mvt1[1] < 0 and mvt2[1] < 0)
    ):
        return True
    else:
        return False


def is_outside(pos, format):
    return (
        pos[0] < 0 or pos[0] > format["w"] - 1 or pos[1] < 0 or pos[1] > format["h"] - 1
    )


def wrap_pos(pos, format):
    new_pos = [None, None]
    if pos[0] % format["w"] > format["w"] - 1:
        new_pos[0] = 0
    else:
        new_pos[0] = pos[0] % format["w"]
    if pos[1] % format["h"] > format["h"] - 1:
        new_pos[1] = 0
    else:
        new_pos[1] = pos[1] % format["h"]
    return tuple(new_pos)


def c_are_compatible(c1, c2):
    # checks whether the new constraint c2 is compatible with c1
    # could be compatible because:
    # * c2 is included in c1
    # * c2 specifies new slots not found in c1
    # * c2 overrules parts of c1

    is_compatible = True
    specifies_new_slots = False  # whether the new causes specify at least one new slot
    overwrites_slots = False  # whether the new causes overwrites at least one slot
    at_least_one_exact_match = False
    constraints_to_add = dict()
    for k, v in c2.items():
        v_type, v_params = v
        if k in c1.keys():
            c_type, c_params = c1[k]
            v_type, v_params = v
            c_params = c_params.copy()
            v_params = v_params.copy()
            # sometimes we assume the default case if we don't see evidence for something else
            # eg we have not seen an avatar shoot, we assume it cannot
            # if we have not seen an object move, we assume it does not
            # here we need to be able to extend constraints in light of new evidence
            # so we assume that 'default' constraints are in fact compatible with new ones
            if v_type == c_type:
                at_least_one_exact_match = True
                if c_params == v_params:
                    pass
                else:
                    keys = [
                        "speed",
                        "cooldown",
                        "prob",
                        "singleton",
                        "total",
                        "scoreChange",
                        "orientation",
                    ]
                    if c_type == "Portal":
                        keys.append("stype")
                    for k1 in keys:
                        if k1 in c_params.keys():
                            del c_params[k1]
                        if k1 in v_params.keys():
                            del v_params[k1]
                    if v_params != c_params:
                        is_compatible = False
                        break
            elif c_type == "emptyInteraction" and v_type in [
                "killIfHasLess",
                "killIfOtherHasMore",
            ]:
                overwrites_slots = True
                constraints_to_add[k] = v
            elif c_type == "killSprite" and v_type in [
                "addResource",
                "removeResource",
                "killIfHasLess",
            ]:
                overwrites_slots = True
                constraints_to_add[k] = v
            elif c_type == "Immovable" and k != "wall":
                # if we assumed an object was immovable, but in fact it's movable, we should be ok with that
                # if we assumed it was immovable, but in fact it dies, or in fact it spawns, we should be ok with that as well
                overwrites_slots = True
                constraints_to_add[k] = v
            elif c_type == "Missile" and v_type == "Bomber":
                # if we assumed it moved but it fact it also bombs, we should be ok with that
                overwrites_slots = True
                constraints_to_add[k] = v
            elif (c_type == "MovingAvatar" and v_type == "ShootAvatar") or (
                c_type == "HorizontalAvatar" and v_type == "FlakAvatar"
            ):
                # if we assumed the avatar did not shoot, we should be ok with shooting
                overwrites_slots = True
                constraints_to_add[k] = v
            else:
                is_compatible = False
                break
        elif isinstance(k, tuple):
            symmetric_k = (k[1], k[0])
            if (
                symmetric_k in c1.keys()
            ):  # there is another interaction between these two objects
                symmetric_int = c1[symmetric_k]
                symmetric_int_type, symmetric_int_params = symmetric_int
                if (
                    v_type not in interactions_that_prevent_others
                    and symmetric_int_type not in interactions_that_prevent_others
                ):
                    # the interactions are not incompatible
                    specifies_new_slots = True
                    constraints_to_add[k] = v
                elif v_type == "stepBack" and symmetric_int_type == "stepBack":
                    pass
                else:
                    # if not, then there is a compatibility problem
                    is_compatible = False
                    break
            else:
                specifies_new_slots = True
                constraints_to_add[k] = v
        else:  # this constraint is compatible
            specifies_new_slots = True
            constraints_to_add[k] = v

    is_included = not overwrites_slots and not specifies_new_slots and is_compatible
    covers = is_compatible and set(list(c1.keys())).issubset(set(list(c2.keys())))
    compatibility_info = dict(
        is_included=is_included,
        overwrites_slots=overwrites_slots,
        specifies_new_slots=specifies_new_slots,
        covers=covers,
        at_least_one_exact_match=at_least_one_exact_match,
    )
    return is_compatible, compatibility_info, constraints_to_add


def copy_and_extend_c(original_c, c_extensions={}, hashable=False):
    new_c = original_c.copy()
    new_c.update(c_extensions)
    if not hashable:
        return new_c
    else:
        return hashdict(new_c)


def copy_and_extend_if_possible(original_c, c_extensions, slots_values, hashable=False):
    new_c = original_c.copy()
    for k, v in c_extensions.items():
        if k in slots_values.keys():
            values = list(slots_values[k])
            if (
                len(values) == 1 and values[0][0] == "emptyInteraction"
            ):  # this is not possible
                if v[0] == "killIfOtherHasMore":
                    pass
                else:
                    continue
        cause = {k: v}
        is_compatible, compatibility_info, constraints_to_add = c_are_compatible(
            new_c, cause
        )
        if is_compatible:
            new_c.update(constraints_to_add)

    if not hashable:
        return new_c
    else:
        return hashdict(new_c)


def find_a_birth(i_step, obj_episode, name):
    assert i_step > 0
    for obj in obj_episode.values():
        if obj["name"] == name:  # the transformed
            if obj["pos"][i_step - 1] is None and obj["pos"][i_step] is not None:
                return True
    return False


def get_birth_positions(i_step, obj_episode, name, return_id=False):
    assert i_step > 0
    birth_pos = []
    birth_id = []
    for obj in obj_episode.values():
        if obj["name"] == name:  # the transformed
            if obj["pos"][i_step - 1] is None and obj["pos"][i_step] is not None:
                birth_pos.append(obj["pos"][i_step])
                birth_id.append(obj["obj_id"])
    if return_id:
        return birth_pos, birth_id
    else:
        return birth_pos


def timeout(seconds=30):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler and alarm
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator
