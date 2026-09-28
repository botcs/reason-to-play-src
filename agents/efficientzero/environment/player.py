# import gym
# import gym_gvgai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import rl_models
import random
from collections import namedtuple, defaultdict
import numpy as np
import pdb
from scipy import misc
import imageio
import sys
from VGDLEnv import VGDLEnv
import csv
import cloudpickle
import wandb

import os
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE


# colors from VGDL?


class Player(object):
    def __init__(self, config):
        self.config = config

        # Set all random seeds BEFORE any initialization
        print('Setting random seed = {}'.format(self.config.random_seed))
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)

        self.Env = VGDLEnv(self.config.game_name, 'all_games')
        self.Env.set_level(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.game_size = np.shape(self.Env.render())
        self.base_channels = self.game_size[2]
        self.n_actions = len(self.Env.actions)

        # Frame stacking for proper state representation
        self.frame_stack_size = getattr(config, 'frame_stack_size', 4)
        self.frame_buffer = []

        # Calculate input channels based on representation type
        if getattr(config, 'use_grid_representation', True):
            # Initialize sprite mapping to get number of object types
            game = self.Env.current_env._game
            sprite_types = list(game.sprite_groups.keys())
            self._num_object_types = len(sprite_types)
            self.input_channels = self.frame_stack_size * self._num_object_types
            print(f"Using grid representation: {self._num_object_types} object types, "
                  f"{self.frame_stack_size} number of frames stacked => "
                  f"{self.input_channels} total input channels")
        else:
            # Use pixel representation
            self.input_channels = self.base_channels * self.frame_stack_size
            print(f"Using pixel representation: {self.input_channels} total input channels")

        self.policy_net = rl_models.rl_model(self)
        self.target_net = rl_models.rl_model(self)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = rl_models.ReplayMemory(self.config.max_mem)

        # Watch model for gradient and parameter tracking
        if not getattr(self.config, 'no_wandb', False):
            wandb.watch(self.policy_net, log='all', log_freq=100)

        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))

        self.steps_done = 0
        self.ended = 0

        self.episode_durations = []

        self.screen_history = []
        self.episode_action_counts = [0] * self.n_actions  # Track action distribution

    def save_screen(self):

        misc.imsave('original.png', self.Env.render())

        misc.imsave('altered.png', np.rollaxis(self.get_screen().cpu().numpy()[0], 0, 3))

    def get_screen(self):
        """Get pixel representation of current game state.
        Returns: Tensor of shape (1, C, H, W) with normalized pixel values [0, 1]
        """
        screen = self.Env.render().transpose((2, 0, 1))  # HWC -> CHW
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return screen.unsqueeze(0).to(self.device)  # Add batch dimension (BCHW)

    def get_one_hot_grid_state(self):
        """Get one-hot encoded grid state representing all objects"""
        game = self.Env.current_env._game

        # Get grid dimensions
        height, width = game.height, game.width

        # Get all sprite types (object types)
        if not hasattr(self, '_sprite_type_mapping'):
            self._sprite_type_mapping = {}
            sprite_types = list(game.sprite_groups.keys())
            for i, sprite_type in enumerate(sorted(sprite_types)):
                self._sprite_type_mapping[sprite_type] = i
            self._num_object_types = len(sprite_types)

        # Create one-hot grid directly as torch tensor: (num_object_types, height, width)
        grid = torch.zeros((self._num_object_types, height, width), dtype=torch.float32, device=self.device)

        # Fill grid with object positions
        for sprite_type, sprites in game.sprite_groups.items():
            if sprite_type in self._sprite_type_mapping:
                type_idx = self._sprite_type_mapping[sprite_type]
                for sprite in sprites:
                    if sprite not in game.kill_list:  # Only include living sprites
                        # Convert pixel position to grid coordinates
                        grid_x = int(sprite.rect.left / game.block_size)
                        grid_y = int(sprite.rect.top / game.block_size)
                        if 0 <= grid_x < width and 0 <= grid_y < height:
                            grid[type_idx, grid_y, grid_x] = 1.0

        # Add batch dimension and return
        return grid.unsqueeze(0)

    def get_stacked_state(self):
        """Get state with frame stacking for temporal information"""
        if getattr(self.config, 'use_grid_representation', True):
            return self.get_stacked_grid_state()
        else:
            return self.get_stacked_pixel_state()

    def get_stacked_grid_state(self):
        """Get stacked one-hot grid states for temporal information"""
        current_grid = self.get_one_hot_grid_state()
        self.frame_buffer.append(current_grid.squeeze(0))  # Remove batch dimension

        # Keep only the required number of frames
        if len(self.frame_buffer) > self.frame_stack_size:
            self.frame_buffer.pop(0)

        # If we don't have enough frames yet, repeat the current frame
        while len(self.frame_buffer) < self.frame_stack_size:
            self.frame_buffer.append(current_grid.squeeze(0))

        # Stack frames: (frame_stack_size, num_object_types, height, width)
        # Then flatten first two dims: (frame_stack_size * num_object_types, height, width)
        stacked_grids = torch.stack(self.frame_buffer, dim=0)
        stacked_state = stacked_grids.view(-1, stacked_grids.shape[-2], stacked_grids.shape[-1])
        return stacked_state.unsqueeze(0)  # Add batch dimension back

    def get_stacked_pixel_state(self):
        """Get stacked pixel frames for temporal information (fallback)"""
        current_frame = self.get_screen()
        self.frame_buffer.append(current_frame.squeeze(0))  # Remove batch dimension

        # Keep only the required number of frames
        if len(self.frame_buffer) > self.frame_stack_size:
            self.frame_buffer.pop(0)

        # If we don't have enough frames yet, repeat the current frame
        while len(self.frame_buffer) < self.frame_stack_size:
            self.frame_buffer.append(current_frame.squeeze(0))

        # Stack frames along channel dimension
        stacked_state = torch.cat(self.frame_buffer, dim=0).unsqueeze(0)  # Add batch dimension back
        return stacked_state

    def save_video(self):
        if not self.screen_history:
            return

        # Convert frames to uint8 format for video creation
        frames_uint8 = []
        for frame in self.screen_history:
            if frame.dtype != np.uint8:
                # Normalize to 0-255 range and convert to uint8
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max > frame_min:
                    frame_normalized = ((frame - frame_min) * 255.0 / (frame_max - frame_min)).astype(np.uint8)
                else:
                    frame_normalized = (frame * 255.0).astype(np.uint8)
            else:
                frame_normalized = frame
            frames_uint8.append(frame_normalized)

        video_filename = 'ddqn_outputs/screens/{}_episode{}.mp4'.format(self.config.game_name, self.episode)
        imageio.mimsave(video_filename, frames_uint8, fps=10, macro_block_size=1)
        print("Video saved: {} ({} frames, 10 fps)".format(video_filename, len(frames_uint8)))

    def append_frame(self):
        # Use full resolution render instead of resized version for better GIF quality
        frame = self.Env.render()  # Get full resolution frame
        self.screen_history.append(frame)

    def save_model(self, is_best=False):

        model_path = 'model_weights/{}_trial{}_{}.pt'.format(self.config.game_name, self.config.trial_num,
                                                              self.config.level_switch)
        torch.save(self.target_net.state_dict(), model_path)

        # Upload best checkpoint to wandb as artifact
        if is_best and not getattr(self.config, 'no_wandb', False):
            artifact = wandb.Artifact(
                name=f'{self.config.game_name}_best_model',
                type='model',
                description=f'Best model for {self.config.game_name} (reward: {self.best_reward:.2f})',
                metadata={
                    'reward': self.best_reward,
                    'episode': self.episode,
                    'step': self.steps,
                    'level': self.Env.lvl
                }
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact, aliases=['latest', 'best'])

    def load_model(self):

        state_dict = torch.load('model_weights/{}'.format(self.config.model_weight_path), map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def level_step(self):
        """Advance to next level after completing steps_per_level training steps.

        Returns:
            1 if all levels completed (training finished)
            0 if advanced to next level
        """
        if self.config.level_switch == 'sequential':
            # Check if we've completed the step budget for this level
            if self.level_steps >= self.config.steps_per_level:

                if self.Env.lvl == len(self.Env.env_list) - 1:  # if this is the last training level
                    print("Learning Finished - All levels completed")
                    return 1

                else:  # if this isn't the last level
                    # Save model at end of current level before advancing
                    print(f"Completed Level {self.Env.lvl} after {self.level_steps} steps - saving model")
                    self.save_model(is_best=False)

                    # Log level completion to wandb
                    if not getattr(self.config, 'no_wandb', False):
                        wandb.log({
                            'level_completed': self.Env.lvl,
                            'level_completion_steps': self.level_steps,
                            'total_steps': self.steps
                        }, step=self.steps)

                    self.Env.lvl += 1
                    self.Env.set_level(self.Env.lvl)
                    print(f"Advancing to Level {self.Env.lvl}")

                    # Reset level counters
                    self.level_steps = 0
                    self.level_episodes = 0
                    self.level_wins = 0
                    self.level_max_reward = float('-inf')
                    self.level_total_reward = 0.0

                    return 0

            # Not yet time to advance
            return 0

        elif self.config.level_switch == 'random':
            self.Env.lvl = np.random.choice(range(len(self.Env.env_list) - 1))
            self.Env.set_level(self.Env.lvl)
            return 0

        else:
            raise Exception('level switch not specified.')

    def model_update(self):

        if self.steps > 1000 and not self.steps % self.config.target_update:

            self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save model when achieving new best reward
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                print("New Best Reward: {:.2f}".format(self.best_reward))
                self.save_model(is_best=True)
                if not getattr(self.config, 'no_wandb', False):
                    wandb.log({
                        'best_reward': self.best_reward,
                        'model_saved': 1
                    }, step=self.steps)
            # Also save periodically for checkpointing
            elif self.steps % 50000 == 0:
                print("Periodic model save at step {}".format(self.steps))
                self.save_model(is_best=False)
                if not getattr(self.config, 'no_wandb', False):
                    wandb.log({'checkpoint_saved': 1}, step=self.steps)

    def select_action(self):

        sample = np.random.uniform()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
                        np.exp(-1. * self.steps_done / self.config.eps_decay)
        self.eps_threshold = eps_threshold  # Store for logging
        self.steps_done += 1.
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(self.state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.choice(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < self.config.batch_size:
            return
        transitions = self.memory.sample(self.config.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        try:
            reward_batch = torch.cat([r.float() for r in batch.reward])
        except:
            pdb.set_trace()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.config.batch_size, device=self.device)

        if self.config.doubleq:
            _, next_state_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)
            # ()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,
                                                                                              next_state_actions).squeeze(
                1)
            next_state_values = next_state_values.data
            # print("Double Q")
        else:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # print("Single Q")
        # Compute the expected Q values

        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch.float()
        # ()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # ()
        self.loss_history = loss
        # print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Apply gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Log training metrics
        if not getattr(self.config, 'no_wandb', False):
            wandb.log({
                'loss': loss.item(),
                'mean_q_value': state_action_values.mean().item(),
                'max_q_value': state_action_values.max().item(),
                'gradient_norm': total_norm,
                'step': self.steps
            }, step=self.steps)

    def train_model(self):

        print("Training Starting")
        print("-" * 25)

        if self.config.pretrain:
            print("Loading Model")

            self.load_model()

        self.steps = 0
        self.level_steps = 0  # Steps taken on current level
        self.level_episodes = 0  # Episodes completed on current level
        self.level_wins = 0  # Wins on current level
        self.level_max_reward = float('-inf')  # Max reward on current level
        self.level_total_reward = 0.0  # Sum of rewards on current level
        self.episode_steps = 0
        self.episode = 0
        self.best_reward = 0
        self.episode_reward = 0
        self.eps_threshold = self.config.eps_start
        # self.reward_history = []

        # Create required directories
        os.makedirs('ddqn_outputs/reward_histories', exist_ok=True)
        os.makedirs('ddqn_outputs/object_interaction_histories', exist_ok=True)
        os.makedirs('ddqn_outputs/pickleFiles', exist_ok=True)
        os.makedirs('ddqn_outputs/screens', exist_ok=True)
        os.makedirs('model_weights', exist_ok=True)

        with open('ddqn_outputs/reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                             self.config.level_switch,
                                                                             self.config.trial_num), "w") as file:
            writer = csv.writer(file)
            writer.writerow(["level", "steps", "ep_reward", "win", "game_name"])

        with open('ddqn_outputs/object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                self.config.game_name, self.config.level_switch, self.config.trial_num), "w") as file:
            interactionfilewriter = csv.writer(file)
            interactionfilewriter.writerow(
                ['agent_type', 'subject_ID', 'modelrun_ID', 'game_name', 'game_level', 'episode_number', 'event_name',
                 'count'])

        ## PEDRO: Rename as needed
        picklefilepath = 'ddqn_outputs/pickleFiles/{}.csv'.format(self.config.game_name)

        self.Env.reset()
        ## store game info once
        # pdb.set_trace()
        ## each episode gets a list of tuples, where each tuple has (avatar.x,, avatar.y, game.time, level_number)
        avatar_position_data = {'game_info': (self.Env.current_env._game.width, self.Env.current_env._game.height),
                                'episodes': [[(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                               self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                               self.Env.current_env._game.time,
                                               self.Env.lvl)]]}

        event_dict = defaultdict(lambda: 0)
        self.state = self.get_stacked_state()

        training_complete = False
        while not training_complete:

            self.steps += 1
            self.level_steps += 1
            self.episode_steps += 1

            # if not self.steps%100:
            # print(self.steps)
            # print(self.episode_reward)

            self.append_frame()

            # Select and perform an action
            self.action = self.select_action()
            self.episode_action_counts[self.action.item()] += 1

            self.reward, self.ended, self.win = self.Env.step(self.action.item())

            avatar_position_data['episodes'][-1].append((self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                         self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                         self.Env.current_env._game.time,
                                                         self.Env.lvl))

            ## PEDRO: 2. Store events that occur at each timestep
            timestep_events = set()
            for e in self.Env.current_env._game.effectListByClass:
                ## because event handling is so weird in Frogs, we need to filter out these events.
                ## Avatar-water and avatar-log collisions will still be reported from the (killSprite avatar water) interaction and (pullWithIt avatar log) interaction
                ## which is what a player perceives when they play
                if e in [('changeResource', 'avatar', 'water'), ('changeResource', 'avatar', 'log')]:
                    pass
                else:
                    timestep_events.add(tuple(sorted((e[1], e[2]))))

            for e in timestep_events:
                event_dict[e] += 1
            # if self.episode_reward < 0: pdb.set_trace()

            self.episode_reward += self.reward

            # Optional reward clipping - configurable to preserve reward structure
            if getattr(self.config, 'clip_rewards', False):
                self.reward = max(-1.0, min(self.reward, 1.0))

            self.reward = torch.tensor([self.reward], device=self.device)

            # print(self.reward)

            # Observe new state
            if not self.ended:
                self.next_state = self.get_stacked_state()
            else:
                self.next_state = None

            # Store the transition in memory
            self.memory.push(self)

            # Move to the next state
            self.state = self.next_state

            # Perform one step of the optimization (on the target network)
            self.optimize_model()

            if self.ended or self.episode_steps > self.config.timeout:

                if self.episode_steps > self.config.timeout: print("Game Timed Out")

                ## PEDRO: 3. At the end of each episode, write events to csv
                with open('ddqn_outputs/object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                        self.config.game_name, self.config.level_switch, self.config.trial_num), "a") as file:
                    interactionfilewriter = csv.writer(file)
                    for event_name, count in event_dict.items():
                        row = ('DDQN', 'NA', 'NA', self.config.game_name, self.Env.lvl, self.episode, event_name, count)
                        interactionfilewriter.writerow(row)

                self.episode += 1
                self.level_episodes += 1
                if self.win:
                    self.level_wins += 1

                # Update level max reward and total reward
                if self.episode_reward > self.level_max_reward:
                    self.level_max_reward = self.episode_reward
                self.level_total_reward += self.episode_reward

                # pdb.set_trace()
                win_status = "WIN" if self.win else "LOSE"
                level_win_rate = self.level_wins / self.level_episodes if self.level_episodes > 0 else 0
                level_avg_reward = self.level_total_reward / self.level_episodes if self.level_episodes > 0 else 0
                print("Episode {}: Level {}, Steps {} (Level: {}), Reward {:.2f}, Rollout Length {}, Status: {}, Level Win Rate: {:.2%}".format(
                    self.episode, self.Env.lvl, self.steps, self.level_steps, self.episode_reward, self.episode_steps, win_status, level_win_rate))
                sys.stdout.flush()

                # Log episode metrics to wandb
                if not getattr(self.config, 'no_wandb', False):
                    # Create action distribution histogram
                    action_indices = []
                    for action_idx, count in enumerate(self.episode_action_counts):
                        action_indices.extend([action_idx] * count)

                    wandb.log({
                        'episode_reward': self.episode_reward,
                        'max_reward': self.best_reward,
                        'level_max_reward': self.level_max_reward,
                        'level_avg_reward': level_avg_reward,
                        'episode_length': self.episode_steps,
                        'episode_win': int(self.win),
                        'win_rate': level_win_rate,
                        'level_wins': self.level_wins,
                        'level_episodes': self.level_episodes,
                        'current_level': self.Env.lvl,
                        'level_steps': self.level_steps,
                        'epsilon': self.eps_threshold,
                        'replay_buffer_size': len(self.memory),
                        'episode': self.episode,
                        'action_distribution': wandb.Histogram(action_indices)
                    }, step=self.steps)

                # Update the target network
                self.model_update()

                # self.reward_history.append([self.Env.lvl, self.steps, self.episode_reward, self.win])
                episde_results = [self.Env.lvl, self.steps, self.episode_reward, self.win, self.config.game_name]

                # Check if we should advance to next level
                if self.level_step():
                    # All levels completed - write final results and exit
                    with open('ddqn_outputs/reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                                         self.config.level_switch,
                                                                                         self.config.trial_num),
                              "a") as file:
                        writer = csv.writer(file)
                        writer.writerow(episde_results)
                    training_complete = True
                    break

                self.episode_reward = 0
                # print(self.recent_history)
                # print("Print Current Level: {}".format(self.Env.lvl))

                self.Env.reset()

                ## PEDRO: Write pickle to file every 100 episodes
                if self.episode % 2 == 0:
                    with open(picklefilepath, 'wb') as f:
                        cloudpickle.dump(avatar_position_data, f)

                ## add a new list for the new episode; populate new list with tuple of first state
                avatar_position_data['episodes'].append([(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                          self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                          self.Env.current_env._game.time, self.Env.lvl)])

                event_dict = defaultdict(lambda: 0)
                self.episode_steps = 0
                # Reset frame buffer for new episode
                self.frame_buffer = []
                self.episode_action_counts = [0] * self.n_actions  # Reset action tracking
                self.state = self.get_stacked_state()

                # if not self.episode % 10:
                # np.save("reward_histories/{}_reward_history_{}_trial{}.npy".format(self.config.game_name, self.config.level_switch, self.config.trial_num), self.reward_history)
                # np.savetxt('ddqn_outputs/reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name, self.config.level_switch, self.config.trial_num), a, fmt='%.2f', delimiter=',', header=" level,  steps,  ep_reward,  win")
                with open('ddqn_outputs/reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config.game_name,
                                                                                     self.config.level_switch,
                                                                                     self.config.trial_num),
                          "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(episde_results)

                if self.config.save_video:
                    self.save_video()
                self.screen_history = []
                # plt.plot(self.total_reward_history)
                # plt.savefig('reward_history{}.png'.format(self.config.game_name))

        self.save_model()
