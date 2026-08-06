# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
import sys
import inspect
from pathlib import Path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default

# Persist artefacts at the repository root, independent of the launch directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs"

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

MAX_STEPS = 400

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

USE_CWND = False

class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate, rng):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.rng = rng

    def get_cur_queue_delay(self, event_time):
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (self.rng.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        if extra_delay + self.queue_delay > self.max_queue_delay:
            return False
        self.queue_delay += extra_delay
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    def __init__(self, senders, links):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                    else:
                        sender.on_packet_acked(cur_latency)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= self.rand.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1
                
                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= self.rand.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)
                   
            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5

        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)

        return reward * REWARD_SCALE

class Sender():
    
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

    _next_id = 1
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gymnasium.Env):
    
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="sent latency inflation,"
                          + "latency ratio,"
                          + "send ratio"),
                bandwidth=200,
                latency=0.03,
                queue=5,
                loss=0.0,
                step_bandwidth=None,
                scenario_name="default",
                model_type=None,
                seed_id=0,
                phase="eval",
                ):
        # network configuration
        self.bandwidth = bandwidth
        self.latency = latency
        self.queue = queue
        self.loss = loss
        self.step_bandwidth = step_bandwidth
        self.scenario_name = scenario_name
        self.seed_id = seed_id
        self.model_type = model_type

        self.viewer = None
        self.rand = None
        self.seed()

        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.history_len = history_len
        print("History length: %d" % history_len)
        self.features = features.split(",")
        print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.observation_space = None
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        # --- logging state ---
        #
        # Two distinct log types, kept in separate subdirectories:
        #
        #   logs/train/  — compact episode-level summaries written during training.
        #                  One line per episode: episode, reward_sum, reward_ewma.
        #                  ~200 KB per (model_type, seed) across all 8 workers.
        #                  Used to plot training convergence curves.
        #
        #   logs/eval/   — full per-step JSONL written during evaluation only.
        #                  Identical to the original format. Fed into the stats pipeline.
        #
        # Per-step events are never buffered during training, so _seal_episode
        # is a no-op for train-phase envs (buffer always empty).
        #
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        self.phase = phase
        self.current_episode_events = []
        self.episodes_run = -1

        (LOG_ROOT / "train").mkdir(parents=True, exist_ok=True)
        (LOG_ROOT / "eval").mkdir(parents=True, exist_ok=True)

        if self.phase == "train":
            # Compact episode-level training log — cleared on fresh run.
            self.train_log_filename = str(
                LOG_ROOT / "train" / f"{self.model_type}_seed_{self.seed_id}.jsonl"
            )
            if os.path.exists(self.train_log_filename):
                os.remove(self.train_log_filename)
            self.log_filename = None   # no per-step eval log for train envs
        else:
            # Full per-step evaluation log — cleared on fresh run.
            self.log_filename = str(
                LOG_ROOT / "eval" / f"{self.model_type}_{self.scenario_name}_seed_{self.seed_id}.jsonl"
            )
            if os.path.exists(self.log_filename):
                os.remove(self.log_filename)
            self.train_log_filename = None   # no training log for eval envs

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        return sender_obs

    def step(self, actions):
        for i in range(0, 1):
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action[1])

        reward = self.net.run_for_dur(self.run_dur)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1

        # step-bandwidth change during evaluation episodes
        if self.step_bandwidth is not None and self.steps_taken == int(self.max_steps / 2):
            for link in self.links:
                link.bw = self.step_bandwidth
            print("Bandwidth step change:", self.step_bandwidth)
        
        sender_obs = self._get_all_sender_obs()
        sender_mi = self.senders[0].get_run_data()

        event = {
            "Name":               "Step",
            "Time":               int(self.steps_taken),
            "Reward":             float(reward),
            "Send Rate":          float(sender_mi.get("send rate")),
            "Throughput":         float(sender_mi.get("recv rate")),
            "Latency":            float(sender_mi.get("avg latency")),
            "Loss Rate":          float(sender_mi.get("loss ratio")),
            "Latency Inflation":  float(sender_mi.get("sent latency inflation")),
            "Latency Ratio":      float(sender_mi.get("latency ratio")),
            "Send Ratio":         float(sender_mi.get("send ratio")),
        }
        # Buffer this step only during evaluation.
        # During training the buffer stays empty — _seal_episode becomes a no-op,
        # and no JSONL data is ever written to disk.
        if self.phase == "eval":
            self.current_episode_events.append(event)

        if event["Latency"] > 0.0:
            self.run_dur = 0.5 * sender_mi.get("avg latency")

        should_stop = False
        self.reward_sum += reward
        terminated = (self.steps_taken >= self.max_steps or should_stop)
        truncated = False
             
        return sender_obs, reward, terminated, truncated, {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw    = self.bandwidth
        lat   = self.latency
        queue = self.queue
        loss  = self.loss
        self.links = [Link(bw, lat, queue, loss, self.rand),
                      Link(bw, lat, queue, loss, self.rand)]
        self.senders = [Sender(self.rand.uniform(0.3, 1.5) * bw,
                               [self.links[0], self.links[1]], 0,
                               self.features, history_len=self.history_len)]
        self.run_dur = 3 * lat

    def _write_training_summary(self, reward_sum, reward_ewma):
        """Append a one-line episode summary to the compact training log.

        Called from reset() after each completed training episode, using the
        reward values captured before reward_sum is zeroed.  Each line is a
        small JSON object — the full log for one worker is ~25 KB regardless
        of episode length, vs ~450 KB per-step for the same worker.
        """
        record = {
            "episode":      self.episodes_run,
            "reward_sum":   round(reward_sum,  6),
            "reward_ewma":  round(reward_ewma, 6),
        }
        with open(self.train_log_filename, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _seal_episode(self):
        """Write the completed episode to the JSONL log file and clear the buffer.

        During training (phase='train') the buffer is always empty and
        log_filename is None, so this method returns immediately — no disk I/O.
        During evaluation (phase='eval') each episode is one JSON object on
        its own line (JSONL format). Write cost is O(1) — only the current
        episode is serialised, never the full history.
        """
        if not self.current_episode_events:
            return  # train phase: always empty; eval phase: interrupted episode

        episode_data = {
            "Scenario": self.scenario_name,
            "Seed":     self.seed_id,
            "Episode":  self.episodes_run,      # 1-indexed after increment in reset()
            "Steps":    self.current_episode_events,
        }
        with open(self.log_filename, 'a') as f:
            f.write(json.dumps(episode_data) + '\n')

        # clear buffer — episode is now safely on disk
        self.current_episode_events = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1

        # seal the episode that just finished.
        # episodes_run == 0 means this is the very first reset — no prior episode exists.
        if self.episodes_run > 0:
            self._seal_episode()   # writes to disk, clears buffer — O(1)

        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)

        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))

        # Write compact episode summary for training convergence plots.
        # Must happen after ewma is updated but before reward_sum is zeroed.
        if self.phase == "train" and self.episodes_run > 0:
            self._write_training_summary(self.reward_sum, self.reward_ewma)

        self.reward_sum = 0.0
    
        return self._get_all_sender_obs(), {}

    def seal_final_episode(self):
        """Seal the last episode after evaluation ends.

        The last episode is never followed by a reset() call, so Train.py
        must call this explicitly after the final evaluation episode completes.
        """
        self.episodes_run += 1   # count the final episode
        self._seal_episode()

    def get_log_filename(self):
        """Return the JSONL log file path for this environment instance.

        Train.py uses this to locate the file after evaluation completes.
        """
        return self.log_filename

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

register(id='PccNs-v0', entry_point='network_sim:SimulatedNetworkEnv')
