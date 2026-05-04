# This file is based on code from https://github.com/Aqasch/TensorRL-QAS,
# licensed under the Apache License, Version 2.0.
# Modifications have been made by Alexandra Xiulan Bates, 2026.

import numpy as np
import random
import torch
import sys
import os
import argparse
import pathlib
import copy
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from environments.utils.utils import get_config
from environments.environment_qulacs import CircuitEnv
from environments.VQAs import VQE_qulacs as vc
import agents
import time
from qulacs import QuantumState

class Saver:
    def __init__(self, results_path, experiment_seed, timestamp, ham_stem, num_layers):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path
        self.timestamp = timestamp
        self.ham_stem = ham_stem
        self.num_layers = num_layers

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss': [],
                                                 'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'opt_ang': [],
                                                 'time' : [],
                                                'save_circ' : [],
                                                'reward' : []
                                                 }
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time' : []
                                                 }

    def save_file(self):
        np.save(f'{self.rpath}/summary_{self.ham_stem}_L{self.num_layers}_{self.exp_seed}_{self.timestamp}.npy', self.stats_file)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

    
def modify_state(state,env):
    
        
    if conf['agent']['en_state']:
        
        state = torch.cat((state, torch.tensor(env.prev_energy,dtype=torch.float,device=device).view(1)))
        
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        state = torch.cat((state, torch.tensor(env.done_threshold,dtype=torch.float,device=device).view(1)))
         
    return state


def snapshot_circuit_metrics(env, n_samples):
    """Build a snapshot of the current best circuit, its parameters and shot statistics."""
    state_snapshot = env.state.detach().cpu().clone()

    qulacs_inst = vc.Parametric_Circuit(
        n_qubits=env.num_qubits,
        noise_models=env.noise_models,
        noise_values=env.noise_values,
    )
    circuit = qulacs_inst.construct_ansatz(state_snapshot)
    avg_energy = float(vc.get_exp_val(env.num_qubits, circuit, env.hamiltonian))

    sim_state = QuantumState(env.num_qubits)
    circuit.update_quantum_state(sim_state)
    samples = np.array(sim_state.sampling(int(n_samples)), dtype=np.int64)
    counts = np.bincount(samples, minlength=2 ** env.num_qubits)

    best_state_idx = int(np.argmax(counts))
    best_state_prob = float(counts[best_state_idx] / max(1, samples.size))
    best_state_bits = format(best_state_idx, f"0{env.num_qubits}b")

    theta_snapshot = state_snapshot[:, env.num_qubits + 3 :, :].numpy()

    return {
        'state_tensor': state_snapshot.numpy(),
        'theta_tensor': theta_snapshot,
        'average_energy': avg_energy,
        'most_likely_state_index': best_state_idx,
        'most_likely_state_bitstring': best_state_bits,
        'most_likely_state_probability': best_state_prob,
        'n_samples': int(n_samples),
    }


def compact_best_summary(best_tracker):
    """Convert best tracker to a JSON-safe summary."""
    if best_tracker['episode'] is None:
        return {
            'found': False,
            'message': 'No circuit snapshot was collected during training.',
        }

    return {
        'found': True,
        'episode': int(best_tracker['episode']),
        'step': int(best_tracker['step']),
        'is_initial_circuit': bool(best_tracker.get('is_initial_circuit', False)),
        'error': float(best_tracker['error']),
        'reward': float(best_tracker['reward']),
        'average_energy': float(best_tracker['average_energy']),
        'most_likely_state_index': int(best_tracker['most_likely_state_index']),
        'most_likely_state_bitstring': best_tracker['most_likely_state_bitstring'],
        'most_likely_state_probability': float(best_tracker['most_likely_state_probability']),
        'n_samples': int(best_tracker['n_samples']),
        'num_actions': len(best_tracker['actions_so_far']),
    }


def evaluate_initial_circuit(env):
    """Optionally optimize initial parameters before any RL action and refresh metrics."""
    nfev = 0
    opt_ang = 0
    if getattr(env, 'optim_method', None) in ["scipy_each_step"]:
        thetas, nfev, opt_ang = env.scipy_optim(env.optim_alg)
        for i in range(env.num_layers):
            for j in range(3):
                env.state[i][env.num_qubits + 3 + j, :] = thetas[i][j, :]

    energy, energy_noiseless = env.get_energy()
    if env.noise_flag is False:
        energy = energy_noiseless

    env.energy = energy
    env.prev_energy = np.copy(energy)
    env.error = float(abs(env.min_eig - energy))
    env.error_noiseless = float(abs(env.min_eig - energy_noiseless))
    env.rwd = 0.0
    env.nfev = nfev
    env.opt_ang_save = opt_ang


def maybe_update_best(best_tracker, env, episode_no, step_no, actions_so_far, reward_value, best_samples, is_initial_circuit=False):
    """Track the circuit with the lowest average energy seen so far."""
    if env.energy < best_tracker['average_energy']:
        snap = snapshot_circuit_metrics(env, best_samples)
        best_tracker.update({
            'episode': int(episode_no),
            'step': int(step_no),
            'is_initial_circuit': bool(is_initial_circuit),
            'error': float(env.error),
            'reward': float(reward_value),
            'actions_so_far': copy.deepcopy(actions_so_far),
            'state_tensor': snap['state_tensor'],
            'theta_tensor': snap['theta_tensor'],
            'average_energy': snap['average_energy'],
            'most_likely_state_index': snap['most_likely_state_index'],
            'most_likely_state_bitstring': snap['most_likely_state_bitstring'],
            'most_likely_state_probability': snap['most_likely_state_probability'],
            'n_samples': snap['n_samples'],
        })


def agent_test(env, agent, episode_no, seed, output_path, threshold, timestamp, ham_stem, num_layers):
    """ Testing function of the trained agent. """    
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env)
    current_epsilon = copy.copy(agent.epsilon)
    agent.policy_net.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        agent.epsilon = 0
        with torch.no_grad():
            action, _ = agent.act(state, ill_action_from_env)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action],train_flag=False)
        next_state = modify_state(next_state, env)
        state = next_state.clone()
        assert type(env.error) == float 
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        
        if done:
            
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            agent.saver.stats_file['test'][episode_no]['bond_distance'] = env.current_bond_distance
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{ham_stem}_L{num_layers}_{seed}_{timestamp}_best_geo_{env.current_bond_distance}_model.pth")
                torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{ham_stem}_L{num_layers}_{seed}_{timestamp}_best_geo_{env.current_bond_distance}_optim.pth")
            agent.epsilon = current_epsilon
            agent.saver.validate_stats(episode_no, 'test')
            
            return reward, t
        

def one_episode(episode_no, env, agent, episodes, best_tracker, best_samples):
    """ Function preforming full trainig episode."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    agent.saver.stats_file['train'][episode_no]['bond_distance'] = env.current_prob
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env)

    # Seed best tracking with the pre-RL circuit state (human/TN init if enabled,
    # otherwise the default initialized ansatz), after classical angle optimization.
    evaluate_initial_circuit(env)
    maybe_update_best(
        best_tracker=best_tracker,
        env=env,
        episode_no=episode_no,
        step_no=-1,
        actions_so_far=[],
        reward_value=env.rwd,
        best_samples=best_samples,
        is_initial_circuit=True,
    )

    agent.policy_net.train()
    rewards4return = []
    episode_energies = []
    
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        action, _ = agent.act(state, ill_action_from_env)
        assert type(action) == int
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        
        next_state, reward, done = env.step(agent.translate[action])
        
        next_state = modify_state(next_state, env)
        agent.remember(state, 
                       torch.tensor(action, device=device), 
                       reward,
                       next_state,
                       torch.tensor(done, device=device))
        state = next_state.clone()
        rewards4return.append(float(reward.clone()))

        episode_energies.append(float(env.energy))

        assert type(env.error) == float
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['train'][episode_no]['opt_ang'].append(env.opt_ang_save)
        agent.saver.stats_file['train'][episode_no]['save_circ'].append(env.save_circ)
        agent.saver.stats_file['train'][episode_no]['nfev'].append(env.nfev)
        agent.saver.stats_file['train'][episode_no]['reward'].append(env.rwd)
        
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time()-t0)

        maybe_update_best(
            best_tracker=best_tracker,
            env=env,
            episode_no=episode_no,
            step_no=itr,
            actions_so_far=agent.saver.stats_file['train'][episode_no]['actions'],
            reward_value=env.rwd,
            best_samples=best_samples,
            is_initial_circuit=False,
        )

        if agent.memory_reset_switch:            
           if env.error < agent.memory_reset_threshold:
               agent.memory_reset_counter += 1
           if agent.memory_reset_counter == agent.memory_reset_switch:
               agent.memory.clean_memory()
               agent.memory_reset_switch = False
               agent.memory_reset_counter = False
               
  
        if done:
            episode_time = time.time() - t0
            print('time:', episode_time)
            if episode_no%1==0:
                print("episode: {}/{}, score: {}, e: {:.2}, rwd: {} \n"
                        .format(episode_no, episodes, itr, agent.epsilon, reward),flush=True)
            return {
                'final_energy': episode_energies[-1],
                'min_energy': min(episode_energies),
                'wall_clock_time': episode_time,
            }
        
        if len(agent.memory) > conf['agent']['batch_size']:
            if "replay_ratio" in conf['agent'].keys():
                if  itr % conf['agent']["replay_ratio"]==0:
                    loss = agent.replay(conf['agent']['batch_size'])
            else:
                loss = agent.replay(conf['agent']['batch_size'])         
            assert type(loss) == float
            agent.saver.stats_file['train'][episode_no]['loss'].append(loss)
            agent.saver.validate_stats(episode_no, 'train')

    # If episode ended without done (shouldn't normally happen), return what we have
    episode_time = time.time() - t0
    return {
        'final_energy': episode_energies[-1] if episode_energies else float('nan'),
        'min_energy': min(episode_energies) if episode_energies else float('nan'),
        'wall_clock_time': episode_time,
    }


def train(agent, env, episodes, seed, output_path, threshold, best_samples, timestamp, ham_stem, num_layers):
    """Training loop"""
    best_tracker = {
        'episode': None,
        'step': None,
        'is_initial_circuit': False,
        'error': None,
        'reward': None,
        'actions_so_far': [],
        'state_tensor': None,
        'theta_tensor': None,
        'average_energy': float('inf'),
        'most_likely_state_index': None,
        'most_likely_state_bitstring': None,
        'most_likely_state_probability': None,
        'n_samples': int(best_samples),
    }

    episode_log = []

    for e in range(episodes):
        
        ep_stats = one_episode(e, env, agent, episodes, best_tracker, best_samples)
        episode_log.append({
            'episode': e,
            'final_energy': ep_stats['final_energy'],
            'min_energy': ep_stats['min_energy'],
            'wall_clock_time': ep_stats['wall_clock_time'],
        })
        
        if e %5==0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{ham_stem}_L{num_layers}_{seed}_{timestamp}_model.pth")
            torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{ham_stem}_L{num_layers}_{seed}_{timestamp}_optim.pth")
            torch.save( {i: a._asdict() for i,a in enumerate(agent.memory.memory)}, f"{output_path}/thresh_{threshold}_{ham_stem}_L{num_layers}_{seed}_{timestamp}_replay_buffer.pth")

    return best_tracker, episode_log

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, required=True, help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='TensorRL_trainable/', help='Name of experiment (default: TensorRL_trainable/)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Execution device: auto, cpu, or cuda')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index to use when device resolves to cuda')
    parser.add_argument('--cpu_threads', type=int, default=1, help='Number of CPU threads for PyTorch (default: 1)')
    parser.add_argument('--best_samples', type=int, required=True, help='Number of shots for best-circuit sampling statistics')
    parser.add_argument('--circuit', type=str, required=True, help='Path to initial circuit .qpy file')
    parser.add_argument('--hamiltonian', type=str, required=True, help='Path to Hamiltonian .npz file')
    parser.add_argument('--outputdir', type=str, required=True, help='Base directory for outputs; files are saved under outputdir/experiment_name/config/')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    full_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = get_args(sys.argv[1:])

    if args.cpu_threads <= 0:
        raise ValueError("--cpu_threads must be a positive integer")
    if not args.circuit.endswith('.qpy'):
        raise ValueError(f"--circuit must be a .qpy file.")
    if not args.hamiltonian.endswith('.npz'):
        raise ValueError(f"--hamiltonian must be a .npz file.")
    torch.set_num_threads(args.cpu_threads)
    print(f"Using {args.cpu_threads} CPU thread(s) for PyTorch")
    ham_stem = os.path.splitext(os.path.basename(args.hamiltonian))[0]


    results_path = args.outputdir
    # Normalize --config: accept either a name stem or a full file path.
    # If a path is given, extract the stem for result dirs and remember the full path for loading.
    _config_full_path = None
    if '/' in args.config or args.config.endswith('.cfg'):
        _config_full_path = args.config if args.config.endswith('.cfg') else args.config + '.cfg'
        args.config = os.path.splitext(os.path.basename(args.config))[0]
    ids = [i for i, v in enumerate(args.config) if v == "_"]
    results_path_to_reload = f'results/finalize/{args.config}/'
    output_dir = os.path.join(results_path, args.experiment_name.rstrip('/'), ham_stem)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_cuda = False
    if args.device == 'cuda':
        use_cuda = True
    elif args.device == 'auto' and torch.cuda.is_available():
        use_cuda = True

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu_id must be in [0, {torch.cuda.device_count() - 1}] for this machine"
            )
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU (cuda:{args.gpu_id})")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if _config_full_path is not None:
        _cfg_dir = os.path.dirname(_config_full_path) or '.'
        _cfg_base = os.path.basename(_config_full_path)
        conf = get_config('', _cfg_base, path=_cfg_dir)
    else:
        conf = get_config(args.experiment_name, f'{args.config}.cfg')

    num_layers = conf['env']['num_layers']
    print(f"num_layers: {num_layers}")

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    

    actions_test = []
    action_test_dict = dict()
    error_test_dict = dict()
    error_noiseless_test_dict=dict()

    
    """ Environment and Agent initialization"""
    environment = CircuitEnv(conf, device=device, circuit_path=args.circuit, hamiltonian_path=args.hamiltonian)
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, environment.action_size, environment.state_size, device)
    agent.saver = Saver(output_dir, args.seed, timestamp, ham_stem, conf['env']['num_layers'])

    if conf['agent']['init_net']: 
        PATH = f"{results_path_to_reload}thresh_{conf['env']['accept_err']}_{args.seed}"
        agent.policy_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.target_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.optim.load_state_dict(torch.load(PATH+f"_optim.pth"))
        agent.policy_net.eval()
        agent.target_net.eval()

        replay_buffer_load = torch.load(f"{PATH}_replay_buffer.pth")
        for i in replay_buffer_load.keys():
            agent.remember(**replay_buffer_load[i])

        if not conf['agent']['epsilon_restart']:
            agent.epsilon = agent.epsilon_min

    best_tracker, episode_log = train(agent, environment, conf['general']['episodes'], args.seed, output_dir, conf['env']['accept_err'], args.best_samples, timestamp, ham_stem, num_layers)
    agent.saver.save_file()
    torch.save(agent.policy_net.state_dict(), f"{output_dir}/thresh_{conf['env']['accept_err']}_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}_model.pth")
    torch.save(agent.optim.state_dict(), f"{output_dir}/thresh_{conf['env']['accept_err']}_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}_optim.pth")

    best_path_base = f"{output_dir}/best_circuit_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}"
    np.save(f"{best_path_base}.npy", best_tracker, allow_pickle=True)
    with open(f"{best_path_base}.json", 'w') as f:
        json.dump(compact_best_summary(best_tracker), f, indent=2)

    final_snap = snapshot_circuit_metrics(environment, args.best_samples)
    final_path_base = f"{output_dir}/final_circuit_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}"
    np.save(f"{final_path_base}.npy", final_snap, allow_pickle=True)
    with open(f"{final_path_base}.json", 'w') as f:
        json.dump({
            'average_energy': final_snap['average_energy'],
            'most_likely_state_index': final_snap['most_likely_state_index'],
            'most_likely_state_bitstring': final_snap['most_likely_state_bitstring'],
            'most_likely_state_probability': final_snap['most_likely_state_probability'],
            'n_samples': final_snap['n_samples'],
        }, f, indent=2)

    # Save per-episode energy CSV
    csv_path = f"{output_dir}/expectation_value_per_episode_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'final_energy', 'min_energy', 'wall_clock_time'])
        writer.writeheader()
        writer.writerows(episode_log)
    print(f"Expectation value CSV saved to {csv_path}")

    # Plot energy vs episode
    episodes_x = [row['episode'] for row in episode_log]
    final_energies = [row['final_energy'] for row in episode_log]
    min_energies = [row['min_energy'] for row in episode_log]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes_x, final_energies, label='Final Expectation Value', alpha=0.8)
    ax.plot(episodes_x, min_energies, label='Min. Expectation Value', alpha=0.8)
    ax.axhline(y=environment.min_eig, color='r', linestyle='--', label=f'Ground state ({environment.min_eig:.4f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Expectation Value')
    ax.set_title(f'Expectation Value per Episode, {ham_stem}: L={num_layers}, thresholds={conf["env"]["thresholds"]}, eps_decay={conf["agent"]["epsilon_decay"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = f"{output_dir}/expectation_value_per_episode_{ham_stem}_L{num_layers}_{args.seed}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Expectation value plot saved to {plot_path}")

    full_end_time = time.time()
    total_runtime = full_end_time - full_start_time
    print(f"Total script runtime: {total_runtime:.2f} seconds")
