# PASF

## Installation

This instruction assumes that you already have a conda installation. 

1. Prepare the environmental paths for Mujoco. Add these lines to the `~/.bashrc`:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/$USER/.mujoco/mujoco200/bin
export MUJOCO_PY_MJPRO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mujoco200/mjkey.txt
```   
Source the bashrc file:

```
source ~/.bashrc
```

I copied the Mujoco 2.0 over to my own directory as well:

```
mkdir ~/.mujoco
cp -r /pkgs/mujoco200 ~/.mujoco/
```
   
2. Install and use the included Ananconda environment
```
$ conda env create -f environment/rlkit_env.yml
$ source activate rlkit_env
```
This Anaconda environment use MuJoCo 2.0 and gym 0.15.7.

It may be easier to install everything instead of `mujoco_py` first and then to install with:
```
pip install --no-cache-dir --upgrade  'mujoco-py<2.1,>=2.0'
```

You can check that the `mujoco_py` was installed properly 

```
(rlkit_env) harris@vremote:~/Fair-SkewFit$ python
Python 3.7.4 (default, Aug 13 2019, 20:35:49)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mujoco_py
>>> import gym
>>> gym.make('Hopper-v2')
/h/harris/miniconda/envs/rlkit_env/lib/python3.7/site-packages/gym/logger.py:30: UserWarning: WARN: Box bound precision lowered by casting to float32
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
<TimeLimit<HopperEnv<Hopper-v2>>>
```
3. Make sure that there is a `data` folder under `Fair-SkewFit`

4. Create a `slrm_trash_log` folder in your home directory (i.e. same level as `Fair-SkewFit`). We will store the output and error log files in these folders

5. Test to see if this environment works by running a trial bash script. From the `Fair-SkewFit` directory, run:

```
source slurm/mmd_constant_launcher_test.sh
```

which will submit a single job to run one experiment. Check if the job lands and does not error out. 
There will be:

1. An error and an output log file in the `~/slrm_trash_log/` folder 
2. Checkpoint folders in `/checkpoint/$USER/####/`. The results will then be copied to the `~/Fair-SkewFit/data` folder. 
