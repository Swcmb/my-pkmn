@echo off
cd /d "D:\pkmn-kaggle"
set PYTHONPATH=C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim
set PYTHONUNBUFFERED=1
set TRAIN_PROF=1
"D:\ProgramFiles\deeplearning\python.exe" -m rl.train_selfplay --decks sample --num-envs 16 --num-steps 64 --num-minibatches 16 --total-timesteps 40960 --log-every 1 --seed 1 --d-model 544 --nhead 8 --nlayers 5 --out D:\pkmn-kaggle\runs\bench-20m\E16\checkpoint
set RC=%ERRORLEVEL%
> "D:\pkmn-kaggle\runs\bench-20m\E16\exit_E16_20261017_215135.txt" echo %RC%
