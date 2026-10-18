@echo off
cd /d "D:\pkmn-kaggle"
set PYTHONPATH=C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim
set PYTHONUNBUFFERED=1
set TRAIN_PROF=1
"D:\ProgramFiles\deeplearning\python.exe" -m rl.train_selfplay --decks sample --num-envs 32 --num-steps 64 --num-minibatches 16 --total-timesteps 81920 --log-every 1 --seed 1 --d-model 544 --nhead 8 --nlayers 5 --out D:\pkmn-kaggle\runs\bench-20m\E32\checkpoint
set RC=%ERRORLEVEL%
> "D:\pkmn-kaggle\runs\bench-20m\E32\exit_E32_20261017_215958.txt" echo %RC%
