# lorenz96

** Generate a single output **
```bash
python l96_cli.py --N 40 --F 8 --tmax 20 --steps 600 --method RK4FIXED \
  --seed 123 --ic-mode random \
  --save-npy out --save-x0 out/x0.npy --save-config out/run.json --no-show
```

** Visualize output into 4 plots **
```bash
python visualize_l96.py --path out
```
