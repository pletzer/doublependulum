# doublependulum
Solve the double pendulum equations

Point masses `m1` (at the joint) and `m2` (at the tip) on massless rigid arms
`l1`, `l2`; angles measured from the downward vertical. Integrated with
`scipy.integrate.solve_ivp` (DOP853 by default); energy drift is reported as an
accuracy check.

```bash
python doublependulum.py --theta1 120 --theta2 -10 --tmax 20
python doublependulum.py --animate
python doublependulum.py --no-show --save summary.png
python doublependulum.py --help
```

Requires numpy, scipy, matplotlib. `m1` must be > 0 (the equations are singular
for a massless joint when the arms align).
